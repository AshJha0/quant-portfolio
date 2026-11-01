"""Null-data guards for the regime overlay, judged on Sharpe.

Two nulls, each over 12 seeds, run in parallel processes:

* No regimes at all (GBM, constant vol): the overlay must not beat random
  timing at equal average exposure. Random timing is the same regime-weight
  path circularly shifted by a random offset, times the same vol scale,
  rescaled to the strategy's mean weight.
* Volatility clustering with no mean-vol link (the project's Markov-switching
  generator with the same 7% drift in every state): vol timing is a real
  effect here, so the overlay is judged against plain vol targeting. It must
  not beat it, because no state has a different expected return.

Bound: the mean Sharpe difference across seeds must be below 3 standard
errors of that mean (one-sided, about a 0.1% false-alarm rate under a
normal approximation). Measured on this configuration (2026-11-01, 12
seeds): GBM strategy minus random timing +0.08 (SE 0.10); equal drift
strategy minus vol targeting -0.01 (SE 0.06); with the planted drifts the
same differences are +0.48 (SE 0.11) and +0.22 (SE 0.08).

The old guard compared CAGR with buy-and-hold, which a de-risking overlay
on a positive-drift asset fails to exceed whatever its skill; it could not
fail for the reason it claimed to guard.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from eq_regime.backtest import walk_forward_backtest
from eq_regime.data import equal_drift_params, make_gbm_panel, make_regime_panel
from eq_regime.gmm import select_k_bic
from eq_regime.research_stats import seed_distribution, sharpe_ratio

BT_KWARGS = dict(
    n_states=2, min_train=300, refit_every=500, cost_bps=5.0, seed=0,
    n_pca=2, detect_kwargs=dict(n_init=1, max_iter=20),
)
SEEDS = tuple(range(1, 13))


def _null_run(job):
    kind, seed = job
    if kind == "gbm":
        panel = make_gbm_panel(n_assets=5, n_days=2500, seed=seed)
    else:
        panel = make_regime_panel(2, 5, 2500, seed=seed, params=equal_drift_params(2, 0.07))
    res = walk_forward_backtest(panel.prices, **BT_KWARGS)
    s = sharpe_ratio(res.ledger["net_ret"].to_numpy())
    other = res.random_timing if kind == "gbm" else res.vol_target
    return s - sharpe_ratio(other["net_ret"].to_numpy())


def _diffs(kind):
    workers = max(1, min(len(SEEDS), (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return np.array(list(ex.map(_null_run, [(kind, s) for s in SEEDS])))


def test_no_timing_edge_over_random_timing_without_regimes():
    d = seed_distribution(_diffs("gbm"))
    assert d.mean < 3.0 * d.se, f"overlay beats random timing on GBM data: {d.fmt(3)}"


def test_no_edge_over_vol_targeting_when_drift_is_state_free():
    d = seed_distribution(_diffs("equal"))
    assert d.mean < 3.0 * d.se, f"overlay beats vol targeting with equal drifts: {d.fmt(3)}"


def test_bic_prefers_fewer_states_on_null_returns():
    """GBM daily returns are i.i.d. Gaussian: BIC must pick 1 component."""
    panel = make_gbm_panel(n_assets=5, n_days=1500, seed=5)
    r = panel.returns.mean(axis=1).to_numpy()
    best_k, scores = select_k_bic(r, k_range=(1, 2, 3), seed=0, n_init=2)
    assert best_k == 1
    assert scores[1] < scores[3]


def test_bic_recovers_states_on_regime_returns(panel3):
    """Contrast: on 3-state returns BIC must prefer >= 2 components over 1."""
    r = panel3.returns.mean(axis=1).to_numpy()
    best_k, scores = select_k_bic(r, k_range=(1, 2, 3, 4), seed=0, n_init=2)
    assert best_k >= 2
    assert scores[1] > min(scores[2], scores[3])
