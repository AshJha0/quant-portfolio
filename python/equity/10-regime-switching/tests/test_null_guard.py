"""Null-data guard: on no-regime GBM data the regime machinery must not
hallucinate structure — no systematic outperformance, and BIC prefers
fewer states."""

from __future__ import annotations

import numpy as np
import pytest

from eq_regime.backtest import summary_stats, walk_forward_backtest
from eq_regime.data import make_gbm_panel
from eq_regime.gmm import select_k_bic

BT_KWARGS = dict(
    n_states=2, min_train=300, refit_every=250, cost_bps=5.0, seed=0,
    n_pca=2, detect_kwargs=dict(n_init=1, max_iter=30),
)


def test_no_spurious_outperformance_on_null_data():
    """Average net CAGR edge of the regime strategy over buy-and-hold across
    seeds must not be meaningfully positive on regime-free data (loose
    statistical test — de-risking on noise should COST money, not make it)."""
    excess = []
    for seed in (1, 2, 3):
        panel = make_gbm_panel(n_assets=5, n_days=900, seed=seed)
        res = walk_forward_backtest(panel.prices, **BT_KWARGS)
        s = summary_stats(res.ledger)
        b = summary_stats(res.benchmark)
        excess.append(s["cagr"] - b["cagr"])
    assert np.mean(excess) < 0.02, (
        f"regime strategy spuriously beats buy-and-hold on null data: {excess}"
    )


def test_bic_prefers_fewer_states_on_null_returns():
    """GBM daily returns are i.i.d. Gaussian: BIC must pick 1 component."""
    panel = make_gbm_panel(n_assets=5, n_days=1500, seed=5)
    r = panel.returns.mean(axis=1).to_numpy()
    best_k, scores = select_k_bic(r, k_range=(1, 2, 3), seed=0, n_init=2)
    assert best_k == 1
    assert scores[1] < scores[3]


def test_bic_recovers_states_on_regime_returns(panel3):
    """Contrast: on genuine 3-state returns BIC must prefer >= 2 components
    over 1 (mixture structure detected)."""
    r = panel3.returns.mean(axis=1).to_numpy()
    best_k, scores = select_k_bic(r, k_range=(1, 2, 3, 4), seed=0, n_init=2)
    assert best_k >= 2
    assert scores[1] > min(scores[2], scores[3])
