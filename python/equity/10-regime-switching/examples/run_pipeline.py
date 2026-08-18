"""End-to-end regime-switching pipeline on the synthetic 3-state panel.

Reproduces every number quoted in README.md and docs/VALIDATION.md:

1.  Generate a seeded 3-state regime panel (bull / transition / bear) with a
    KNOWN transition matrix and per-state mean / vol / correlation.
2.  Build the point-in-time feature table (expanding z-scores, no lookahead).
3.  PCA from scratch: scree table and loading interpretation of PC1/PC2.
4.  k-selection by BIC on the index returns (should recover 3).
5.  Full-sample HMM fit: recovered transition matrix and state parameters
    vs the truth table (permutation-matched), stationary distribution and
    expected regime durations.
6.  Filtered vs smoothed probabilities around one true regime transition —
    the honesty table: smoothed "sees" the turn before it happens.
7.  Average regime-detection lag in days (regimes are detected LATE by
    construction).
8.  Consequences of fitting k=2 on a 3-state world.
9.  Regime timeline summary from walk-forward detection.
10. Walk-forward strategy vs buy-and-hold and the 200d-MA rule, net of
    costs, plus per-regime risk report, transition attribution and
    flip-aftermath analysis.

Runs offline in well under 150 seconds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from eq_regime import (
    build_features,
    expected_durations,
    fit_hmm,
    fit_pca,
    flip_aftermath,
    match_permutation,
    per_regime_stats,
    regime_runs,
    scree_table,
    select_k_bic,
    stationary_distribution,
    summary_stats,
    transition_attribution,
    walk_forward_backtest,
)
from eq_regime.data import make_regime_panel

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda v: f"{v:10.4f}")

T0 = time.time()


def hdr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}   [t={time.time() - T0:5.1f}s]\n{'=' * 78}")


# ----------------------------------------------------------------- 1. data
hdr("1. Synthetic 3-state regime panel (seeded, ground truth known)")
panel = make_regime_panel(n_states=3, n_assets=8, n_days=2520, seed=7)
r_idx = panel.returns.mean(axis=1)
print(f"panel: {panel.returns.shape[0]} days x {panel.returns.shape[1]} assets")
truth = pd.DataFrame(
    {
        "mu_ann": panel.mu,
        "sigma_ann": panel.sigma,
        "corr": panel.corr,
        "p_stay": np.diag(panel.transition),
        "E[duration]d": expected_durations(panel.transition),
        "days_in_state": [int((panel.states == k).sum()) for k in range(3)],
    },
    index=["state0 (bull)", "state1 (transition)", "state2 (bear)"],
)
print(truth)

# ------------------------------------------------------------- 2. features
hdr("2. Point-in-time feature table (expanding z-scores, no lookahead)")
features = build_features(panel.prices)
print(f"features: {features.shape[0]} days x {features.shape[1]} columns "
      f"(warmup rows dropped, zero NaN: {int(features.isna().sum().sum())})")
print(features.tail(3))

# ------------------------------------------------------------------ 3. PCA
hdr("3. PCA from scratch — scree and loadings")
pca = fit_pca(features, n_components=4)
print(scree_table(pca))
loadings = pd.DataFrame(
    pca.components, index=features.columns, columns=[f"PC{i+1}" for i in range(4)]
)
print("\nLoadings (sign convention: largest |loading| positive):")
print(loadings)
print(
    "\nInterpretation: PC1 loads with one sign on all vol measures, dispersion,"
    "\navg correlation, drawdown and credit proxy, and opposite on trend/term —"
    "\na single 'market stress' factor. PC2 separates trend from slow vol:"
    "\na 'regime maturity' factor."
)

# ----------------------------------------------------- 4. k selection (BIC)
hdr("4. GMM k-selection by BIC on daily index returns")
best_k, scores = select_k_bic(
    r_idx.to_numpy(), k_range=(1, 2, 3, 4, 5), seed=0, n_init=3
)
print(pd.Series(scores, name="BIC").rename_axis("k").to_frame())
print(f"BIC-selected k = {best_k}  (true number of states: 3)")

# --------------------------------------------------------- 5. HMM recovery
hdr("5. Full-sample 3-state HMM fit — recovery vs truth (permutation-matched)")
hfit = fit_hmm(r_idx.to_numpy(), 3, seed=0, n_init=3, max_iter=200)
true_mu_d = panel.mu / 252.0 - 0.5 * (panel.sigma / np.sqrt(252.0)) ** 2
perm = match_permutation(true_mu_d[:, None] * 1e4, hfit.means * 1e4)
est_mu = np.array([hfit.means[perm[k], 0] for k in range(3)])
est_sig = np.array([np.sqrt(hfit.covariances[perm[k], 0, 0]) for k in range(3)])
est_p = hfit.transmat[np.ix_(perm, perm)]
recov = pd.DataFrame(
    {
        "true_mu_ann": panel.mu,
        "est_mu_ann": est_mu * 252 + 0.5 * (est_sig**2) * 252,
        "true_sigma_ann": panel.sigma,
        "est_sigma_ann": est_sig * np.sqrt(252),
        "true_p_stay": np.diag(panel.transition),
        "est_p_stay": np.diag(est_p),
        "true_E[dur]d": expected_durations(panel.transition),
        "est_E[dur]d": expected_durations(est_p),
    },
    index=["bull", "transition", "bear"],
)
print(recov)
print("\nEstimated transition matrix (rows: from, cols: to, truth-ordered):")
print(pd.DataFrame(est_p, index=["bull", "trans", "bear"], columns=["bull", "trans", "bear"]))
pi = stationary_distribution(est_p)
print(f"\nStationary distribution: {np.round(pi, 3)}  "
      f"(empirical occupancy: {np.round([np.mean(panel.states == k) for k in range(3)], 3)})")
print(f"log-likelihood: {hfit.log_likelihood:.1f}  (EM iterations: {hfit.n_iter}, "
      f"monotone: {bool(np.all(np.diff(hfit.log_likelihood_history) >= -1e-8))})")

# --------------------------------- 6. filtered vs smoothed around one turn
hdr("6. Filtered vs smoothed bear probability around one true bull->bear turn")
# label on emission VOL for 1-D returns: bear = highest emission variance
vols = np.sqrt(hfit.covariances[:, 0, 0])
bear_state = int(np.argmax(vols))
filt, _ = hfit.filter(r_idx.to_numpy())
smth = hfit.smooth(r_idx.to_numpy())
# first true entry into the bear state after day 300
flips = np.where((panel.states[1:] == 2) & (panel.states[:-1] != 2))[0] + 1
t_flip = int(flips[flips > 300][0])
window = slice(t_flip - 5, t_flip + 8)
tbl = pd.DataFrame(
    {
        "true_state": panel.states[window],
        "daily_ret_%": 100 * r_idx.to_numpy()[window],
        "P_bear_filtered": filt[window, bear_state],
        "P_bear_smoothed": smth[window, bear_state],
    },
    index=[f"t{d:+d}" for d in range(-5, 8)],
)
print(f"true transition into bear at index {t_flip} ({r_idx.index[t_flip].date()})")
print(tbl)
print(
    "\nNote how the SMOOTHED probability rises BEFORE the flip (it has seen"
    "\nthe future) while the FILTERED probability reacts only after evidence"
    "\narrives. Trading on smoothed probabilities is lookahead bias."
)

# ------------------------------------------------------- 7. detection lag
hdr("7. Average detection lag (filtered argmax vs true state)")
filt_path = filt.argmax(axis=1)
# map estimated state indices onto truth ordering
inv = np.empty(3, dtype=int)
for k in range(3):
    inv[perm[k]] = k
mapped = inv[filt_path]
lags_bear, lags_exit = [], []
for f in np.where((panel.states[1:] == 2) & (panel.states[:-1] != 2))[0] + 1:
    hit = np.where(mapped[f : f + 40] == 2)[0]
    if len(hit):
        lags_bear.append(int(hit[0]))
for f in np.where((panel.states[1:] != 2) & (panel.states[:-1] == 2))[0] + 1:
    hit = np.where(mapped[f : f + 40] != 2)[0]
    if len(hit):
        lags_exit.append(int(hit[0]))
print(f"bear ENTRY detection lag: mean {np.mean(lags_bear):.1f}d, "
      f"median {np.median(lags_bear):.0f}d over {len(lags_bear)} true entries")
print(f"bear EXIT  detection lag: mean {np.mean(lags_exit):.1f}d, "
      f"median {np.median(lags_exit):.0f}d over {len(lags_exit)} true exits")
print("Regime changes are detected LATE by construction: the filter needs")
print("several days of evidence before the posterior crosses over.")

# ---------------------------------------- 8. k=2 on a 3-state world
hdr("8. Mis-specified k=2 fit on the 3-state world")
h2 = fit_hmm(r_idx.to_numpy(), 2, seed=0, n_init=3, max_iter=200)
v2 = np.sqrt(h2.covariances[:, 0, 0]) * np.sqrt(252)
print(f"k=2 log-lik {h2.log_likelihood:.1f} vs k=3 log-lik {hfit.log_likelihood:.1f}")
print(f"k=2 state vols (ann): {np.round(np.sort(v2), 3)} — the transition state is"
      f"\nabsorbed into its neighbours; true vols were {panel.sigma}.")

# --------------------------------------------- 9-10. walk-forward strategy
hdr("9. Walk-forward backtest: HMM strategy vs buy-and-hold vs 200d-MA rule")
res = walk_forward_backtest(
    panel.prices,
    n_states=3,
    min_train=378,
    refit_every=63,
    cost_bps=5.0,
    seed=0,
    enter=0.70,
    exit_=0.30,
    target_vol=0.10,
    n_pca=3,
)
stats = pd.DataFrame(
    {
        "strategy": summary_stats(res.ledger),
        "buy_and_hold": summary_stats(res.benchmark),
        "ma_200d": summary_stats(res.ma_rule),
    }
)
print(stats)

print("\nRegime timeline (walk-forward, filtered argmax, first/last 5 runs):")
runs = regime_runs(res.detection["regime"])
print(pd.concat([runs.head(5), runs.tail(5)]))
print(f"total runs: {len(runs)}, median run length: {runs['days'].median():.0f}d")

hdr("10. Regime-conditional risk report (strategy vs benchmark)")
regimes = res.detection["regime"].reindex(res.ledger.index).ffill().bfill()
print("Strategy net returns by detected regime:")
print(per_regime_stats(res.ledger["net_ret"], regimes))
print("\nBuy-and-hold net returns by detected regime:")
print(per_regime_stats(res.benchmark["net_ret"], regimes))
print("\nTransition P&L attribution (strategy):")
print(transition_attribution(res.ledger["net_ret"], regimes))
after = flip_aftermath(res.ledger["net_ret"], regimes, k=10)
print(f"\nFlip-aftermath: cumulative strategy P&L in the 10 days after each of "
      f"{len(after)} regime flips:")
print(after.groupby("to_regime")["pnl_next_10d"].agg(["count", "mean", "min"]))

print(f"\nDone in {time.time() - T0:.1f}s")
