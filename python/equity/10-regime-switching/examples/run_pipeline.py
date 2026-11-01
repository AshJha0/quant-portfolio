"""End-to-end regime-switching pipeline on synthetic 3-state panels.

Every number in README.md and docs/VALIDATION.md comes from this script.
All data are synthetic: the bear state's negative drift is planted by the
generator, and no real-data result exists yet.

1-8.  Diagnostics on one seeded panel (seed 7): truth table, features, PCA,
      K selection (GMM-BIC on returns and out-of-sample HMM likelihood),
      HMM recovery, filtered vs smoothed probabilities, detection lag, k=2.
9.    One walk-forward path with Lo/Newey-West SE, a paired block-bootstrap
      CI, deflated Sharpe at the logged trial count and PBO over variants.
10.   Regime-conditional risk report using the regime known at t-1.
11.   Multi-seed Sharpe distributions: 20 fresh seeds per arm, two arms
      (planted drifts; equal 7% drift in every state).
12.   Edge decomposition: planted bear drift vs vol timing.
13.   Execution and min-duration variants.

Run: ``python examples/run_pipeline.py [--workers N]``. Prints only; writes
nothing to disk. About 12 minutes with 10 workers (measured, see README).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from eq_regime import (
    ExecutionConfig,
    backtest_from_detection,
    build_features,
    expected_durations,
    fit_hmm,
    fit_pca,
    flip_aftermath,
    match_permutation,
    per_regime_stats,
    regime_runs,
    regimes_known_at_decision,
    scree_table,
    select_k_bic,
    stationary_distribution,
    summary_stats,
    transition_attribution,
    walk_forward_backtest,
)
from eq_regime.data import equal_drift_params, make_regime_panel
from eq_regime.pca import project
from eq_regime.research_stats import (
    TrialLog,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    lo_sharpe_se,
    newey_west_lag,
    pbo_cscv,
    seed_distribution,
    sharpe_ratio,
)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda v: f"{v:10.4f}")

T0 = time.time()


def hdr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}   [t={time.time() - T0:5.1f}s]\n{'=' * 78}", flush=True)


def diagnostics() -> None:
    """Sections 1-8 on one synthetic panel (seed 7)."""
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
    hdr("3. PCA from scratch : scree and loadings")
    pca = fit_pca(features, n_components=4)
    print(scree_table(pca))
    loadings = pd.DataFrame(
        pca.components, index=features.columns, columns=[f"PC{i+1}" for i in range(4)]
    )
    print("\nLoadings (sign convention: largest |loading| positive):")
    print(loadings)
    print(
        "\nInterpretation: PC1 loads with one sign on all vol measures, dispersion,"
        "\navg correlation, drawdown and credit proxy, and opposite on trend/term :"
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
    hdr("5. Full-sample 3-state HMM fit : recovery vs truth (permutation-matched)")
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
        "\nThe smoothed probability rises before the flip because it uses later"
        "\nreturns. The filtered probability reacts only after evidence arrives."
        "\nTrading on smoothed probabilities is look-ahead bias."
    )

    # ------------------------------------- 4b. K by out-of-sample likelihood
    hdr("4b. K for the traded model: out-of-sample HMM log-likelihood on PCA-3 features")
    x_feat = features.to_numpy()
    cut = int(0.7 * len(x_feat))
    pca_k = fit_pca(x_feat[:cut], 3)
    z = project(x_feat, pca_k)
    oos = {}
    for k in (2, 3, 4):
        hk = fit_hmm(z[:cut], k, seed=0, n_init=2, max_iter=100)
        oos[k] = hk.score(z[cut:]) / (len(z) - cut)
    print(pd.Series(oos, name="OOS loglik per day").rename_axis("k").to_frame())
    print(f"best K out of sample: {max(oos, key=oos.get)} (trained on the first 70%, scored on the last 30%)")
    print("The GMM-BIC of section 4 is on 1-D index returns and ignores dynamics; this check")
    print("scores the model class that trades (HMM on PCA features) on held-out days.")

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
    print(f"k=2 state vols (ann): {np.round(np.sort(v2), 3)} : the transition state is"
          f"\nabsorbed into its neighbours; true vols were {panel.sigma}.")


# ------------------------------------------------- 9-13. walk-forward study
# Walk-forward settings, fixed before any run in this file and shared by
# every arm and seed. HMM effort is reduced (one start, 50 EM iterations per
# refit) so 2 arms x N_SEEDS walk-forwards fit in about 10 minutes on 10
# worker processes.
WF = dict(n_states=3, min_train=378, refit_every=63, cost_bps=5.0, seed=0,
          enter=0.70, exit_=0.30, target_vol=0.10, n_pca=3,
          detect_kwargs=dict(n_init=1, max_iter=50))
N_SEEDS = 20
SEEDS = tuple(range(101, 101 + N_SEEDS))  # fresh seeds, never used to choose settings
ARMS = ("planted", "equal_drift")
# Strategy variants evaluated on every seed: these are the logged trials.
VARIANTS = {
    "default (next fill f=0.5)": dict(),
    "min_duration=5": dict(min_duration=5),
    "close auction (10bp)": dict(execution=ExecutionConfig("close_auction")),
    "next close (f=1)": dict(execution=ExecutionConfig("next", fill_fraction=1.0)),
}
BASELINES = ("benchmark", "ma_rule", "random_timing", "vol_target")


def _panel(arm: str, seed: int):
    params = None if arm == "planted" else equal_drift_params(3, mu=0.07)
    return make_regime_panel(n_states=3, n_assets=8, n_days=2520, seed=seed, params=params)


def run_one(job):
    """One walk-forward on one (arm, seed); every variant reuses its detection."""
    arm, seed = job
    panel = _panel(arm, seed)
    res = walk_forward_backtest(panel.prices, **WF)
    out = {"arm": arm, "seed": seed}
    for b in BASELINES:
        out[b] = getattr(res, b)["net_ret"]
    bt = {k: WF[k] for k in ("cost_bps", "enter", "exit_", "target_vol")}
    for name, kw in VARIANTS.items():
        r = backtest_from_detection(panel.prices, res.detection, **bt, **kw)
        out[name] = r.ledger["net_ret"]
    if seed == SEEDS[0]:
        out["result"] = res
        out["true_states"] = pd.Series(panel.states, index=panel.returns.index)
    return out


def _sr(x) -> float:
    return sharpe_ratio(np.asarray(x, dtype=float))


def walk_forward_study(runs: list) -> None:
    by = {(r["arm"], r["seed"]): r for r in runs}
    default = next(iter(VARIANTS))

    hdr(f"9. Walk-forward, one synthetic path (arm planted, seed {SEEDS[0]})")
    first = by[("planted", SEEDS[0])]
    res = first["result"]
    stats = pd.DataFrame({
        "strategy": summary_stats(res.ledger),
        "buy_and_hold": summary_stats(res.benchmark),
        "ma_200d": summary_stats(res.ma_rule),
        "random_timing": summary_stats(res.random_timing),
        "vol_target": summary_stats(res.vol_target),
    })
    print(stats)
    net = res.ledger["net_ret"].to_numpy()
    bh = res.benchmark["net_ret"].to_numpy()
    print(f"\nstrategy Sharpe {_sr(net):.2f}, Lo/Newey-West SE {lo_sharpe_se(net):.2f} "
          f"(lag {newey_west_lag(net.size)})")
    pos = np.arange(net.size, dtype=float)
    pt, lo, hi = block_bootstrap_ci(
        pos, lambda i: _sr(net[i.astype(int)]) - _sr(bh[i.astype(int)]), n_boot=1000, seed=1)
    print(f"Sharpe(strategy) - Sharpe(buy & hold): {pt:.2f}, paired block-bootstrap 95% CI "
          f"[{lo:.2f}, {hi:.2f}] (synthetic, one path)")

    log = TrialLog()
    mat = []
    for name in VARIANTS:
        r = first[name].to_numpy()
        log.log(name, r.mean() / r.std(ddof=1))
        mat.append(r)
    dsr = deflated_sharpe_ratio(net, log.n_trials)
    pbo = pbo_cscv(np.column_stack(mat), n_splits=10)
    print(f"Trial log: {log.n_trials} strategy variants evaluated on every seed "
          f"({', '.join(t['name'] for t in log.trials)}).")
    print(f"Deflated Sharpe of the default variant at N={log.n_trials}: {dsr:.3f}; "
          f"PBO (CSCV, 10 blocks) over the variants: {pbo:.2f}")

    runs_tbl = regime_runs(res.detection["regime"])
    print(f"\nRegime runs: {len(runs_tbl)}, median length {runs_tbl['days'].median():.0f}d")

    hdr("10. Regime-conditional risk report (label known at the decision date)")
    regimes = regimes_known_at_decision(res.detection["regime"], res.ledger.index)
    s_net = res.ledger["net_ret"].loc[regimes.index]
    b_net = res.benchmark["net_ret"].loc[regimes.index]
    print("Strategy net returns by regime detected at t-1:")
    print(per_regime_stats(s_net, regimes))
    print("\nBuy-and-hold net returns by regime detected at t-1:")
    print(per_regime_stats(b_net, regimes))
    print("\nTransition P&L attribution (strategy):")
    print(transition_attribution(s_net, regimes))
    after = flip_aftermath(s_net, regimes, k=10)
    print(f"\nFlip aftermath: strategy P&L in the 10 days after each of {len(after)} flips:")
    print(after.groupby("to_regime")["pnl_next_10d"].agg(["count", "mean", "min"]))

    hdr(f"11. Multi-seed Sharpe, {N_SEEDS} fresh seeds per arm, mean [95% CI] (synthetic)")
    cols = list(VARIANTS) + list(BASELINES)
    rows = {}
    sr = {}
    for arm in ARMS:
        for c in cols:
            sr[(arm, c)] = np.array([_sr(by[(arm, s)][c]) for s in SEEDS])
            rows[(arm, c)] = seed_distribution(sr[(arm, c)]).fmt(2)
    tbl = pd.DataFrame({arm: {c: rows[(arm, c)] for c in cols} for arm in ARMS})
    print(tbl.to_string())

    hdr("12. Edge decomposition: planted bear drift vs vol timing (synthetic)")
    d = {}
    for arm in ARMS:
        d[(arm, "vs_bh")] = sr[(arm, default)] - sr[(arm, "benchmark")]
        d[(arm, "vs_random")] = sr[(arm, default)] - sr[(arm, "random_timing")]
        d[(arm, "vs_voltarget")] = sr[(arm, default)] - sr[(arm, "vol_target")]
        d[(arm, "voltarget_vs_bh")] = sr[(arm, "vol_target")] - sr[(arm, "benchmark")]
    lines = {
        "strategy - buy & hold": "vs_bh",
        "strategy - random timing (equal exposure)": "vs_random",
        "strategy - plain vol targeting": "vs_voltarget",
        "plain vol targeting - buy & hold": "voltarget_vs_bh",
    }
    out = pd.DataFrame({arm: {k: seed_distribution(d[(arm, v)]).fmt(2) for k, v in lines.items()}
                        for arm in ARMS})
    print(out.to_string())
    drift_part = seed_distribution(d[("planted", "vs_bh")] - d[("equal_drift", "vs_bh")])
    print(f"\nPart of the Sharpe gain over buy & hold due to the planted bear drift "
          f"(planted minus equal-drift arm, same seeds): {drift_part.fmt(2)}")
    print("Equal-drift arm: every state has drift 7%/yr; only vol and correlation switch.")

    hdr("13. Execution and filter variants (default arm planted, synthetic)")
    for name in VARIANTS:
        if name == default:
            continue
        diff = seed_distribution(sr[("planted", name)] - sr[("planted", default)])
        print(f"{name:28s} Sharpe minus default: {diff.fmt(3)}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args()
    jobs = [(a, s) for a in ARMS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_one, j) for j in jobs]
        diagnostics()  # runs in this process while the workers fit
        runs = [f.result() for f in futures]
    walk_forward_study(runs)
    print(f"\nDone in {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
