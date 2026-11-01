"""End-to-end FX regime-switching pipeline on synthetic RORO panels.

Every number printed here comes from synthetic data whose regimes, crash
drifts and carry were planted by ``fx_regime.data.synthetic``. No real-data
result exists yet.

Sections
--------
1. One illustrative synthetic panel: PCA, BIC, HMM recovery.
2. DESIGN on seeds 1001-1003 only: 12 configurations (feature windows x
   detection columns x risk-off basket) are each run once and logged in a
   TrialLog. The best mean Sharpe picks the configuration. PBO (CSCV) is
   computed over the 12.
3. EVALUATION on fresh seeds 2001-2008, never used in design: oracle,
   filtered, static carry and random timing at equal exposure, each seed
   run twice: with the planted crash drift and with equal (zero) drift in
   every state (vol-clustering null). Mean [95% CI] across seeds, deflated
   Sharpe at the logged trial count, block-bootstrap CI.
4. Execution variants (fill fraction 0.5, 1.0, close auction), detection
   lag, and the carry front-loading table, all across evaluation seeds.

Runtime: about 10-20 minutes on a 12-core machine (detections run in
parallel processes). Pass ``--workers N`` to change the pool size. The
script prints only; it writes no files.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fx_regime import (  # noqa: E402
    CURRENCIES,
    EM,
    G10_CARRY,
    HAVENS,
    TRANSITION_2,
    TRANSITION_3,
    DetectionConfig,
    ExecutionConfig,
    StrategyConfig,
    apply_hysteresis,
    build_features,
    carry_drawdown_decomposition,
    detection_lag_report,
    expected_durations,
    fit_hmm,
    fit_pca,
    generate_roro_panel,
    hmm_bic,
    match_states,
    oracle_regimes,
    perf_stats,
    random_timing_regimes,
    roro_axis_check,
    run_backtest,
    run_detection,
    select_k_bic,
    static_carry_regimes,
    viterbi,
)
from fx_regime.features import FeatureConfig  # noqa: E402
from fx_regime.research_stats import (  # noqa: E402
    TrialLog,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    lo_sharpe_se,
    pbo_cscv,
    seed_distribution,
    sharpe_ratio,
)

N_DAYS = 1500
N_STATES = 3
DESIGN_SEEDS = (1001, 1002, 1003)
EVAL_SEEDS = tuple(range(2001, 2009))
N_RANDOM_SHIFTS = 4

WINDOWS = {
    "long_21_10_21": FeatureConfig(vol_window=21, ret_window=10, corr_window=21),
    "short_8_5_12": FeatureConfig(vol_window=8, ret_window=5, corr_window=12),
}
COLUMNS = {
    "seven_with_fwd_ts": ("avg_vol", "carry_ret", "haven_rs", "usd_corr",
                          "em_g10", "usd_str", "fwd_ts"),
    "six_with_usd_str": ("avg_vol", "carry_ret", "haven_rs", "usd_corr", "em_g10",
                         "usd_str"),
    "five_no_usd_str": ("avg_vol", "carry_ret", "haven_rs", "usd_corr", "em_g10"),
}
BASKETS = {
    "aud_nzd": ("AUD", "NZD"),
    "aud_nzd_em": ("AUD", "NZD", "MXN", "ZAR", "BRL"),
}


def det_config(cols) -> DetectionConfig:
    # HMM effort is set for runtime: refit every 63 days, one cold start,
    # a cold restart competing every 4th refit, 40 EM iterations.
    return DetectionConfig(n_states=N_STATES, min_train=252, refit_every=63,
                           n_init=1, max_iter=40, detect_columns=tuple(cols))


def hr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# --------------------------------------------------------------------------
# workers (top level so they can run in a process pool)
# --------------------------------------------------------------------------


def design_worker(args):
    seed, wkey, ckey = args
    panel = generate_roro_panel(N_DAYS, n_states=N_STATES, seed=seed)
    feats = build_features(panel.returns, panel.deposit_rates, WINDOWS[wkey])
    det = run_detection(feats, det_config(COLUMNS[ckey]), seed=0)
    out = {}
    for bkey, basket in BASKETS.items():
        cfg = StrategyConfig(risk_proxy_currencies=basket)
        bt = run_backtest(panel.returns, panel.deposit_rates, det.regimes, cfg)
        out[bkey] = bt.net
    return seed, wkey, ckey, out


def eval_worker(args):
    seed, equal_drift, wkey, ckey, bkey = args
    panel = generate_roro_panel(N_DAYS, n_states=N_STATES, seed=seed,
                                equal_drift=equal_drift)
    feats = build_features(panel.returns, panel.deposit_rates, WINDOWS[wkey])
    det = run_detection(feats, det_config(COLUMNS[ckey]), seed=0)
    cfg = StrategyConfig(risk_proxy_currencies=BASKETS[bkey])
    R, D = panel.returns, panel.deposit_rates
    start = det.regimes.index[0]
    true_lab = pd.Series([panel.state_names[s] for s in panel.states],
                         index=R.index)
    filt = run_backtest(R, D, det.regimes, cfg)
    idx = filt.net.index
    oracle = run_backtest(R, D, oracle_regimes(R.index, panel.states,
                                               panel.state_names).loc[start:], cfg)
    static = run_backtest(R, D, static_carry_regimes(det.regimes.index), cfg)
    rnd = [
        perf_stats(run_backtest(R, D, random_timing_regimes(det.regimes, k), cfg)
                   .net.reindex(idx))["sharpe"]
        for k in range(N_RANDOM_SHIFTS)
    ]
    res = {
        "seed": seed,
        "equal_drift": equal_drift,
        "net_filtered": filt.net,
        "sharpe": {
            "oracle": perf_stats(oracle.net.reindex(idx))["sharpe"],
            "filtered": perf_stats(filt.net)["sharpe"],
            "static_carry": perf_stats(static.net.reindex(idx))["sharpe"],
            "random_timing": float(np.mean(rnd)),
        },
        "ann_return": {
            "oracle": perf_stats(oracle.net.reindex(idx))["ann_return"],
            "filtered": perf_stats(filt.net)["ann_return"],
            "static_carry": perf_stats(static.net.reindex(idx))["ann_return"],
        },
        "max_dd": {
            "oracle": perf_stats(oracle.net.reindex(idx))["max_drawdown"],
            "filtered": perf_stats(filt.net)["max_drawdown"],
            "static_carry": perf_stats(static.net.reindex(idx))["max_drawdown"],
        },
    }
    if not equal_drift:
        ex = {}
        for name, e in {
            "next f=0.5": ExecutionConfig(),
            "next f=1.0": ExecutionConfig(fill_fraction=1.0),
            "close auction 10bp": ExecutionConfig(mode="close_auction"),
        }.items():
            ex[name] = perf_stats(run_backtest(R, D, det.regimes, cfg, e).net)["sharpe"]
        res["exec"] = ex
        true_al = true_lab.reindex(det.regimes.index)
        lag = {}
        for name, kw in {
            "fast 0.70/0.30/2d": dict(enter_threshold=0.70, exit_threshold=0.30,
                                      min_duration=2),
            "conservative 0.90/0.10/5d": dict(enter_threshold=0.90,
                                              exit_threshold=0.10, min_duration=5),
        }.items():
            regs = apply_hysteresis(det.probs, **kw)
            bt = filt if name.startswith("fast") else run_backtest(R, D, regs, cfg)
            rep = detection_lag_report(true_al, regs, oracle.net, bt.net)
            lag[name] = {
                "n_flips": rep["n_flips"],
                "detection_rate": rep["detection_rate"],
                "mean_lag_days": rep["mean_lag_days"],
                "sharpe": perf_stats(bt.net)["sharpe"],
            }
        res["lag"] = lag
        dd = carry_drawdown_decomposition(
            static.net, true_lab.reindex(static.net.index), horizons=(1, 3, 5, 10))
        res["front_load"] = dd["share_of_risk_pnl"].to_dict()
        risk = ["risk_off", "usd_squeeze"]
        res["accuracy"] = float(
            (det.regimes.reindex(idx).isin(risk)
             == true_lab.reindex(idx).isin(risk)).mean()
        )
    return res


# --------------------------------------------------------------------------


def section_illustration() -> None:
    hr("1. ONE SYNTHETIC PANEL: PCA, BIC and HMM recovery (seed 11)")
    panel = generate_roro_panel(N_DAYS, n_states=3, seed=11)
    freq = np.bincount(panel.states, minlength=3) / N_DAYS
    print(f"{N_DAYS} business days, {len(CURRENCIES)} currencies vs USD (synthetic)")
    print(f"true state frequencies : {dict(zip(panel.state_names, freq.round(3)))}")
    print(f"expected durations (d) : "
          f"{dict(zip(panel.state_names, expected_durations(TRANSITION_3).round(1)))}")
    X = panel.returns.to_numpy()
    Xs = (X - X.mean(0)) / X.std(0)
    pca = fit_pca(Xs, 3)
    is_roro = roro_axis_check(
        pca.components, list(CURRENCIES), list(G10_CARRY) + list(EM), list(HAVENS))
    print(f"PC1 carry/EM one sign, JPY+CHF opposite : {is_roro}; "
          f"explained variance {pca.explained_variance_ratio[0]:.1%}")

    feats = build_features(panel.returns, panel.deposit_rates)
    Xf = feats[list(DetectionConfig().detect_columns)].to_numpy()
    best_k_gmm, gmm_bics = select_k_bic(Xf, k_max=4, seed=0, n_init=1)
    hmm_bics = {k: hmm_bic(fit_hmm(Xf, k, seed=0, n_init=1, max_iter=40), Xf)
                for k in range(1, 5)}
    tab = pd.DataFrame({"GMM_BIC": gmm_bics, "HMM_BIC": hmm_bics})
    tab.index.name = "k"
    print(tab.round(0).to_string())
    print(f"BIC-selected k: GMM {best_k_gmm}, HMM {min(hmm_bics, key=hmm_bics.get)}"
          f" (true k = 3). The strategy uses k = 3 by assumption.")

    panel2 = generate_roro_panel(N_DAYS, n_states=2, seed=11)
    hmm2 = fit_hmm(panel2.returns.to_numpy(), 2, seed=0, n_init=2, max_iter=100)
    perm2 = match_states(panel2.means, hmm2.means,
                         true_covs=panel2.covs, est_covs=hmm2.covs)
    o2 = np.argsort(perm2)
    A2 = hmm2.transmat[np.ix_(o2, o2)]
    acc2 = (perm2[viterbi(hmm2, panel2.returns.to_numpy())] == panel2.states).mean()
    print(f"2-state HMM on returns: max abs transition error "
          f"{np.abs(A2 - TRANSITION_2).max():.3f}, Viterbi accuracy {acc2:.1%}")


def section_design(pool) -> tuple[tuple[str, str, str], TrialLog]:
    hr("2. DESIGN on seeds 1001-1003 only: 12 configurations, all logged")
    jobs = [(s, w, c) for s in DESIGN_SEEDS for w in WINDOWS for c in COLUMNS]
    results = list(pool.map(design_worker, jobs))
    nets: dict[tuple[str, str, str], list[pd.Series]] = {}
    for seed, w, c, out in results:
        for b, net in out.items():
            nets.setdefault((w, c, b), []).append(net)
    log = TrialLog()
    rows = {}
    for key, series in nets.items():
        sh = [perf_stats(s)["sharpe"] for s in series]
        pooled = pd.concat(series, ignore_index=True).to_numpy()
        log.log("/".join(key), float(pooled.mean() / pooled.std(ddof=1)),
                windows=key[0], columns=key[1], basket=key[2])
        rows["/".join(key)] = {"mean_sharpe": np.mean(sh), "min": np.min(sh),
                               "max": np.max(sh)}
    tab = pd.DataFrame(rows).T.sort_values("mean_sharpe", ascending=False)
    print("filtered strategy Sharpe on the 3 design seeds (synthetic):")
    print(tab.round(3).to_string())
    best = tuple(tab.index[0].split("/"))
    # PBO: each configuration's daily returns, concatenated over design
    # seeds on a common per-seed length, as the T x N CSCV matrix
    keys = list(nets)
    per_seed_len = [min(len(nets[k][i]) for k in keys) for i in range(len(DESIGN_SEEDS))]
    M = np.column_stack([
        np.concatenate([nets[k][i].to_numpy()[-per_seed_len[i]:]
                        for i in range(len(DESIGN_SEEDS))])
        for k in keys
    ])
    pbo = pbo_cscv(M, n_splits=10)
    print(f"\nlogged trials: {log.n_trials} (the 2 window sets, 3 column sets and "
          "2 baskets tried during development)")
    print(f"PBO (CSCV, 10 blocks) over the 12 configurations: {pbo:.2f}")
    print(f"chosen on design seeds: windows={best[0]}, columns={best[1]}, basket={best[2]}")
    lib = (DetectionConfig().detect_columns, StrategyConfig().risk_proxy_currencies)
    print("library defaults equal the chosen columns and basket: "
          f"{lib == (COLUMNS[best[1]], BASKETS[best[2]])}")
    return best, log


def section_evaluation(pool, best, log: TrialLog) -> None:
    hr("3. EVALUATION on fresh seeds 2001-2008 (never used in design)")
    w, c, b = best
    jobs = [(s, ed, w, c, b) for s in EVAL_SEEDS for ed in (False, True)]
    res = list(pool.map(eval_worker, jobs))
    planted = [r for r in res if not r["equal_drift"]]
    null = [r for r in res if r["equal_drift"]]

    def dist(rs, fn):
        return seed_distribution([fn(r) for r in rs])

    print("Sharpe ratio across 8 seeds, mean [95% CI] (synthetic, planted crash drift):")
    for k in ("oracle", "filtered", "static_carry", "random_timing"):
        print(f"  {k:<14s} {dist(planted, lambda r: r['sharpe'][k]).fmt()}")
    print("annual return and max drawdown (additive, return units), mean [95% CI]:")
    for k in ("oracle", "filtered", "static_carry"):
        print(f"  {k:<14s} ret {dist(planted, lambda r: r['ann_return'][k]).fmt(3)}"
              f"   maxDD {dist(planted, lambda r: r['max_dd'][k]).fmt(3)}")

    print("\nEdge decomposition, Sharpe differences, mean [95% CI] over 8 seeds (synthetic):")
    rows = [
        ("filtered - static, planted drift",
         [r["sharpe"]["filtered"] - r["sharpe"]["static_carry"] for r in planted]),
        ("filtered - static, zero drift (vol-clustering null)",
         [r["sharpe"]["filtered"] - r["sharpe"]["static_carry"] for r in null]),
        ("  part due to planted drift (paired difference)",
         [p["sharpe"]["filtered"] - p["sharpe"]["static_carry"]
          - (q["sharpe"]["filtered"] - q["sharpe"]["static_carry"])
          for p, q in zip(planted, null)]),
        ("filtered - random timing, planted drift",
         [r["sharpe"]["filtered"] - r["sharpe"]["random_timing"] for r in planted]),
        ("filtered - random timing, zero drift",
         [r["sharpe"]["filtered"] - r["sharpe"]["random_timing"] for r in null]),
    ]
    for name, vals in rows:
        print(f"  {name:<52s} {seed_distribution(vals).fmt()}")

    pooled = pd.concat([r["net_filtered"] for r in planted], ignore_index=True).to_numpy()
    one = planted[0]["net_filtered"].to_numpy()
    print(f"\nseed {EVAL_SEEDS[0]} filtered: Sharpe {sharpe_ratio(one):.2f}, "
          f"Lo/Newey-West SE {lo_sharpe_se(one):.2f}")
    pt, lo, hi = block_bootstrap_ci(pooled, sharpe_ratio, block_length=20, n_boot=1000)
    print(f"pooled 8-seed filtered Sharpe {pt:.2f}, block-bootstrap 95% CI "
          f"[{lo:.2f}, {hi:.2f}] (20-day blocks)")
    dsr = deflated_sharpe_ratio(pooled, log.n_trials, log.sharpes())
    print(f"deflated Sharpe ratio at {log.n_trials} logged trials: {dsr:.3f}")

    hr("4. EXECUTION, DETECTION LAG, CARRY FRONT-LOADING (8 evaluation seeds, synthetic)")
    print("filtered Sharpe by fill model, mean [95% CI]:")
    for k in planted[0]["exec"]:
        print(f"  {k:<20s} {dist(planted, lambda r: r['exec'][k]).fmt()}")
    print("\nhysteresis calibrations, mean [95% CI]:")
    for k in planted[0]["lag"]:
        print(f"  {k}")
        for m in ("n_flips", "detection_rate", "mean_lag_days", "sharpe"):
            vals = [r["lag"][k][m] for r in planted
                    if np.isfinite(r["lag"][k][m])]
            print(f"    {m:<16s} {seed_distribution(vals).fmt()}")
    print(f"\nrisk-state classification accuracy (risk_off and usd_squeeze merged): "
          f"{dist(planted, lambda r: r['accuracy']).fmt(3)}")
    print("\nshare of static-carry risk-state P&L in the first K days of a spell:")
    for K in (1, 3, 5, 10):
        vals = [r["front_load"][K] for r in planted if np.isfinite(r["front_load"][K])]
        print(f"  K={K:<3d} {seed_distribution(vals).fmt()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    pd.set_option("display.width", 120)
    section_illustration()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        best, log = section_design(pool)
        section_evaluation(pool, best, log)
    print(f"\nTotal runtime: {time.time() - t0:.0f}s with {args.workers} workers")


if __name__ == "__main__":
    main()
