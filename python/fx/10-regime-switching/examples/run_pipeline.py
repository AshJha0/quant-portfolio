"""End-to-end FX regime-switching pipeline on a 3-state RORO panel.

Data -> features -> PCA (PC1 = RORO axis) -> BIC k-selection -> HMM
recovery vs truth -> filtered timeline around a planted 2008-style flip
-> walk-forward strategy vs static carry vs oracle -> detection-lag and
per-regime risk report.

Runs offline in under 150 seconds; every number is seeded and
reproducible.  These are the numbers quoted in README.md and
docs/VALIDATION.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fx_regime import (  # noqa: E402
    CURRENCIES,
    apply_hysteresis,
    EM,
    G10_CARRY,
    HAVENS,
    TRANSITION_3,
    DetectionConfig,
    StrategyConfig,
    build_features,
    carry_drawdown_decomposition,
    comparison_table,
    detection_lag_report,
    expected_durations,
    fit_hmm,
    fit_pca,
    generate_roro_panel,
    hmm_bic,
    match_states,
    oracle_gap_decomposition,
    oracle_regimes,
    per_regime_stats,
    perf_stats,
    roro_axis_check,
    run_backtest,
    run_detection,
    select_k_bic,
    static_carry_regimes,
    stationary_distribution,
    transition_attribution,
    viterbi,
)

DETECT_COLS = ["avg_vol", "carry_ret", "haven_rs", "usd_corr", "em_g10"]


def hr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def main() -> None:
    t_start = time.time()
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v: .4f}")

    # ------------------------------------------------------------------
    hr("1. DATA: 3-state RORO panel with a planted 2008-style flip")
    n_days, flip_at, flip_len = 2000, 1500, 45
    panel = generate_roro_panel(
        n_days, n_states=3, seed=11,
        plant_flip_at=flip_at, plant_flip_len=flip_len, plant_flip_state=1,
    )
    freq = np.bincount(panel.states, minlength=3) / n_days
    pi = stationary_distribution(TRANSITION_3)
    print(f"{n_days} business days, {len(CURRENCIES)} currencies vs USD")
    print(f"true state frequencies : {dict(zip(panel.state_names, freq.round(3)))}")
    print(f"stationary distribution: {dict(zip(panel.state_names, pi.round(3)))}")
    print(f"expected durations (d) : "
          f"{dict(zip(panel.state_names, expected_durations(TRANSITION_3).round(1)))}")
    flip_date = panel.returns.index[flip_at]
    print(f"planted risk-off flip  : {flip_date.date()} for {flip_len} days")

    # ------------------------------------------------------------------
    hr("2. FEATURES + PCA: PC1 of the currency panel IS the RORO axis")
    feats = build_features(panel.returns, panel.deposit_rates)
    print(f"feature block: {feats.shape[0]} days x {list(feats.columns)}")

    X = panel.returns.to_numpy()
    Xs = (X - X.mean(0)) / X.std(0)
    pca = fit_pca(Xs, 3)
    load = pd.Series(pca.components[0], index=list(CURRENCIES))
    print("\nPC1 loadings (standardised currency-vs-USD returns):")
    print(load.round(3).to_frame("PC1").T.to_string())
    is_roro = roro_axis_check(
        pca.components, list(CURRENCIES), list(G10_CARRY) + list(EM), list(HAVENS)
    )
    print(f"\ncarry/EM one sign, JPY+CHF opposite sign : {is_roro}")
    print(f"PC1 explained variance ratio             : "
          f"{pca.explained_variance_ratio[0]:.1%}")

    # ------------------------------------------------------------------
    hr("3. MODEL SELECTION: BIC over k (GMM and HMM on detection features)")
    Xf = feats[DETECT_COLS].to_numpy()
    best_k_gmm, gmm_bics = select_k_bic(Xf, k_max=4, seed=0, n_init=2)
    hmm_bics = {}
    for k in range(1, 5):
        m = fit_hmm(Xf, k, seed=0, n_init=2, max_iter=60)
        hmm_bics[k] = hmm_bic(m, Xf)
    tab = pd.DataFrame({"GMM_BIC": gmm_bics, "HMM_BIC": hmm_bics})
    tab.index.name = "k"
    print(tab.round(0).to_string())
    best_k_hmm = min(hmm_bics, key=hmm_bics.get)
    print(f"\nBIC-selected k : GMM -> {best_k_gmm}, HMM -> {best_k_hmm} "
          f"(true k = 3)")
    print("note: on rolling-window features (autocorrelated, fat-tailed)")
    print("HMM-BIC tends to over-select k; the regime count is treated as")
    print("an ECONOMIC choice (2 or 3) with BIC as evidence, not verdict —")
    print("see docs/METHODOLOGY.md.  On raw iid returns BIC recovers the")
    print("true k exactly (tests/test_gmm.py).")

    # ------------------------------------------------------------------
    hr("4. HMM RECOVERY vs TRUTH (fit on the returns panel, "
       "permutation-matched)")
    from fx_regime import TRANSITION_2  # noqa: E402

    print("(a) 2-state RORO panel — the canonical carry-crash cycle:")
    panel2 = generate_roro_panel(2000, n_states=2, seed=11)
    hmm2 = fit_hmm(panel2.returns.to_numpy(), 2, seed=0, n_init=2,
                   max_iter=100)
    perm2 = match_states(panel2.means, hmm2.means,
                         true_covs=panel2.covs, est_covs=hmm2.covs)
    o2 = np.argsort(perm2)
    A2 = hmm2.transmat[np.ix_(o2, o2)]
    print("    est transmat " + str(A2.round(3)).replace("\n", "\n                 "))
    print("    true transmat" + str(TRANSITION_2.round(3)).replace("\n", "\n                 "))
    print(f"    max abs transition error : {np.abs(A2 - TRANSITION_2).max():.3f}")
    print(f"    durations est vs true (d): "
          f"{expected_durations(A2).round(1)} vs "
          f"{expected_durations(TRANSITION_2).round(1)}")
    vit2 = viterbi(hmm2, panel2.returns.to_numpy())
    acc2 = (perm2[vit2] == panel2.states).mean()
    print(f"    Viterbi state accuracy   : {acc2:.1%}")

    print("\n(b) 3-state panel — honesty check on the rare squeeze state:")
    hmm3 = fit_hmm(panel.returns.to_numpy(), 3, seed=0, n_init=2, max_iter=80)
    vit3 = viterbi(hmm3, panel.returns.to_numpy())
    ct = pd.crosstab(pd.Series(vit3, name="est"),
                     pd.Series(panel.states, name="true"))
    print(ct.to_string())
    n_squeeze = int((panel.states == 2).sum())
    print(f"    with only {n_squeeze} true squeeze days "
          f"({n_squeeze / n_days:.1%} of the sample) the crash states\n"
          "    largely MERGE: risk_on is recovered nearly perfectly, but\n"
          "    risk_off and usd_squeeze are hard to tell apart at this\n"
          "    sample size — a documented limitation (docs/VALIDATION.md),\n"
          "    and why both crash books are defensive (carry is cut in\n"
          "    either label).")

    # ------------------------------------------------------------------
    hr("5. FILTERED DETECTION (expanding refit) + planted-flip timeline")
    t0 = time.time()
    det = run_detection(
        feats, DetectionConfig(n_states=3, min_train=252, refit_every=21),
        seed=0,
    )
    print(f"expanding detection: {len(det.refit_dates)} refits, "
          f"{time.time() - t0:.1f}s")
    true_lab = pd.Series(
        [panel.state_names[s] for s in panel.states],
        index=panel.returns.index, name="true",
    )
    window = det.probs.loc[
        flip_date - pd.Timedelta(days=8): flip_date + pd.Timedelta(days=18)
    ]
    timeline = window.round(3).assign(
        regime=det.regimes.reindex(window.index),
        true=true_lab.reindex(window.index),
    )
    print(f"\nfiltered probabilities around the planted flip ({flip_date.date()}):")
    print(timeline.to_string())

    # ------------------------------------------------------------------
    hr("6. WALK-FORWARD STRATEGY: oracle vs filtered vs static carry")
    cfg = StrategyConfig()
    start = det.regimes.index[0]
    bt_filtered = run_backtest(panel.returns, panel.deposit_rates,
                               det.regimes, cfg)
    bt_oracle = run_backtest(
        panel.returns, panel.deposit_rates,
        oracle_regimes(panel.returns.index, panel.states,
                       panel.state_names).loc[start:], cfg,
    )
    bt_static = run_backtest(panel.returns, panel.deposit_rates,
                             static_carry_regimes(det.regimes.index), cfg)
    comp = comparison_table(
        {"oracle": bt_oracle.net, "filtered": bt_filtered.net,
         "static_carry": bt_static.net}
    )
    print("net of pip costs, carry accrual included, 10% vol target:")
    print(comp.round(3).to_string())
    print("\nvalue of the regime filter (filtered - static, ann. return): "
          f"{comp.loc['filtered', 'ann_return'] - comp.loc['static_carry', 'ann_return']: .2%}")
    print("cost of imperfect detection (oracle - filtered, ann. return): "
          f"{comp.loc['oracle', 'ann_return'] - comp.loc['filtered', 'ann_return']: .2%}")

    # ------------------------------------------------------------------
    hr("7. DETECTION LAG vs ORACLE (the honesty metric)")
    print("same filtered probabilities, two hysteresis calibrations —")
    print("the lag / false-alarm trade-off a desk actually governs:\n")
    settings = {
        "fast (0.70/0.30, 2d)": dict(enter_threshold=0.70,
                                     exit_threshold=0.30, min_duration=2),
        "conservative (0.90/0.10, 5d)": dict(enter_threshold=0.90,
                                             exit_threshold=0.10,
                                             min_duration=5),
    }
    rows = {}
    true_aligned = true_lab.reindex(det.regimes.index)
    for name, kw in settings.items():
        regs = apply_hysteresis(det.probs, **kw)
        bt = (bt_filtered if name.startswith("fast")
              else run_backtest(panel.returns, panel.deposit_rates, regs, cfg))
        rep = detection_lag_report(true_aligned, regs, bt_oracle.net, bt.net)
        gap = oracle_gap_decomposition(bt_oracle.net, bt.net, true_lab)
        # pooled lateness cost: every day the market is truly in a risk
        # state while the committed book is still risk_on
        diff = (bt_oracle.net - bt.net).dropna()
        late_mask = (
            true_lab.reindex(diff.index).isin(["risk_off", "usd_squeeze"])
            & (bt.ledger["regime"].reindex(diff.index) == "risk_on")
        )
        stats = perf_stats(bt.net.reindex(bt_oracle.net.index).dropna())
        # stable, drift-based price of one day of lag: on true risk days,
        # the defensive (oracle) book out-drifts the carry book the late
        # filter is still holding by this much per day
        risk_mask = true_lab.reindex(diff.index).isin(
            ["risk_off", "usd_squeeze"]
        )
        per_day_gap_bp = (
            bt_oracle.net[risk_mask].mean()
            - bt_static.net.reindex(diff.index)[risk_mask].mean()
        ) * 1e4
        rows[name] = {
            "n_flips": rep["n_flips"],
            "det_rate": rep["detection_rate"],
            "mean_lag_d": rep["mean_lag_days"],
            "late_days": int(late_mask.sum()),
            "est_lag_cost_per_flip_bp": rep["mean_lag_days"] * per_day_gap_bp,
            "realized_per_flip_bp": rep["mean_cost_per_flip"] * 1e4,
            "false_alarm_cost_bp": gap["gap_calm_days"] * 1e4,
            "sharpe": stats["sharpe"],
        }
    print(pd.DataFrame(rows).T.round(2).to_string())
    print("\nreading: the fast calibration flags a fresh flip ~1 day late;")
    print("one late day costs the drift gap between the defensive and the")
    print("carry book on true risk days (est_lag_cost_per_flip).  The")
    print("realized per-flip cost is the same quantity measured over the")
    print("actual lag windows — with two near-opposite, levered books it is")
    print("dominated by daily noise at this flip count, which is exactly why")
    print("the stable estimate is reported alongside it.  The economically")
    print("binding constraint here is not lag but FALSE ALARMS: the cost of")
    print("sitting defensively through risk-on carry days dwarfs both.  The")
    print("conservative calibration confirms for longer, which slows exits")
    print("too — both costs grow, and fast hysteresis dominates on Sharpe.")

    # ------------------------------------------------------------------
    hr("8. RISK REPORT: per-regime stats, attribution, drawdown decomposition")
    print("filtered strategy, partitioned by TRUE regime:")
    print(per_regime_stats(bt_filtered.net,
                           true_lab.reindex(bt_filtered.net.index))
          .round(3).to_string())
    print("\nstatic carry, partitioned by TRUE regime "
          "(the carry-crash punchline):")
    print(per_regime_stats(bt_static.net,
                           true_lab.reindex(bt_static.net.index))
          .round(3).to_string())
    print("\ntransition attribution of static carry "
          "(first 5 days of each spell vs rest):")
    print(transition_attribution(bt_static.net,
                                 true_lab.reindex(bt_static.net.index),
                                 window=5).round(4).to_string())
    dd = carry_drawdown_decomposition(
        bt_static.net, true_lab.reindex(bt_static.net.index),
        horizons=(1, 3, 5, 10),
    )
    print("\nshare of static-carry risk-state losses in first K days:")
    print(dd.round(3).to_string())

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
