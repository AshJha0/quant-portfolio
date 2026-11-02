"""Risk tests: partitions, attribution identities, oracle ordering, null guard."""

import numpy as np
import pandas as pd
import pytest
from fx_regime import (
    DetectionConfig,
    StrategyConfig,
    build_features,
    carry_drawdown_decomposition,
    comparison_table,
    detection_lag,
    detection_lag_report,
    generate_roro_panel,
    oracle_gap_decomposition,
    oracle_regimes,
    per_regime_stats,
    perf_stats,
    random_timing_regimes,
    regime_spells,
    run_backtest,
    run_detection,
    static_carry_regimes,
    transition_attribution,
)


def _series(vals, start="2020-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def test_perf_stats_hand_check():
    r = _series([0.01] * 252)
    s = perf_stats(r)
    assert s["ann_return"] == pytest.approx(0.01 * 252)
    assert s["ann_vol"] == pytest.approx(0.0)
    assert s["hit_rate"] == 1.0
    assert s["max_drawdown"] == 0.0


def test_max_drawdown_hand_check():
    r = _series([0.10, -0.05, -0.10, 0.03])
    s = perf_stats(r)
    assert s["max_drawdown"] == pytest.approx(-0.15)


def test_per_regime_partition_exact():
    r = _series(np.random.default_rng(0).standard_normal(100) * 0.01)
    reg = _series(["risk_on"] * 60 + ["risk_off"] * 40)
    tab = per_regime_stats(r, reg)
    assert tab["total_pnl"].sum() == pytest.approx(r.sum(), abs=1e-12)
    assert tab.loc["risk_on", "n_days"] == 60
    assert tab.loc["risk_off", "n_days"] == 40


def test_per_regime_missing_labels_raise():
    r = _series([0.01] * 10)
    reg = _series(["risk_on"] * 5)  # does not cover
    with pytest.raises(ValueError):
        per_regime_stats(r, reg)


def test_regime_spells():
    reg = _series(list("AAABBAAC"))
    sp = regime_spells(reg)
    assert list(sp["label"]) == ["A", "B", "A", "C"]
    assert list(sp["length"]) == [3, 2, 2, 1]
    assert list(sp["start"]) == [0, 3, 5, 7]


def test_transition_attribution_identity():
    rng = np.random.default_rng(1)
    r = _series(rng.standard_normal(200) * 0.01)
    reg = _series(
        ["risk_on"] * 80 + ["risk_off"] * 20 + ["risk_on"] * 70
        + ["risk_off"] * 30
    )
    tab = transition_attribution(r, reg, window=5)
    # per-label identity
    assert np.allclose(
        tab["transition_pnl"] + tab["steady_pnl"], tab["total_pnl"], atol=1e-12
    )
    # grand identity
    assert tab["total_pnl"].sum() == pytest.approx(r.sum(), abs=1e-12)
    assert tab.loc["risk_off", "n_spells"] == 2


def test_detection_lag_hand_constructed():
    true = _series(["risk_on"] * 10 + ["risk_off"] * 10 + ["risk_on"] * 10)
    detected = _series(
        ["risk_on"] * 13 + ["risk_off"] * 7 + ["risk_on"] * 10
    )  # flags 3 days late
    flips = detection_lag(true, detected, target_labels=("risk_off",))
    assert len(flips) == 1
    assert flips["lag_days"].iloc[0] == 3
    assert bool(flips["detected"].iloc[0])


def test_detection_lag_undetected_censored():
    true = _series(["risk_on"] * 10 + ["risk_off"] * 5 + ["risk_on"] * 15)
    detected = _series(["risk_on"] * 30)
    flips = detection_lag(true, detected, target_labels=("risk_off",))
    assert len(flips) == 1
    assert not bool(flips["detected"].iloc[0])
    assert flips["lag_days"].iloc[0] >= 5


def test_detection_lag_report_positive_and_costed(panel2, det2, backtests2, true_labels2):
    rep = detection_lag_report(
        true_labels2.reindex(det2.regimes.index),
        det2.regimes,
        backtests2["oracle"].net,
        backtests2["filtered"].net,
        target_labels=("risk_off",),
    )
    assert rep["n_flips"] > 0
    assert rep["mean_lag_days"] > 0  # a filter is never early on average
    assert np.isfinite(rep["mean_cost_per_flip"])
    assert len(rep["flips"]) == rep["n_flips"]
    assert 0.0 <= rep["detection_rate"] <= 1.0


def test_drawdown_decomposition_front_loaded(panel2, backtests2, true_labels2):
    """Carry losses in risk-off are front-loaded: share rises with K and
    the first 10 days carry most of the pain."""
    static_net = backtests2["static"].net
    tab = carry_drawdown_decomposition(
        static_net, true_labels2.reindex(static_net.index),
        horizons=(1, 3, 5, 10), risk_labels=("risk_off",),
    )
    assert (tab.index == [1, 3, 5, 10]).all()
    # carry loses money in the entry window of risk-off at every horizon
    assert (tab["pnl_first_k"] < 0).all()
    # ... and the first 5 days already account for over half the total
    # risk-off loss (front-loading: the crash happens at the flip)
    assert tab.loc[5, "share_of_risk_pnl"] > 0.5
    assert np.isfinite(tab["share_of_risk_pnl"]).all()
    # identity: with K longer than any spell, the share is exactly 1
    full = carry_drawdown_decomposition(
        static_net, true_labels2.reindex(static_net.index),
        horizons=(10_000,), risk_labels=("risk_off",),
    )
    assert full.loc[10_000, "share_of_risk_pnl"] == pytest.approx(1.0, abs=1e-12)


def test_oracle_gap_decomposition_identity(backtests2, true_labels2):
    gap = oracle_gap_decomposition(
        backtests2["oracle"].net, backtests2["filtered"].net, true_labels2,
        risk_labels=("risk_off",),
    )
    total = (
        backtests2["oracle"].net - backtests2["filtered"].net
    ).dropna().sum()
    assert gap["gap_total"] == pytest.approx(total, abs=1e-12)
    assert gap["gap_risk_days"] + gap["gap_calm_days"] == pytest.approx(
        gap["gap_total"], abs=1e-12
    )


def test_oracle_gap_decomposition_invalid_raises():
    a = _series([0.01] * 10)
    b = _series([0.02] * 10, start="2021-01-01")  # disjoint dates
    with pytest.raises(ValueError):
        oracle_gap_decomposition(a, b, _series(["risk_on"] * 10))


def test_comparison_table_alignment(backtests2):
    tab = comparison_table(
        {
            "oracle": backtests2["oracle"].net,
            "filtered": backtests2["filtered"].net,
            "static_carry": backtests2["static"].net,
        }
    )
    assert (tab["n_days"] == tab["n_days"].iloc[0]).all()
    assert set(tab.index) == {"oracle", "filtered", "static_carry"}


def test_oracle_beats_filtered_beats_static_statistically():
    """Oracle >= filtered >= static, averaged over seeds (loose tolerances)."""
    sharpes = {"oracle": [], "filtered": [], "static": []}
    cfg = StrategyConfig()
    for seed in (0, 1, 2, 3):
        panel = generate_roro_panel(900, n_states=2, seed=seed)
        feats = build_features(panel.returns, panel.deposit_rates)
        det = run_detection(
            feats, DetectionConfig(n_states=2, min_train=252, refit_every=63),
            seed=0,
        )
        start = det.regimes.index[0]
        bt_f = run_backtest(panel.returns, panel.deposit_rates, det.regimes, cfg)
        bt_o = run_backtest(
            panel.returns, panel.deposit_rates,
            oracle_regimes(panel.returns.index, panel.states,
                           panel.state_names).loc[start:], cfg,
        )
        bt_s = run_backtest(
            panel.returns, panel.deposit_rates,
            static_carry_regimes(det.regimes.index), cfg,
        )
        tab = comparison_table(
            {"oracle": bt_o.net, "filtered": bt_f.net, "static": bt_s.net}
        )
        for k in sharpes:
            sharpes[k].append(tab.loc[k, "sharpe"])
    mo = np.mean(sharpes["oracle"])
    mf = np.mean(sharpes["filtered"])
    ms = np.mean(sharpes["static"])
    assert mo > ms + 0.10          # oracle clearly beats static carry
    assert mo >= mf - 0.10         # oracle >= filtered (loose)
    assert mf >= ms - 0.10         # filtered >= static (loose)


def _timing_edge(panel, n_shifts=4):
    """Sharpe of the filtered book minus the mean Sharpe of its
    random-timing twins (same labels, same occupancy, shifted in time)."""
    feats = build_features(panel.returns, panel.deposit_rates)
    det = run_detection(
        feats, DetectionConfig(n_states=2, min_train=252, refit_every=63,
                               n_init=1, max_iter=30),
        seed=0,
    )
    cfg = StrategyConfig()
    bt_f = run_backtest(panel.returns, panel.deposit_rates, det.regimes, cfg)
    sh_f = perf_stats(bt_f.net)["sharpe"]
    sh_r = []
    for k in range(n_shifts):
        reg = random_timing_regimes(det.regimes, seed=k)
        bt_r = run_backtest(panel.returns, panel.deposit_rates, reg, cfg)
        sh_r.append(perf_stats(bt_r.net.reindex(bt_f.net.index))["sharpe"])
    return sh_f - float(np.mean(sh_r))


NULL_SEEDS = (101, 102, 103, 104, 105, 106)


def test_vol_regime_null_no_timing_edge():
    """Null with vol clustering but ZERO mean link (every state has zero
    drift; vol and correlation regimes kept). The filter's Sharpe must not
    beat random timing at equal exposure. Judged on the mean over 6 seeds
    against its standard error: the guard fails if the mean edge exceeds
    3 SE (one-sided, about 0.1% false-failure rate under a true null)."""
    edges = np.array([
        _timing_edge(generate_roro_panel(900, n_states=2, seed=s, equal_drift=True))
        for s in NULL_SEEDS
    ])
    se = edges.std(ddof=1) / np.sqrt(len(edges))
    assert edges.mean() < 3 * se, (edges.mean(), se)


def _oracle_timing_edge(panel, n_shifts=4):
    """Same statistic as _timing_edge, with the true states as the signal."""
    cfg = StrategyConfig()
    reg = oracle_regimes(panel.returns.index, panel.states,
                         panel.state_names).iloc[252:]
    bt = run_backtest(panel.returns, panel.deposit_rates, reg, cfg)
    sh = perf_stats(bt.net)["sharpe"]
    sh_r = [
        perf_stats(run_backtest(panel.returns, panel.deposit_rates,
                                random_timing_regimes(reg, seed=k), cfg)
                   .net.reindex(bt.net.index))["sharpe"]
        for k in range(n_shifts)
    ]
    return sh - float(np.mean(sh_r))


def test_timing_edge_statistic_has_power():
    """The guard above must be able to fail. With the true states as the
    signal, the same statistic on the same seeds detects the planted crash
    drift (mean > 3 SE) and finds nothing on the zero-drift panels
    (|mean| < 3 SE). Detector-specific edges are reported, with CIs, by
    examples/run_pipeline.py on longer panels."""
    seeds = range(201, 213)  # 12 seeds of 2000 days: per-seed sd about 0.6
    planted = np.array([_oracle_timing_edge(generate_roro_panel(2000, n_states=2, seed=s))
                        for s in seeds])
    se = planted.std(ddof=1) / np.sqrt(len(planted))
    assert planted.mean() > 3 * se, (planted.mean(), se)
    null = np.array([
        _oracle_timing_edge(generate_roro_panel(2000, n_states=2, seed=s, equal_drift=True))
        for s in seeds
    ])
    se0 = null.std(ddof=1) / np.sqrt(len(null))
    assert abs(null.mean()) < 3 * se0, (null.mean(), se0)


def test_equal_drift_panel_keeps_states_and_shocks():
    a = generate_roro_panel(300, n_states=3, seed=9)
    b = generate_roro_panel(300, n_states=3, seed=9, equal_drift=True)
    np.testing.assert_array_equal(a.states, b.states)
    np.testing.assert_allclose(b.means, 0.0)
    np.testing.assert_allclose(a.covs, b.covs)
    # returns differ by exactly the planted drift of each day's state
    drift = a.means[a.states]
    np.testing.assert_allclose(a.returns.to_numpy() - b.returns.to_numpy(), drift, atol=1e-15)


def test_random_timing_keeps_occupancy():
    reg = pd.Series(["risk_on"] * 70 + ["risk_off"] * 30)
    sh = random_timing_regimes(reg, seed=3)
    assert (sh.value_counts() == reg.value_counts()).all()
    assert not (sh == reg).all()
    with pytest.raises(ValueError):
        random_timing_regimes(reg.iloc[:5])


def test_empty_and_invalid_inputs_raise():
    with pytest.raises(ValueError):
        perf_stats(pd.Series(dtype=float))
    with pytest.raises(ValueError):
        comparison_table({})
    with pytest.raises(ValueError):
        regime_spells(pd.Series(dtype=object))
