"""Impact calibration, full-backtest capacity, causal ADV and the execution race."""

import inspect

import numpy as np
import pandas as pd
import pytest
from eq_algo import (
    BacktestConfig,
    IntradayConfig,
    IntradayMarket,
    capacity_backtest,
    capacity_curve,
    cs_zscore,
    evaluate_schedules,
    generate_daily_panel,
    momentum,
    pov_schedule,
    twap_schedule,
    vwap_schedule,
)


def test_default_impact_coefficient_is_in_the_cited_range():
    # METHODOLOGY cites Y of order 0.5-1 (Almgren et al. 2005; Toth et al. 2011)
    assert 0.5 <= BacktestConfig().impact_coef <= 1.0
    default = inspect.signature(capacity_curve).parameters["impact_coef"].default
    assert 0.5 <= default <= 1.0


def test_capacity_backtest_monotone_in_aum_and_impact():
    panel = generate_daily_panel(n_stocks=40, n_days=330, seed=5)
    sig = cs_zscore(momentum(panel.prices, 252, 21))
    cfg = BacktestConfig(n_quantiles=5, linear_cost_bps=5.0)
    cap = capacity_backtest(panel.prices, sig, panel.volumes, cfg,
                            aum_grid=[1e7, 1e9], impact_coefs=(0.3, 1.0), start=10)
    s = cap["net_sharpe"]
    d = cap["drag_bps_per_day"]
    assert d[(0.3, 1e9)] > d[(0.3, 1e7)]
    assert d[(1.0, 1e9)] > d[(0.3, 1e9)]
    assert s[(1.0, 1e9)] < s[(0.3, 1e9)]
    # participation is linear in AUM at fixed weights
    p = cap["p95_participation"]
    assert p[(0.3, 1e9)] == pytest.approx(100.0 * p[(0.3, 1e7)], rel=1e-9)


def test_full_sample_adv_is_deprecated_and_trailing_adv_is_causal():
    panel = generate_daily_panel(n_stocks=5, n_days=60, seed=1)
    with pytest.warns(DeprecationWarning):
        _ = panel.adv_dollars
    adv = panel.trailing_adv_dollars(10)
    # mutate the future: values up to the cutoff must not change
    panel.volumes.iloc[40:] *= 3.0
    adv2 = panel.trailing_adv_dollars(10)
    pd.testing.assert_frame_equal(adv.iloc[:40], adv2.iloc[:40])
    assert adv.iloc[:9].isna().all().all()


def test_race_publishes_deterministic_expected_cost():
    cfg = IntradayConfig(mid0=100.0, day_volume=1e6, n_buckets=26, sigma_daily=0.02,
                         spread_bps=5.0, temp_coef=1.0, perm_coef=0.5, vol_noise=0.2)
    mkt = IntradayMarket(cfg)
    X = 50_000.0
    sch = {"TWAP": twap_schedule(X, 26), "VWAP": vwap_schedule(X, cfg.profile),
           "POV": pov_schedule(X, cfg.profile * cfg.day_volume, 0.10)}
    tab = evaluate_schedules(mkt, sch, n_reps=50, seed=3)
    # expected IS does not depend on the replication seed or count
    tab2 = evaluate_schedules(mkt, sch, n_reps=20, seed=99)
    np.testing.assert_allclose(tab["expected_is_bps"], tab2["expected_is_bps"], rtol=0, atol=0)
    # hand check for TWAP on the zero-noise market: half spread + bucket
    # square-root impact + permanent drift from earlier child orders
    q = X / 26
    prof = cfg.profile
    mid = 100.0
    cost = 0.0
    for j in range(26):
        temp = 1.0 * 0.02 * np.sqrt(q / (26 * prof[j] * 1e6))
        cost += q * mid * (1 + 2.5e-4 + temp)
        mid += 0.5 * 0.02 * (q / 1e6) * mid
    expected = (cost / X - 100.0) / 100.0 * 1e4
    assert tab.loc["TWAP", "expected_is_bps"] == pytest.approx(expected, rel=1e-10)
    # the Monte Carlo mean differs from the expectation by noise, measured by its SE
    for k in sch:
        z = (tab.loc[k, "mean_is_bps"] - tab.loc[k, "expected_is_bps"]) / tab.loc[k, "se_mean_is_bps"]
        assert abs(z) < 5.0
