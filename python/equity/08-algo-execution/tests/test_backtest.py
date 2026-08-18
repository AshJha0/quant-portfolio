"""Backtester tests: exact hand-computed ledger, no-lookahead mutation on the
full pipeline, exposure and cap enforcement, dollar neutrality, cost model."""

import numpy as np
import pandas as pd
import pytest

from eq_algo import (BacktestConfig, cs_zscore, generate_daily_panel,
                     long_short_weights, momentum, run_backtest)


def _toy():
    dates = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0],
         [11.0, 19.0, 33.0, 38.0],
         [11.0, 19.0, 33.0, 38.0]],
        index=dates, columns=list("ABCD"))
    signal = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]] * 3, index=dates, columns=list("ABCD"))
    return prices, signal


def test_ledger_hand_computed_exact():
    prices, signal = _toy()
    cfg = BacktestConfig(n_quantiles=2, gross_exposure=2.0, max_weight=1.0,
                         linear_cost_bps=10.0, impact_coef=0.0)
    res = run_backtest(prices, signal, cfg)
    led = res.ledger
    # weights every day: C,D long 0.5 each; A,B short 0.5 each
    w = res.weights.iloc[0]
    np.testing.assert_allclose(w[list("ABCD")], [-0.5, -0.5, 0.5, 0.5], atol=1e-12)
    # day0: no prior position -> gross 0; trade turnover 2.0, cost 10bp*2 = 2e-3
    assert led["gross_ret"].iloc[0] == 0.0
    assert led["turnover"].iloc[0] == pytest.approx(2.0, abs=1e-12)
    assert led["cost"].iloc[0] == pytest.approx(2e-3, abs=1e-15)
    assert led["net_ret"].iloc[0] == pytest.approx(-2e-3, abs=1e-15)
    # day1 gross: -0.5*(0.1) -0.5*(-0.05) + 0.5*(0.1) + 0.5*(-0.05) = 0.0
    r_a, r_b, r_c, r_d = 0.1, -0.05, 0.1, -0.05
    expected = -0.5 * r_a - 0.5 * r_b + 0.5 * r_c + 0.5 * r_d
    assert led["gross_ret"].iloc[1] == pytest.approx(expected, abs=1e-15)
    # day1: same weights -> zero turnover, zero cost, net == gross
    assert led["turnover"].iloc[1] == pytest.approx(0.0, abs=1e-12)
    assert led["net_ret"].iloc[1] == pytest.approx(expected, abs=1e-15)
    # day2 gross: flat prices -> 0
    assert led["gross_ret"].iloc[2] == pytest.approx(0.0, abs=1e-15)


def test_no_lookahead_full_pipeline_mutation():
    """Mutate all data strictly after the cutoff; ledger rows <= cutoff must
    be bit-identical (features -> signal -> weights -> costs -> returns)."""
    panel = generate_daily_panel(n_stocks=40, n_days=420, seed=21)
    cfg = BacktestConfig(n_quantiles=5, linear_cost_bps=5.0, impact_coef=0.1,
                         aum=50e6, rebalance_band=0.002)

    def full_pipeline(prices, volumes):
        sig = cs_zscore(momentum(prices, 252, 21))
        return run_backtest(prices, sig, cfg, volumes=volumes).ledger

    base = full_pipeline(panel.prices, panel.volumes)
    cut = 400
    p2, v2 = panel.prices.copy(), panel.volumes.copy()
    rng = np.random.default_rng(1)
    p2.iloc[cut + 1:] = p2.iloc[cut + 1:].to_numpy() * rng.uniform(0.5, 2.0, p2.iloc[cut + 1:].shape)
    v2.iloc[cut + 1:] = v2.iloc[cut + 1:].to_numpy() * rng.uniform(0.5, 2.0, v2.iloc[cut + 1:].shape)
    mutated = full_pipeline(p2, v2)
    pd.testing.assert_frame_equal(base.iloc[:cut + 1], mutated.iloc[:cut + 1])


def test_exposure_limits_and_dollar_neutrality():
    panel = generate_daily_panel(n_stocks=50, n_days=350, seed=2)
    sig = cs_zscore(momentum(panel.prices, 252, 21))
    cfg = BacktestConfig(n_quantiles=5, gross_exposure=1.6, max_weight=0.03)
    res = run_backtest(panel.prices, sig, cfg)
    led, w = res.ledger, res.weights
    active = led["gross_exposure"] > 0
    assert active.any()
    assert (led["gross_exposure"] <= 1.6 + 1e-9).all()
    assert led.loc[active, "net_exposure"].abs().max() < 1e-12  # L/S dollar neutral
    assert w.abs().to_numpy().max() <= 0.03 + 1e-12             # position limit


def test_max_weight_cap_binds_and_neutrality_preserved():
    # 10 names, quintiles -> 2 names/side, natural weight 0.5 > cap 0.1
    sig = pd.DataFrame([np.arange(10.0)], columns=[f"S{i}" for i in range(10)])
    w = long_short_weights(sig, n_quantiles=5, gross_exposure=2.0, max_weight=0.1)
    row = w.iloc[0]
    assert row.abs().max() <= 0.1 + 1e-12
    assert row.sum() == pytest.approx(0.0, abs=1e-12)
    assert row.abs().sum() == pytest.approx(0.4, abs=1e-12)  # gross shrinks to caps


def test_impact_cost_increases_total_cost_and_scales_with_aum():
    panel = generate_daily_panel(n_stocks=40, n_days=340, seed=8)
    sig = cs_zscore(momentum(panel.prices, 252, 21))
    base = run_backtest(panel.prices, sig, BacktestConfig(impact_coef=0.0, aum=None))
    small = run_backtest(panel.prices, sig,
                         BacktestConfig(impact_coef=0.1, aum=10e6),
                         volumes=panel.volumes)
    big = run_backtest(panel.prices, sig,
                       BacktestConfig(impact_coef=0.1, aum=1e9),
                       volumes=panel.volumes)
    c0, c1, c2 = (r.ledger["cost"].sum() for r in (base, small, big))
    assert c1 > c0
    assert c2 > c1  # impact grows with AUM (sqrt law)
    # gross returns identical: costs never feed back into positions
    pd.testing.assert_series_equal(base.ledger["gross_ret"], big.ledger["gross_ret"])


def test_rebalance_band_reduces_turnover_in_backtest():
    panel = generate_daily_panel(n_stocks=50, n_days=350, seed=3)
    sig = cs_zscore(momentum(panel.prices, 126, 21))
    naive = run_backtest(panel.prices, sig, BacktestConfig(rebalance_band=0.0))
    banded = run_backtest(panel.prices, sig, BacktestConfig(rebalance_band=0.3))
    assert banded.ledger["turnover"].sum() < naive.ledger["turnover"].sum()
    # still a live, dollar-neutral book
    active = banded.ledger["gross_exposure"] > 0
    assert active.any()
    assert banded.ledger.loc[active, "net_exposure"].abs().max() < 1e-12


def test_volumes_required_with_aum():
    prices, signal = _toy()
    with pytest.raises(ValueError, match="volumes"):
        run_backtest(prices, signal, BacktestConfig(aum=1e6))


@pytest.mark.parametrize("kwargs", [
    {"n_quantiles": 1}, {"gross_exposure": 0.0}, {"max_weight": -0.1},
    {"linear_cost_bps": -1.0}, {"aum": -5.0}, {"rebalance_band": -0.01},
])
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)
