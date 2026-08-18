"""Edge cases (documentation contract item 6): every scenario here is also
described in docs/VALIDATION.md."""

import numpy as np
import pandas as pd
import pytest

from eq_algo import (ACParams, BacktestConfig, IntradayConfig, IntradayMarket,
                     ac_trades, ac_trajectory, cs_zscore, deflated_sharpe_ratio,
                     generate_daily_panel, momentum, pov_schedule, run_backtest,
                     twap_schedule)


def test_one_day_backtest():
    prices = pd.DataFrame([[10.0, 20.0, 30.0, 40.0]],
                          index=pd.bdate_range("2020-01-01", periods=1),
                          columns=list("ABCD"))
    signal = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=prices.index,
                          columns=prices.columns)
    res = run_backtest(prices, signal, BacktestConfig(n_quantiles=2, max_weight=1.0))
    assert len(res.ledger) == 1
    assert res.ledger["gross_ret"].iloc[0] == 0.0     # no return day yet
    assert res.ledger["turnover"].iloc[0] > 0         # but the book was built


def test_single_stock_universe_stays_flat():
    dates = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.0, 11.5, 12.5]}, index=dates)
    signal = pd.DataFrame({"A": [1.0] * 5}, index=dates)
    res = run_backtest(prices, signal, BacktestConfig(n_quantiles=2))
    # cannot build a long-short book from one name -> flat, no costs, no P&L
    assert (res.ledger["gross_exposure"] == 0.0).all()
    assert (res.ledger["net_ret"] == 0.0).all()


def test_all_nan_signal_is_flat():
    panel = generate_daily_panel(n_stocks=10, n_days=300, seed=0)
    signal = panel.prices * np.nan
    res = run_backtest(panel.prices, signal)
    assert (res.ledger["net_ret"] == 0.0).all()
    assert (res.ledger["turnover"] == 0.0).all()


def test_parent_order_larger_than_day_volume_informative_error():
    # 2m shares vs 1m ADV at a 20% cap -> must fail loudly, not silently clip
    vols = np.full(26, 1_000_000.0 / 26)
    with pytest.raises(ValueError) as exc:
        pov_schedule(2_000_000.0, vols, participation=0.20)
    assert "Split across days" in str(exc.value)
    # and even at 100% participation it cannot be done in-day
    with pytest.raises(ValueError):
        pov_schedule(2_000_000.0, vols, participation=1.0)


def test_zero_volume_bucket_pov_skips_it():
    vols = np.array([1000.0, 0.0, 1000.0, 1000.0])
    q = pov_schedule(600.0, vols, participation=0.25)
    assert q[1] == 0.0                    # nothing scheduled into the halt
    assert q.sum() == pytest.approx(600.0, rel=1e-12)


def test_zero_risk_aversion_ac_is_twap():
    p = ACParams(total_shares=10_000.0, n_slices=13)
    np.testing.assert_allclose(ac_trades(p, 0.0), twap_schedule(10_000.0, 13),
                               rtol=1e-12)


def test_single_slice_ac():
    p = ACParams(total_shares=5_000.0, n_slices=1, eta=1e-4, gamma=1e-6)
    for lam in (0.0, 1e-4):
        x = ac_trajectory(p, lam)
        np.testing.assert_allclose(x, [5_000.0, 0.0], atol=1e-9)
        n = ac_trades(p, lam)
        assert n[0] == pytest.approx(5_000.0, rel=1e-12)


def test_single_bucket_intraday_market():
    cfg = IntradayConfig(n_buckets=1, price_noise=0.0, temp_coef=0.0,
                         perm_coef=0.0)
    mkt = IntradayMarket(cfg)
    res = mkt.execute(np.array([1000.0]), side=1, seed=0)
    assert res.avg_price == pytest.approx(
        cfg.mid0 * (1 + cfg.spread_bps * 1e-4 / 2), rel=1e-12)


def test_deflated_sharpe_edge_inputs():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio([0.01, 0.01, 0.01, 0.01], n_trials=5)  # zero vol
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(rng.standard_normal(100) * 0.01, n_trials=0)


def test_momentum_with_missing_history_is_nan_not_wrong():
    prices = pd.DataFrame({"A": [np.nan] * 260 + [100.0, 101.0]})
    m = momentum(prices, 252, 21)
    assert m["A"].isna().all()            # missing history never fabricates a value


def test_backtest_with_nan_prices_mid_sample():
    """A name that delists mid-sample (prices become NaN) leaves the universe
    without corrupting the ledger."""
    panel = generate_daily_panel(n_stocks=30, n_days=330, seed=6)
    prices = panel.prices.copy()
    prices.iloc[300:, 0] = np.nan         # delist S000
    sig = cs_zscore(momentum(prices, 126, 21))
    res = run_backtest(prices, sig, BacktestConfig(n_quantiles=3))
    assert np.isfinite(res.ledger["net_ret"]).all()
    assert (res.weights.iloc[301:, 0].fillna(0.0) == 0.0).all()
