"""Execution convention: decide at close t, fill at a modelled t+1 price.

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost.
"""

import numpy as np
import pandas as pd
import pytest
from eq_algo import BacktestConfig, ExecutionConfig, run_backtest


def _toy():
    dates = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0],
         [11.0, 19.0, 33.0, 38.0],
         [11.0, 19.0, 33.0, 38.0]],
        index=dates, columns=list("ABCD"))
    signal = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]] * 3, index=dates, columns=list("ABCD"))
    return prices, signal


CFG = BacktestConfig(n_quantiles=2, gross_exposure=2.0, max_weight=1.0,
                     linear_cost_bps=10.0, impact_coef=0.0)


@pytest.mark.parametrize("f", [0.5, 0.25, 1.0])
def test_fill_price_is_interpolated_next_price(f):
    prices, signal = _toy()
    led = run_backtest(prices, signal, CFG, execution=ExecutionConfig(fill_fraction=f)).ledger
    # day 0: target decided at close 0, nothing filled yet
    assert led["turnover"].iloc[0] == 0.0 and led["gross_ret"].iloc[0] == 0.0
    # day 1: fill at p0 + f (p1 - p0); new weights earn p1 / fill - 1
    w = np.array([-0.5, -0.5, 0.5, 0.5])
    p0, p1 = prices.iloc[0].to_numpy(), prices.iloc[1].to_numpy()
    fill = p0 + f * (p1 - p0)
    expected = float(w @ (p1 / fill - 1.0))
    assert led["gross_ret"].iloc[1] == pytest.approx(expected, abs=1e-15)
    assert led["turnover"].iloc[1] == pytest.approx(2.0, abs=1e-12)
    assert led["cost"].iloc[1] == pytest.approx(2.0 * 10e-4, abs=1e-15)
    # day 2: flat prices, no trade
    assert led["gross_ret"].iloc[2] == pytest.approx(0.0, abs=1e-15)
    assert led["turnover"].iloc[2] == pytest.approx(0.0, abs=1e-12)


def test_close_auction_charges_exactly_the_auction_cost():
    prices, signal = _toy()
    ex = ExecutionConfig(mode="close_auction", auction_cost_bp=7.0)
    led = run_backtest(prices, signal, CFG, execution=ex).ledger
    # fills at close 0: turnover 2.0 on day 0, cost (10 + 7) bp per unit
    assert led["turnover"].iloc[0] == pytest.approx(2.0, abs=1e-12)
    assert led["cost"].iloc[0] == pytest.approx(2.0 * 17e-4, abs=1e-15)
    # the weights then earn the whole close-to-close return of day 1
    w = np.array([-0.5, -0.5, 0.5, 0.5])
    r1 = (prices.iloc[1] / prices.iloc[0] - 1.0).to_numpy()
    assert led["gross_ret"].iloc[1] == pytest.approx(float(w @ r1), abs=1e-15)
    base = run_backtest(prices, signal, CFG,
                        execution=ExecutionConfig(mode="close_auction", auction_cost_bp=0.0))
    diff = led["cost"] - base.ledger["cost"]
    np.testing.assert_allclose(diff, 7e-4 * led["turnover"], atol=1e-15)


def _foresight_panel(n_days=2000, n_names=20, seed=0):
    rng = np.random.default_rng(seed)
    r = 0.01 * rng.standard_normal((n_days, n_names))
    dates = pd.bdate_range("2010-01-01", periods=n_days)
    prices = pd.DataFrame(100.0 * np.cumprod(1.0 + r, axis=0), index=dates,
                          columns=[f"S{i}" for i in range(n_names)])
    rets = prices.pct_change()
    signal = rets.shift(-1)       # perfect knowledge of r_{t+1} at close t
    return prices, signal


def test_perfect_foresight_earns_nothing_when_filled_at_next_close():
    """Fails if the engine fills a decision taken at t at t's own close."""
    prices, signal = _foresight_panel()
    cfg = BacktestConfig(n_quantiles=5, linear_cost_bps=0.0, impact_coef=0.0, max_weight=1.0)
    g = run_backtest(prices, signal, cfg,
                     execution=ExecutionConfig(fill_fraction=1.0)).ledger["gross_ret"].iloc[5:-5]
    se = g.std(ddof=1) / np.sqrt(len(g))
    assert abs(g.mean()) < 4.0 * se
    # the same signal filled at t's close (close-auction, zero auction cost)
    # earns the full cross-sectional spread of r_{t+1}: a money machine
    cheat = run_backtest(prices, signal, cfg,
                         execution=ExecutionConfig(mode="close_auction", auction_cost_bp=0.0))
    gc = cheat.ledger["gross_ret"].iloc[5:-5]
    assert gc.mean() > 20.0 * se
    # with the default f = 0.5 roughly half the move is lost before the fill
    half = run_backtest(prices, signal, cfg).ledger["gross_ret"].iloc[5:-5]
    assert 0.3 * gc.mean() < half.mean() < 0.7 * gc.mean()


@pytest.mark.parametrize("kw", [
    {"fill_fraction": 0.0}, {"fill_fraction": 1.5}, {"fill_fraction": float("nan")},
    {"auction_cost_bp": -1.0}, {"auction_cost_bp": float("inf")}, {"mode": "same_close"},
])
def test_execution_config_rejects_bad_values(kw):
    with pytest.raises(ValueError):
        ExecutionConfig(**kw)


def test_borrow_fee_charged_on_short_gross():
    prices, signal = _toy()
    cfg = BacktestConfig(n_quantiles=2, gross_exposure=2.0, max_weight=1.0,
                         linear_cost_bps=0.0, impact_coef=0.0, borrow_bps_annual=252.0)
    led = run_backtest(prices, signal, cfg).ledger
    # short gross 1.0 held from day 1: 252 bp / 252 = 1 bp per day
    assert led["borrow"].iloc[0] == 0.0
    np.testing.assert_allclose(led["borrow"].iloc[1:], 1e-4, atol=1e-18)
    np.testing.assert_allclose(led["net_ret"], led["gross_ret"] - led["cost"] - led["borrow"])


@pytest.mark.parametrize("kw", [{"impact_coef": float("nan")}, {"aum": float("inf")},
                                {"borrow_bps_annual": -1.0}, {"linear_cost_bps": float("inf")}])
def test_backtest_config_rejects_non_finite(kw):
    with pytest.raises(ValueError):
        BacktestConfig(**kw)


def test_rejects_non_positive_prices():
    prices, signal = _toy()
    bad = prices.copy()
    bad.iloc[1, 0] = -1.0
    with pytest.raises(ValueError):
        run_backtest(bad, signal, CFG)
