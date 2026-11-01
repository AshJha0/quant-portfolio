"""Fill-timing convention.

Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t); a close-auction variant fills at p_t and pays
an auction cost. These tests fail on an engine that fills a day-t decision at
the close of day t for free.
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as hst

from eq_signal_backtest import ExecutionConfig, run_backtest, strategy_returns


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


@pytest.mark.parametrize("f", [0.25, 0.5, 1.0])
def test_hand_ledger_fill_price(f):
    # Decision to go long at close of day 1 (p=100), filled on day 2.
    prices = _series([100.0, 100.0, 110.0, 121.0])
    signal = _series([0.0, 1.0, 1.0, 0.0])
    rets, pos, trades = strategy_returns(prices, signal, cost_bps=0.0,
                                         execution=ExecutionConfig("next", f))
    p_fill = 100.0 + f * (110.0 - 100.0)
    assert rets.iloc[2] == pytest.approx(110.0 / p_fill - 1.0, rel=1e-14)
    # Day 3: held all day from 110 to 121.
    assert rets.iloc[3] == pytest.approx(0.1, rel=1e-14)
    assert pos.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert trades.tolist() == [0.0, 0.0, 1.0, 0.0]


def test_hand_ledger_exit_keeps_part_of_next_day():
    # Exit decided at close of day 2 (p=110), filled on day 3 at 110 + 0.5*11.
    prices = _series([100.0, 100.0, 110.0, 121.0])
    signal = _series([1.0, 1.0, 0.0, 0.0])
    rets, _, _ = strategy_returns(prices, signal, cost_bps=0.0)
    assert rets.iloc[3] == pytest.approx(115.5 / 110.0 - 1.0, rel=1e-14)


def _foresight_case(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    r = 0.01 * rng.standard_normal(n)
    prices = pd.Series(100.0 * np.cumprod(1 + r),
                       index=pd.bdate_range("2000-01-03", periods=n))
    nxt = prices.pct_change().shift(-1)
    signal = (nxt > 0).astype(float)  # long only on days before an up day
    return prices, signal


def test_perfect_foresight_earns_nothing_with_next_close_fill():
    prices, signal = _foresight_case()
    rets, _, _ = strategy_returns(prices, signal, 0.0, ExecutionConfig("next", 1.0))
    x = rets.to_numpy()[2:]
    se = x.std(ddof=1) / math.sqrt(x.size)
    assert abs(x.mean()) < 4 * se
    # A same-close fill (the bug this test guards) would earn about E|r|
    # per long day; check that it is far away.
    cheat = (signal.shift(1) * prices.pct_change()).to_numpy()[2:]
    assert cheat.mean() > 20 * se  # about 37 SE on this path


def test_close_auction_charges_exact_auction_cost():
    prices = _series([100.0, 101.0, 102.0, 103.0, 104.0])
    signal = _series([0.0, 1.0, 1.0, 0.0, 0.0])
    ex = ExecutionConfig("close_auction", auction_cost_bp=10.0)
    with_cost, _, trades = strategy_returns(prices, signal, 5.0, ex)
    free, _, _ = strategy_returns(prices, signal, 5.0,
                                  ExecutionConfig("close_auction", auction_cost_bp=0.0))
    diff = (free - with_cost).to_numpy()
    np.testing.assert_allclose(diff, trades.to_numpy() * 10.0 / 10_000, atol=1e-15)
    assert trades.sum() == 2


@pytest.mark.parametrize(
    "kw",
    [dict(mode="open"), dict(fill_fraction=0.0), dict(fill_fraction=1.5),
     dict(fill_fraction=float("nan")), dict(auction_cost_bp=-1.0),
     dict(auction_cost_bp=float("inf"))],
)
def test_execution_config_rejects_bad_input(kw):
    with pytest.raises(ValueError):
        ExecutionConfig(**kw)


def test_run_backtest_rejects_non_config():
    prices = _series([100.0, 101.0, 102.0])
    with pytest.raises(ValueError):
        run_backtest(prices, _series([0.0, 1.0, 1.0]), execution="next")


@settings(max_examples=60, deadline=None)
@given(seed=hst.integers(0, 10_000), f=hst.floats(0.05, 1.0))
def test_always_long_compounds_to_buy_and_hold_after_first_fill(seed, f):
    # Long from the first fill onwards: equity equals p_T / p_fill exactly.
    rng = np.random.default_rng(seed)
    p = 100.0 * np.cumprod(1 + 0.02 * rng.standard_normal(30))
    prices = _series(p)
    res = run_backtest(prices, _series(np.ones(30)), 0.0, ExecutionConfig("next", f))
    p_fill = p[0] + f * (p[1] - p[0])
    assert res.equity.iloc[-1] == pytest.approx(p[-1] / p_fill, rel=1e-10)
