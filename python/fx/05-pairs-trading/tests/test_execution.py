"""Execution convention: decide at close t, fill at a modelled t+1 price.

These tests fail if the engine fills a decision taken at the close of t at
that same close (the convention used by this project before the change).
"""

import fx_pairs as fp
import numpy as np
import pandas as pd
import pytest


def _single_leg(prices):
    """Pair 2 constant and beta 0: the spread is pair 1 alone."""
    idx = pd.bdate_range("2021-03-01", periods=len(prices))
    p1 = pd.Series(prices, index=idx, dtype=float)
    p2 = pd.Series(1.0, index=idx)
    return p1, p2


@pytest.mark.parametrize("f", [0.25, 0.5, 1.0])
def test_hand_ledger_fill_price(f):
    # decision +1 at close of day 1, flat at close of day 3
    px = [1.0000, 1.0100, 1.0300, 1.0200, 1.0000]
    p1, p2 = _single_leg(px)
    pos = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    res = fp.run_backtest(p1, p2, pos, 0.0, pip_spread_1=2.0, pip_spread_2=0.0,
                          execution=fp.ExecutionConfig(fill_fraction=f))
    q_buy = px[1] + f * (px[2] - px[1])   # fills during period 2
    q_sell = px[3] + f * (px[4] - px[3])  # fills during period 4
    spot = np.zeros(5)
    spot[2] = np.log(px[2] / q_buy)
    spot[3] = np.log(px[3] / px[2])
    spot[4] = np.log(q_sell / px[3])
    cost = np.zeros(5)
    cost[2] = -0.5 * 2.0 * 1e-4 / q_buy
    cost[4] = -0.5 * 2.0 * 1e-4 / q_sell
    np.testing.assert_allclose(res.spot_pnl.values, spot, rtol=0, atol=1e-15)
    np.testing.assert_allclose(res.cost_pnl.values, cost, rtol=0, atol=1e-18)
    # the round trip earns exactly log(q_sell / q_buy): bought at the modelled
    # t+1 price, never at the decision close
    assert res.spot_pnl.sum() == pytest.approx(np.log(q_sell / q_buy), abs=1e-15)


def test_perfect_foresight_cannot_profit_with_next_close_fill():
    rng = np.random.default_rng(11)
    n = 3000
    r = 0.004 * rng.standard_normal(n)
    px = np.exp(np.cumsum(r))
    p1, p2 = _single_leg(px)
    lr = np.diff(np.log(px))
    signal = np.zeros(n)
    signal[:-1] = np.sign(lr)  # knows the next return at the close of t
    nxt = fp.run_backtest(p1, p2, signal, 0.0, pip_spread_1=0.0, pip_spread_2=0.0,
                          execution=fp.ExecutionConfig(fill_fraction=1.0))
    g = nxt.spot_pnl.values[1:]
    se = g.std(ddof=1) / np.sqrt(g.size)
    assert abs(g.mean()) < 4 * se
    # a same-close fill would earn |r| every day; the old engine did this
    same = fp.run_backtest(p1, p2, signal, 0.0, pip_spread_1=0.0, pip_spread_2=0.0,
                           execution=fp.ExecutionConfig("close_auction",
                                                        auction_cost_bp=0.0))
    assert same.spot_pnl.sum() == pytest.approx(np.abs(lr).sum(), rel=1e-12)


def test_default_is_next_with_half_fill():
    ex = fp.ExecutionConfig()
    assert (ex.mode, ex.fill_fraction, ex.auction_cost_bp) == ("next", 0.5, 10.0)


def test_close_auction_charges_exactly_the_auction_cost():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2021-01-04", periods=200)
    p1 = pd.Series(np.exp(np.cumsum(0.005 * rng.standard_normal(200))), index=idx)
    p2 = pd.Series(np.exp(np.cumsum(0.005 * rng.standard_normal(200))), index=idx)
    pos = np.sign(rng.standard_normal(200))
    beta = 0.7
    a = fp.run_backtest(p1, p2, pos, beta, pip_spread_1=1.0, pip_spread_2=1.0,
                        execution=fp.ExecutionConfig("close_auction", auction_cost_bp=0.0))
    b = fp.run_backtest(p1, p2, pos, beta, pip_spread_1=1.0, pip_spread_2=1.0,
                        execution=fp.ExecutionConfig("close_auction", auction_cost_bp=12.5))
    dn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    expected = -12.5e-4 * dn * (1.0 + abs(beta))
    np.testing.assert_allclose((b.cost_pnl - a.cost_pnl).values, expected, atol=1e-18)
    np.testing.assert_array_equal(a.spot_pnl.values, b.spot_pnl.values)


@pytest.mark.parametrize("kw", [
    {"fill_fraction": 0.0}, {"fill_fraction": 1.5}, {"fill_fraction": float("nan")},
    {"auction_cost_bp": -1.0}, {"auction_cost_bp": float("inf")}, {"mode": "vwap"},
])
def test_execution_config_rejects_bad_input(kw):
    with pytest.raises(ValueError):
        fp.ExecutionConfig(**kw)


def test_execution_must_be_config():
    p1, p2 = _single_leg([1.0, 1.1, 1.2])
    with pytest.raises(ValueError):
        fp.run_backtest(p1, p2, [0, 1, 0], 0.0, execution="next")
