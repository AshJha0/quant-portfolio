"""Execution convention: decide at the close of t, fill at a modelled t+1 price.

These tests fail if the engine fills a decision taken at the close of t at
that same close (the convention used before this module existed).
"""

import numpy as np
import pandas as pd
import pytest
from fx_regime import ExecutionConfig, StrategyConfig, run_backtest

CCYS = ["AUD", "JPY"]


def _two_leg_case(n=400, seed=0, aud_vol=0.006):
    """AUD moves, JPY is flat; rates equal except a tiny AUD pickup.

    risk_on books long AUD / short JPY, usd_squeeze books short both, so the
    sign of the AUD position is set by the regime label.
    """
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame(
        {"AUD": rng.standard_normal(n) * aud_vol, "JPY": np.zeros(n)}, index=idx
    )
    rates = pd.DataFrame({"AUD": 1e-6, "JPY": 0.0, "USD": 0.0}, index=idx)
    cfg = StrategyConfig(
        cov_window=20, n_carry_long=1, n_carry_short=1, rebalance_every=1,
        haven_currencies=("JPY",), risk_proxy_currencies=("AUD",),
        spread_pips={"AUD": 0.0, "JPY": 0.0},
    )
    return idx, rets, rates, cfg


def test_fill_price_is_interpolated_t_plus_one_price():
    """(a) Hand ledger: the new book earns p_{t+1}/p_fill - 1, the old book
    p_fill/p_t - 1, with p_fill = p_t + f (p_{t+1} - p_t), from price levels."""
    idx, rets, rates, cfg = _two_leg_case(n=60)
    regimes = pd.Series("risk_on", index=idx)
    regimes.iloc[40:] = "usd_squeeze"  # decided at close of idx[40]
    for f in (0.25, 0.5, 1.0):
        bt = run_backtest(rets, rates, regimes, cfg, ExecutionConfig(fill_fraction=f))
        day = idx[41]  # first day the new book is filled
        k = bt.ledger.index.get_loc(day)
        w_old, w_new = bt.weights.iloc[k - 1], bt.weights.iloc[k]
        assert w_old["AUD"] > 0 > w_new["AUD"]
        p = np.exp(rets["AUD"].cumsum())  # AUDUSD level path, any start
        p_t, p_t1 = p.loc[idx[40]], p.loc[day]
        p_fill = p_t + f * (p_t1 - p_t)
        expected = w_old["AUD"] * (p_fill / p_t - 1) + w_new["AUD"] * (p_t1 / p_fill - 1)
        assert bt.ledger.loc[day, "spot"] == pytest.approx(expected, rel=1e-12, abs=1e-16)


def test_perfect_foresight_signal_earns_nothing_with_next_close_fill():
    """(b) s_t = sign(r_{t+1}) with f = 1 must earn about zero gross.

    A same-close fill would earn |w| |R| every day. With f = 1 the new book
    misses day t+1 entirely and then earns sign(r_{t+1}) r_{t+2}, mean zero.
    """
    idx, rets, rates, cfg = _two_leg_case(n=1500, seed=3)
    nxt = rets["AUD"].shift(-1)
    regimes = pd.Series(np.where(nxt > 0, "risk_on", "usd_squeeze"), index=idx)[:-1]
    lag1 = run_backtest(rets, rates, regimes, cfg, ExecutionConfig(fill_fraction=1.0))
    spot = lag1.ledger["spot"].iloc[5:]
    se = spot.std(ddof=1) / np.sqrt(len(spot))
    assert abs(spot.mean()) < 4 * se
    # the same signal filled at the decision close is a money machine; this
    # is what the engine did before the convention was introduced
    same_close = run_backtest(
        rets, rates, regimes, cfg,
        ExecutionConfig(mode="close_auction", auction_cost_bp=0.0),
    )
    sc = same_close.ledger["spot"].iloc[5:]
    assert sc.mean() > 20 * sc.std(ddof=1) / np.sqrt(len(sc))
    assert (sc > -1e-15).all()


def test_close_auction_charges_exactly_the_auction_cost():
    """(c) close_auction cost = spread cost + auction_cost_bp * turnover."""
    idx, rets, rates, _cfg = _two_leg_case(n=200, seed=5)
    regimes = pd.Series(np.where(np.arange(200) % 30 < 15, "risk_on", "usd_squeeze"), index=idx)
    cfg_cost = StrategyConfig(
        cov_window=20, n_carry_long=1, n_carry_short=1, rebalance_every=1,
        haven_currencies=("JPY",), risk_proxy_currencies=("AUD",),
    )
    a = run_backtest(rets, rates, regimes, cfg_cost,
                     ExecutionConfig(mode="close_auction", auction_cost_bp=0.0))
    b = run_backtest(rets, rates, regimes, cfg_cost,
                     ExecutionConfig(mode="close_auction", auction_cost_bp=7.5))
    extra = b.ledger["cost"] - a.ledger["cost"]
    np.testing.assert_allclose(extra, 7.5e-4 * b.ledger["turnover"], rtol=0, atol=1e-16)
    assert b.ledger["turnover"].sum() > 0
    # close-auction books the full day's move on the new book
    day = b.ledger.index[50]
    w = b.weights.loc[day]
    assert b.ledger.loc[day, "spot"] == pytest.approx(
        float(w @ np.expm1(rets.loc[day])), rel=1e-12  # (1+R)/1-1 vs R: rounding only
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"fill_fraction": 0.0},
        {"fill_fraction": 1.5},
        {"fill_fraction": float("nan")},
        {"auction_cost_bp": -1.0},
        {"auction_cost_bp": float("inf")},
        {"mode": "same_close"},
    ],
)
def test_execution_config_rejects_bad_values(kw):
    with pytest.raises(ValueError):
        ExecutionConfig(**kw)


def test_backtest_rejects_non_finite_returns():
    idx, rets, rates, cfg = _two_leg_case(n=60)
    rets.iloc[30, 0] = np.nan
    with pytest.raises(ValueError):
        run_backtest(rets, rates, pd.Series("risk_on", index=idx), cfg)
