"""Execution convention: decide at the close of t, fill at a modelled t+1 price.

These tests fail if the engine fills a decision taken at the close of t at
that same close (the convention used before this module existed).
"""

import numpy as np
import pandas as pd
import pytest
from fx_regime import ExecutionConfig, StrategyConfig, run_backtest
from fx_regime.strategy import regime_weights

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


def _target(rets, rates, regime, t, cfg):
    """The book the strategy targets for day t (decided at close t-1)."""
    p = rets.shape[1]
    cov = np.cov(rets.to_numpy()[t - cfg.cov_window : t].T, ddof=1).reshape(p, p)
    return regime_weights(regime, list(rets.columns), rates.iloc[t - 1], cov, cfg).to_numpy()


def test_fill_price_is_interpolated_t_plus_one_price():
    """(a) Hand ledger from price levels, no costs. Before the fill the book
    held overnight, h, earns p_fill/p_t - 1; the trade resets it to the
    target w of the NAV at the fill, which then earns p_{t+1}/p_fill - 1:
    1 + net = (1 + h . a)(1 + w . b), p_fill = p_t + f (p_{t+1} - p_t)."""
    idx, rets, rates, cfg = _two_leg_case(n=60)
    regimes = pd.Series("risk_on", index=idx)
    regimes.iloc[40:] = "usd_squeeze"  # decided at close of idx[40]
    for f in (0.25, 0.5, 1.0):
        bt = run_backtest(rets, rates, regimes, cfg, ExecutionConfig(fill_fraction=f))
        day = idx[41]  # first day the new book is filled
        k = bt.ledger.index.get_loc(day)
        h = bt.weights.iloc[k - 1].to_numpy()
        w = _target(rets, rates, "usd_squeeze", 41, cfg)
        assert h[0] > 0 > w[0]
        p = np.exp(rets["AUD"].cumsum())  # AUDUSD level path, any start
        p_t, p_t1 = p.loc[idx[40]], p.loc[day]
        p_fill = p_t + f * (p_t1 - p_t)
        a = np.array([p_fill / p_t - 1, 0.0])
        b = np.array([p_t1 / p_fill - 1, 0.0])
        expected = (1 + h @ a) * (1 + w @ b) - 1
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
    """(c) close_auction premium = auction_cost_bp * traded notional."""
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
    # the auction premium is paid on the notional actually traded (turnover
    # depends on the post-trade NAV, so compare within run b): with zero
    # spreads the whole cost is the premium
    z = run_backtest(rets, rates, regimes, _cfg,
                     ExecutionConfig(mode="close_auction", auction_cost_bp=7.5))
    np.testing.assert_allclose(z.ledger["cost"], 7.5e-4 * z.ledger["turnover"], rtol=0,
                               atol=1e-16)
    assert (a.ledger["cost"] > 0).any()
    assert b.ledger["turnover"].sum() > 0
    # close-auction: filled at close t-1, so the new book earns the whole day:
    # spot = N (w . R) with N the NAV after paying for the trade
    day = b.ledger.index[50]
    t = rets.index.get_loc(day)
    w = _target(rets, rates, regimes.iloc[t - 1], t, cfg_cost)
    n_post = 1.0 - b.ledger.loc[day, "cost"]
    assert b.ledger.loc[day, "spot"] == pytest.approx(
        n_post * float(w @ np.expm1(rets.loc[day])), rel=1e-12, abs=1e-16
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
