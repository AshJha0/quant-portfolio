"""Shared execution convention: decide at close t, fill at a modelled t+1 price.

Signals use data up to the close of period t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as hst

from fx_algo import BacktestConfig, ExecutionConfig, IntradayBacktester

PIP = 1e-4
FLAT = {"asia": 0.4, "london": 0.4, "overlap": 0.4, "ny": 0.4, "late": 0.4}


def toy_bars(hours, closes):
    idx = pd.Index(np.asarray(hours, dtype=float), name="bar_end_hours")
    return pd.DataFrame(
        {"close": np.asarray(closes, dtype=float), "hour": np.mod(hours, 24.0)}, index=idx
    )


def bt():
    return IntradayBacktester(BacktestConfig(pip_size=PIP, spread_pips_by_session=FLAT))


def test_default_is_next_with_half_fill():
    ex = ExecutionConfig()
    assert ex.mode == "next" and ex.fill_fraction == 0.5


def test_hand_ledger_fill_at_modelled_next_price():
    # closes 1.1000, 1.1010, 1.0990, 1.1005; buy 1 at close of bar 0.
    bars = toy_bars([8, 9, 10, 11], [1.1000, 1.1010, 1.0990, 1.1005])
    pos = pd.Series([1.0, -2.0, 0.0, 0.0], index=bars.index)
    ledger, s = bt().run(bars, pos, execution=ExecutionConfig(fill_fraction=0.25))
    # fills: bar1 at 1.1000 + .25*.0010; bar2 at 1.1010 - .25*.0020; bar3 at 1.0990 + .25*.0015
    assert ledger["fill_price"].iloc[1] == pytest.approx(1.10025, abs=1e-12)
    assert ledger["fill_price"].iloc[2] == pytest.approx(1.1005, abs=1e-12)
    assert ledger["fill_price"].iloc[3] == pytest.approx(1.099375, abs=1e-12)
    assert np.isnan(ledger["fill_price"].iloc[0])
    # gross pips: bar1 0*2.5 + 1*7.5; bar2 1*(-5) + (-2)*(-15); bar3 (-2)*3.75 + 0
    assert np.allclose(ledger["gross_pips"], [0.0, 7.5, 25.0, -7.5])
    # cost charged in the bar the order fills: |dpos| * 0.2 pip
    assert np.allclose(ledger["cost_pips"], [0.0, 0.2, 0.6, 0.4])


def test_perfect_foresight_does_not_pay_with_next_close_fill():
    # s_t = sign(r_{t+1}). A same-close fill earns sum |r|; a t+1 fill with
    # f = 1 only earns sign(r_{t+1}) * r_{t+2}, mean zero on a random walk.
    rng = np.random.default_rng(11)
    n = 4000
    closes = 1.1 + 0.0005 * np.cumsum(rng.standard_normal(n))
    d = np.diff(closes)
    sig = np.concatenate([np.sign(d), [0.0]])
    bars = toy_bars(np.arange(n) % 24, closes)
    ledger, _ = bt().run(bars, pd.Series(sig, index=bars.index),
                         execution=ExecutionConfig(fill_fraction=1.0))
    g = ledger["gross_quote"].to_numpy()[2:]
    se = g.std(ddof=1) / np.sqrt(g.size)
    assert abs(g.mean()) < 4 * se
    same_close = np.abs(d).mean()  # what a same-close fill would earn per bar
    assert g.mean() < 0.1 * same_close


def test_close_auction_charges_exact_extra_cost():
    bars = toy_bars([8, 9, 10], [1.2, 1.2, 1.2])
    pos = pd.Series([1.0, -1.0, 0.0], index=bars.index)
    led0, _ = bt().run(bars, pos, execution=ExecutionConfig("close_auction", auction_cost_bp=0.0))
    led1, _ = bt().run(bars, pos, execution=ExecutionConfig("close_auction", auction_cost_bp=0.5))
    extra = (led1["cost_quote"] - led0["cost_quote"]).to_numpy()
    # turnover 1, 2, 1 at price 1.2, 0.5 bp each
    assert np.allclose(extra, np.array([1.0, 2.0, 1.0]) * 0.5e-4 * 1.2, atol=1e-16)
    assert np.allclose(led0["cost_pips"], [0.2, 0.4, 0.2])


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, np.nan, np.inf])
def test_fill_fraction_validation(bad):
    with pytest.raises(ValueError):
        ExecutionConfig(fill_fraction=bad)


def test_bad_mode_and_cost_rejected():
    with pytest.raises(ValueError):
        ExecutionConfig(mode="same_close")
    with pytest.raises(ValueError):
        ExecutionConfig(mode="close_auction", auction_cost_bp=np.nan)
    with pytest.raises(ValueError):
        ExecutionConfig(auction_cost_bp=-1.0)


@settings(max_examples=40, deadline=None)
@given(f=hst.floats(0.05, 1.0), seed=hst.integers(0, 1000))
def test_gross_pnl_equals_position_path_integral(f, seed):
    # Property: total gross P&L equals sum over bars of the held position times
    # the price move, where the held position switches at the modelled fill.
    rng = np.random.default_rng(seed)
    n = 30
    closes = 1.0 + 0.001 * np.cumsum(rng.standard_normal(n))
    pos = rng.choice([-1.0, 0.0, 1.0], size=n)
    bars = toy_bars(np.arange(n) % 24, closes)
    ledger, _ = bt().run(bars, pd.Series(pos, index=bars.index),
                         execution=ExecutionConfig(fill_fraction=f))
    total = 0.0
    for t in range(n - 1):
        fill = closes[t] + f * (closes[t + 1] - closes[t])
        old = pos[t - 1] if t >= 1 else 0.0
        total += old * (fill - closes[t]) + pos[t] * (closes[t + 1] - fill)
    assert ledger["gross_quote"].sum() == pytest.approx(total, abs=1e-12)
