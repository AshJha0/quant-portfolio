"""Backtester: exact toy ledger, session spreads, carry, no-lookahead."""

import numpy as np
import pandas as pd
import pytest

from fx_algo import BacktestConfig, IntradayBacktester, build_bars, generate_ticks

PIP = 1e-4
SPREADS = {"asia": 0.6, "london": 0.4, "overlap": 0.2, "ny": 0.4, "late": 1.0}


def toy_bars(hours, closes):
    idx = pd.Index(np.asarray(hours, dtype=float), name="bar_end_hours")
    return pd.DataFrame(
        {"close": np.asarray(closes, dtype=float), "hour": np.mod(hours, 24.0)}, index=idx
    )


def test_ledger_exact_on_toy_scenario_with_session_spreads():
    bars = toy_bars([8, 9, 10, 11], [1.1000, 1.1010, 1.0990, 1.1005])
    pos = pd.Series([1.0, -2.0, 0.0, 0.0], index=bars.index)
    bt = IntradayBacktester(BacktestConfig(pip_size=PIP, spread_pips_by_session=SPREADS))
    ledger, summary = bt.run(bars, pos)

    # gross: pos_{t-1} * (close_t - close_{t-1})
    assert np.allclose(ledger["gross_pips"], [0.0, 10.0, 40.0, 0.0])
    # cost: |dpos| * half spread of the bar-t session (hours 8..10 london=0.4)
    assert np.allclose(ledger["cost_pips"], [1 * 0.2, 3 * 0.2, 2 * 0.2, 0.0])
    assert (ledger["carry_pips"] == 0.0).all()
    assert summary["net_pips"] == pytest.approx(50.0 - 1.2)
    assert summary["n_trades"] == 3
    assert summary["turnover"] == pytest.approx(6.0)
    # base-ccy P&L = quote P&L / spot
    assert np.allclose(ledger["net_base"], ledger["net_quote"] / ledger["close"])


def test_session_spread_applied_at_trade_bar_session():
    bars = toy_bars([22, 23], [1.1, 1.1])  # late session: spread 1.0 -> half 0.5
    pos = pd.Series([1.0, 1.0], index=bars.index)
    bt = IntradayBacktester(BacktestConfig(pip_size=PIP, spread_pips_by_session=SPREADS))
    ledger, _ = bt.run(bars, pos)
    assert ledger["cost_pips"].iloc[0] == pytest.approx(0.5)


def test_carry_accrual_hand_exact():
    # long 1.0 held over the 21:00 rollover; r_base=3%, r_quote=1%, S=1.1
    bars = toy_bars([20, 21, 22], [1.1, 1.1, 1.1])
    pos = pd.Series([1.0, 1.0, 0.0], index=bars.index)
    cfg = BacktestConfig(
        pip_size=PIP, spread_pips_by_session=SPREADS, r_base=0.03, r_quote=0.01
    )
    ledger, summary = IntradayBacktester(cfg).run(bars, pos)
    expected_quote = 1.0 * 1.1 * 0.02 / 365.0
    assert ledger["carry_quote"].iloc[1] == pytest.approx(expected_quote, abs=1e-15)
    assert ledger["carry_quote"].iloc[0] == 0.0
    assert ledger["carry_quote"].iloc[2] == 0.0
    assert summary["carry_pips"] == pytest.approx(expected_quote / PIP)


def test_negative_carry_costs_short_base_rate_advantage():
    bars = toy_bars([20, 21], [1.1, 1.1])
    pos = pd.Series([-1.0, -1.0], index=bars.index)
    cfg = BacktestConfig(pip_size=PIP, r_base=0.03, r_quote=0.01)
    ledger, _ = IntradayBacktester(cfg).run(bars, pos)
    assert ledger["carry_quote"].iloc[1] == pytest.approx(-1.1 * 0.02 / 365.0)


def test_rollover_detected_across_midnight_wrap():
    cfg = BacktestConfig(pip_size=PIP, rollover_hour=21.0, r_base=0.02, r_quote=0.0)
    bars = toy_bars([20.5, 22.5], [1.0, 1.0])  # 2h bars crossing 21:00
    pos = pd.Series([1.0, 0.0], index=bars.index)
    ledger, _ = IntradayBacktester(cfg).run(bars, pos)
    assert ledger["carry_quote"].iloc[1] > 0


def test_no_lookahead_pnl_uses_lagged_position():
    rng = np.random.default_rng(0)
    closes = 1.1 + 0.001 * np.cumsum(rng.standard_normal(50))
    bars = toy_bars(np.arange(8, 58) % 24, closes)
    # "cheating" positions equal to the sign of the SAME bar's return
    cheat = pd.Series(np.sign(np.concatenate([[0.0], np.diff(closes)])), index=bars.index)
    bt = IntradayBacktester(BacktestConfig(pip_size=PIP, spread_pips_by_session=SPREADS))
    ledger, _ = bt.run(bars, cheat)
    # engine must produce the honest (lagged) gross, not the cheat gross
    honest = np.concatenate([[0.0], cheat.to_numpy()[:-1] * np.diff(closes)])
    cheat_gross = np.concatenate([[0.0], cheat.to_numpy()[1:] * np.diff(closes)])
    assert np.allclose(ledger["gross_quote"], honest, atol=1e-15)
    assert not np.allclose(ledger["gross_quote"].sum(), cheat_gross.sum())


def test_no_lookahead_future_mutation_leaves_past_ledger_unchanged():
    ticks = generate_ticks(n_days=4, seed=2)
    bars = build_bars(ticks, 1.0)[["close", "hour"]]
    rng = np.random.default_rng(1)
    pos = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=len(bars)), index=bars.index)
    bt = IntradayBacktester(BacktestConfig(pip_size=PIP, spread_pips_by_session=SPREADS))
    ledger, _ = bt.run(bars, pos)

    cutoff = 48.0
    bars_mut = bars.copy()
    future = bars_mut.index > cutoff
    bars_mut.loc[future, "close"] *= 1.01
    ledger_mut, _ = bt.run(bars_mut, pos)
    past = ledger.index <= cutoff
    pd.testing.assert_frame_equal(ledger.loc[past], ledger_mut.loc[past])


def test_misaligned_positions_raise():
    bars = toy_bars([8, 9], [1.1, 1.1])
    bt = IntradayBacktester(BacktestConfig(pip_size=PIP))
    with pytest.raises(ValueError):
        bt.run(bars, pd.Series([1.0], index=[8.0]))
    with pytest.raises(ValueError):
        bt.run(bars, pd.Series([np.nan, 1.0], index=bars.index))


def test_invalid_pip_size_raises():
    with pytest.raises(ValueError):
        IntradayBacktester(BacktestConfig(pip_size=0.0))


def test_zero_positions_zero_pnl():
    bars = toy_bars([8, 9, 10], [1.1, 1.2, 1.0])
    pos = pd.Series(np.zeros(3), index=bars.index)
    ledger, summary = IntradayBacktester(BacktestConfig(pip_size=PIP)).run(bars, pos)
    assert summary["net_pips"] == 0.0
    assert (ledger["net_quote"] == 0.0).all()
