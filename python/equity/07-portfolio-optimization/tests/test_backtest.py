"""Tests for eq_port.backtest: no-lookahead guarantee, exact turnover and
cost accounting, drift mechanics, and the strategy zoo."""

import numpy as np
import pandas as pd
import pytest

from eq_port.backtest import (
    make_erc_strategy,
    make_min_variance_strategy,
    make_static_strategy,
    make_tangency_strategy,
    run_backtest,
    run_race,
    strategy_equal_weight,
)
from eq_port.data import generate_panel

PANEL = generate_panel(n_assets=4, n_periods=320, seed=31)
RET = PANEL.returns


# ----------------------------------------------------------------- mechanics

def test_equal_weight_daily_rebalance_matches_mean_return_exactly():
    res = run_backtest(RET, strategy_equal_weight, window=20, rebalance_every=1)
    expected = RET.iloc[20:].mean(axis=1)
    np.testing.assert_allclose(res.net_returns.to_numpy(), expected.to_numpy(), atol=1e-15)
    assert res.costs.sum() == 0.0


def test_buy_and_hold_matches_direct_wealth_computation():
    """One rebalance, then drift only: portfolio wealth must equal the
    weighted sum of asset wealth paths (exact drift accounting)."""
    w0 = np.array([0.4, 0.3, 0.2, 0.1])
    res = run_backtest(
        RET, make_static_strategy(w0), window=10, rebalance_every=10_000
    )
    port_wealth = float(np.prod(1.0 + res.net_returns.to_numpy()))
    asset_growth = np.prod(1.0 + RET.iloc[10:].to_numpy(), axis=0)
    np.testing.assert_allclose(port_wealth, float(w0 @ asset_growth), rtol=1e-12)


def test_no_lookahead_window_contents():
    """The estimation frame passed to the strategy must end strictly
    before the rebalance day, and match returns.iloc[t-window:t] exactly."""
    windows: list[pd.DataFrame] = []

    def spy(est: pd.DataFrame) -> np.ndarray:
        windows.append(est)
        return np.full(est.shape[1], 1.0 / est.shape[1])

    window, k = 30, 7
    run_backtest(RET, spy, window=window, rebalance_every=k)
    reb_positions = range(window, len(RET), k)
    assert len(windows) == len(list(reb_positions))
    for est, t in zip(windows, reb_positions):
        pd.testing.assert_frame_equal(est, RET.iloc[t - window : t])
        # last estimation date strictly before the rebalance date
        assert est.index[-1] < RET.index[t]


def test_no_lookahead_cheat_detection():
    """Asset 0 spikes +100% exactly on the first backtest day. A
    momentum strategy must NOT be able to position for it: the window
    ends the day before."""
    r = np.zeros((3, 2))
    r[0] = [0.0, 0.01]
    r[1] = [0.0, 0.01]
    r[2] = [1.0, 0.01]  # spike, first (and only) backtest day
    df = pd.DataFrame(r)

    def pick_best_mean(est: pd.DataFrame) -> np.ndarray:
        w = np.zeros(est.shape[1])
        w[int(np.argmax(est.mean(axis=0)))] = 1.0
        return w

    res = run_backtest(df, pick_best_mean, window=2, rebalance_every=1)
    # a leaky backtester would return 1.0 here; the honest one holds asset 1
    assert res.net_returns.iloc[0] == pytest.approx(0.01, abs=1e-15)


def test_hand_computed_two_rebalance_cost_ledger():
    """Exact ledger on a 2-rebalance scenario, cost = 100bp on turnover."""
    r = pd.DataFrame(
        [[0.0, 0.0], [0.10, -0.05], [0.02, 0.02], [0.0, 0.0]],
        columns=["A", "B"],
    )
    res = run_backtest(
        r, make_static_strategy(np.array([0.6, 0.4])),
        window=1, rebalance_every=2, cost_bps=100.0,
    )
    # rebalance 1 at t=1: buy-in turnover 1.0, cost 0.01
    # gross_1 = 0.6*0.10 + 0.4*(-0.05) = 0.04 ; net_1 = 0.03
    # drift: wA = 0.66/1.04, wB = 0.38/1.04
    # t=2 (no rebalance): gross = 0.02 (both assets +2%), weights unchanged
    # rebalance 2 at t=3: turnover = |0.6-0.66/1.04| + |0.4-0.38/1.04|
    wA, wB = 0.66 / 1.04, 0.38 / 1.04
    to2 = abs(0.6 - wA) + abs(0.4 - wB)
    np.testing.assert_allclose(res.turnover.to_numpy(), [1.0, to2], atol=1e-15)
    np.testing.assert_allclose(res.costs.to_numpy(), [0.01, 0.01 * to2], atol=1e-15)
    np.testing.assert_allclose(res.gross_returns.to_numpy(), [0.04, 0.02, 0.0], atol=1e-15)
    np.testing.assert_allclose(
        res.net_returns.to_numpy(), [0.03, 0.02, -0.01 * to2], atol=1e-15
    )
    assert res.total_cost == pytest.approx(0.01 + 0.01 * to2, abs=1e-15)


def test_first_rebalance_turnover_is_buy_in():
    res = run_backtest(RET, strategy_equal_weight, window=15, rebalance_every=21,
                       cost_bps=10.0)
    assert res.turnover.iloc[0] == pytest.approx(1.0, abs=1e-12)
    assert res.costs.iloc[0] == pytest.approx(10.0 / 1e4, abs=1e-15)


def test_levered_weights_cash_convention():
    """Weights summing to 0.5 earn half the asset return (cash at 0%)."""
    half = make_static_strategy(np.array([0.5, 0.0, 0.0, 0.0]))
    res = run_backtest(RET, half, window=5, rebalance_every=1)
    expected = 0.5 * RET.iloc[5:, 0].to_numpy()
    np.testing.assert_allclose(res.net_returns.to_numpy(), expected, atol=1e-15)


def test_costs_scale_linearly_with_cost_bps():
    res1 = run_backtest(RET, strategy_equal_weight, window=20, rebalance_every=10,
                        cost_bps=5.0)
    res2 = run_backtest(RET, strategy_equal_weight, window=20, rebalance_every=10,
                        cost_bps=10.0)
    np.testing.assert_allclose(2.0 * res1.costs.to_numpy(), res2.costs.to_numpy(),
                               atol=1e-16)
    np.testing.assert_allclose(res1.turnover.to_numpy(), res2.turnover.to_numpy())


# -------------------------------------------------------------- strategy zoo

def test_run_race_runs_all_strategies_on_same_span():
    strategies = {
        "EW": strategy_equal_weight,
        "MinVar": make_min_variance_strategy(),
        "Tan-JS": make_tangency_strategy(shrink_mean=True),
        "ERC": make_erc_strategy(),
        "Static": make_static_strategy(np.full(4, 0.25)),
    }
    race = run_race(RET, strategies, window=60, rebalance_every=21, cost_bps=10)
    assert set(race) == set(strategies)
    spans = {tuple(res.net_returns.index) for res in race.values()}
    assert len(spans) == 1  # identical evaluation span


def test_min_variance_strategy_returns_long_only_weights():
    w = make_min_variance_strategy()(RET.iloc[:100])
    assert np.all(w >= -1e-9) and w.sum() == pytest.approx(1.0, abs=1e-8)


def test_tangency_strategy_falls_back_when_all_means_negative():
    df = pd.DataFrame(
        np.random.default_rng(0).normal(-0.01, 0.005, size=(80, 3))
    )
    w = make_tangency_strategy(shrink_mean=False)(df)
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0, abs=1e-8)


# ----------------------------------------------------------------- validation

def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="window"):
        run_backtest(RET, strategy_equal_weight, window=0)
    with pytest.raises(ValueError, match="rebalance_every"):
        run_backtest(RET, strategy_equal_weight, window=10, rebalance_every=0)
    with pytest.raises(ValueError, match="cost_bps"):
        run_backtest(RET, strategy_equal_weight, window=10, cost_bps=-1.0)
    with pytest.raises(ValueError, match="nothing to backtest"):
        run_backtest(RET.iloc[:10], strategy_equal_weight, window=10)


def test_strategy_returning_wrong_size_raises():
    with pytest.raises(ValueError, match="weights"):
        run_backtest(RET, lambda est: np.ones(2), window=10)


def test_static_strategy_wrong_size_raises():
    with pytest.raises(ValueError, match="static weights"):
        run_backtest(RET, make_static_strategy(np.ones(6) / 6), window=10)
