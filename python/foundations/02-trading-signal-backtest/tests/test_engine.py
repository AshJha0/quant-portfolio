"""run_backtest: no-look-ahead structural proof, exact cost accounting,
and performance_stats against hand-computable equity curves."""

import numpy as np
import pandas as pd
import pytest

from eq_signal_backtest.engine import performance_stats, run_backtest


# ---------------------------------------------------------------------------
# No-look-ahead
# ---------------------------------------------------------------------------


class TestNoLookAhead:
    def test_position_equals_prior_day_signal_structurally(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        prices = pd.Series([100, 101, 99, 102, 105, 103], index=dates, dtype=float)
        signal = pd.Series([1, 0, 1, 1, 0, 1], index=dates, dtype=float)
        res = run_backtest(prices, signal, cost_bps=0.0)
        for t in range(1, len(dates)):
            assert res.position.iloc[t] == signal.iloc[t - 1]

    def test_first_day_position_is_zero_not_nan(self):
        """shift(1) introduces a leading NaN on day 0; it must be filled to
        0.0 (flat), not silently left as NaN or defaulted to "long"."""
        dates = pd.bdate_range("2024-01-01", periods=4)
        prices = pd.Series([100, 101, 102, 103], index=dates, dtype=float)
        for first_signal_value in (0.0, 1.0):
            signal = pd.Series(
                [first_signal_value, 1.0, 0.0, 1.0], index=dates, dtype=float
            )
            res = run_backtest(prices, signal, cost_bps=0.0)
            assert not pd.isna(res.position.iloc[0])
            assert res.position.iloc[0] == 0.0

    def test_cheat_profits_from_a_jump_honest_engine_does_not(self):
        """Detector test: an oracle signal fires exactly on the day of a
        30% overnight jump. Same-day ("cheat") execution captures the
        jump; the engine's t-1 lag means the position that day was decided
        before the jump was known, so it must NOT capture it."""
        dates = pd.bdate_range("2024-01-01", periods=5)
        prices = pd.Series([100.0, 100.0, 100.0, 130.0, 130.0], index=dates)
        signal = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0], index=dates)

        honest = run_backtest(prices, signal, cost_bps=0.0)

        rets = prices.pct_change().fillna(0.0)
        cheat_equity = (1 + signal * rets).cumprod()  # same-day execution bug

        assert cheat_equity.iloc[-1] > 1.25
        assert honest.equity.iloc[-1] == pytest.approx(1.0, abs=1e-9)

    def test_position_never_uses_same_day_signal_value(self):
        """A signal that is 1 only on the final day can never produce a
        nonzero position anywhere in the series (there is no day t+1 for
        the engine to trade it on)."""
        dates = pd.bdate_range("2024-01-01", periods=5)
        prices = pd.Series([100, 105, 95, 110, 120], index=dates, dtype=float)
        signal = pd.Series([0, 0, 0, 0, 1], index=dates, dtype=float)
        res = run_backtest(prices, signal, cost_bps=0.0)
        assert (res.position == 0.0).all()
        assert res.n_trades == 0


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------


class TestTransactionCosts:
    def test_position_change_incurs_exact_cost_bps_drag(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        prices = pd.Series([100.0, 105.0, 110.0], index=dates)
        signal = pd.Series([0.0, 1.0, 1.0], index=dates)  # trades in on day 2
        cost_bps = 5.0

        costed = run_backtest(prices, signal, cost_bps=cost_bps)
        free = run_backtest(prices, signal, cost_bps=0.0)

        day2_ret_costed = costed.equity.iloc[2] / costed.equity.iloc[1] - 1
        day2_ret_free = free.equity.iloc[2] / free.equity.iloc[1] - 1
        assert day2_ret_free - day2_ret_costed == pytest.approx(
            cost_bps / 10_000, abs=1e-12
        )
        assert costed.n_trades == 1

    def test_no_position_change_no_cost(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        prices = pd.Series([100.0, 105.0, 95.0], index=dates)
        signal = pd.Series([0.0, 0.0, 0.0], index=dates)
        costed = run_backtest(prices, signal, cost_bps=25.0)
        free = run_backtest(prices, signal, cost_bps=0.0)
        pd.testing.assert_series_equal(costed.equity, free.equity)
        assert costed.n_trades == 0

    def test_round_trip_pays_cost_on_each_leg(self):
        dates = pd.bdate_range("2024-01-01", periods=4)
        prices = pd.Series([100.0, 101.0, 102.0, 103.0], index=dates)
        signal = pd.Series([1.0, 0.0, 1.0, 0.0], index=dates)
        # position = signal.shift(1).fillna(0) = [0, 1, 0, 1]: three changes
        # (day0 flat->N/A counts as no trade; day1 0->1, day2 1->0, day3 0->1)
        res = run_backtest(prices, signal, cost_bps=10.0)
        assert res.n_trades == 3

    def test_cost_free_always_long_matches_buy_and_hold_exactly(self):
        """cost_bps=0 reproduces buy-and-hold-while-signal-on exactly: with
        the signal always on, the only difference from buy&hold is the
        first day's flat position, and pct_change().fillna(0.0) already
        makes both day-0 returns zero, so the two curves are identical."""
        dates = pd.bdate_range("2024-01-01", periods=10)
        prices = pd.Series(
            [100, 102, 101, 105, 103, 108, 107, 110, 109, 112],
            index=dates,
            dtype=float,
        )
        signal = pd.Series(1.0, index=dates)
        res = run_backtest(prices, signal, cost_bps=0.0)
        pd.testing.assert_series_equal(res.equity, res.benchmark, check_names=False)


# ---------------------------------------------------------------------------
# performance_stats
# ---------------------------------------------------------------------------


class TestPerformanceStats:
    def test_known_cagr(self):
        target_cagr = 0.20
        r = (1 + target_cagr) ** (1 / 252) - 1
        returns = pd.Series([r] * 252, index=pd.bdate_range("2020-01-01", periods=252))
        equity = (1 + returns).cumprod()
        stats = performance_stats(returns, equity)
        assert stats["cagr"] == pytest.approx(target_cagr, rel=1e-9)

    def test_known_max_drawdown(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        equity = pd.Series([1.0, 1.2, 0.9, 1.1], index=idx)
        returns = equity.pct_change().fillna(0.0)
        stats = performance_stats(returns, equity)
        # cummax = [1, 1.2, 1.2, 1.2] -> dd = [0, 0, -0.25, -0.0833...] -> min -0.25
        assert stats["max_drawdown"] == pytest.approx(-0.25, abs=1e-9)

    def test_known_sharpe_from_constructed_returns(self):
        n = 250
        mu, d = 0.0005, 0.001
        values = np.array([mu + d, mu - d] * (n // 2))
        returns = pd.Series(values, index=pd.bdate_range("2020-01-01", periods=n))
        equity = (1 + returns).cumprod()
        stats = performance_stats(returns, equity)
        expected_std = d * np.sqrt(n / (n - 1))  # ddof=1, equal split around mu
        expected_sharpe = mu / expected_std * np.sqrt(252)
        assert stats["sharpe"] == pytest.approx(expected_sharpe, rel=1e-9)

    def test_zero_volatility_sharpe_is_nan_not_zero_or_inf(self):
        # exactly 0.0 every day (e.g. a flat/no-trade book) has exactly
        # zero variance in floating point, unlike a repeated nonzero
        # decimal whose mean/deviations can pick up float noise.
        idx = pd.bdate_range("2020-01-01", periods=20)
        returns = pd.Series([0.0] * 20, index=idx)
        equity = (1 + returns).cumprod()
        stats = performance_stats(returns, equity)
        assert np.isnan(stats["sharpe"])

    def test_single_observation_sharpe_is_nan(self):
        idx = pd.bdate_range("2020-01-01", periods=1)
        returns = pd.Series([0.01], index=idx)
        equity = (1 + returns).cumprod()
        stats = performance_stats(returns, equity)
        assert np.isnan(stats["sharpe"])
        assert np.isfinite(stats["cagr"])
        assert stats["max_drawdown"] == pytest.approx(0.0)

    def test_non_decreasing_equity_has_zero_drawdown(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        returns = pd.Series([0.01, 0.02, 0.0, 0.03, 0.01], index=idx)
        equity = (1 + returns).cumprod()
        stats = performance_stats(returns, equity)
        assert stats["max_drawdown"] == pytest.approx(0.0)

    def test_exposure_is_share_of_nonzero_return_days(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        returns = pd.Series([0.0, 0.01, 0.0, -0.01], index=idx)
        equity = (1 + returns).cumprod()
        stats = performance_stats(returns, equity)
        assert stats["exposure"] == pytest.approx(0.5)
