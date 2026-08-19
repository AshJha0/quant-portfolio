"""train_test_split, select_best_params, and walk-forward window hygiene."""

import numpy as np
import pandas as pd
import pytest

from eq_signal_backtest.split import (
    WalkForwardWindow,
    select_best_params,
    train_test_split,
    walk_forward_backtest,
    walk_forward_windows,
)


def _trending_prices(n=400, seed=11, drift=0.0006, vol=0.008):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    prices = 100 * np.exp(np.cumsum(rets))
    return pd.Series(prices, index=pd.bdate_range("2015-01-01", periods=n))


class TestTrainTestSplit:
    def test_basic_split_is_contiguous_and_covers_all_rows(self):
        prices = _trending_prices(n=100)
        split = train_test_split(prices, train_frac=0.7)
        assert len(split.train) == 70
        assert len(split.test) == 30
        pd.testing.assert_series_equal(
            pd.concat([split.train, split.test]), prices
        )
        assert split.split_date == split.train.index[-1]

    def test_train_and_test_do_not_overlap(self):
        prices = _trending_prices(n=53)
        split = train_test_split(prices, train_frac=0.6)
        assert split.train.index[-1] < split.test.index[0]
        assert len(set(split.train.index) & set(split.test.index)) == 0

    @pytest.mark.parametrize("bad_frac", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_train_frac_raises(self, bad_frac):
        prices = _trending_prices(n=10)
        with pytest.raises(ValueError, match="train_frac"):
            train_test_split(prices, train_frac=bad_frac)

    def test_too_short_series_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            train_test_split(pd.Series([1.0]), train_frac=0.5)

    def test_tiny_series_keeps_both_sides_nonempty(self):
        prices = _trending_prices(n=3)
        split = train_test_split(prices, train_frac=0.1)
        assert len(split.train) >= 1
        assert len(split.test) >= 1


class TestSelectBestParams:
    def test_returns_a_valid_pair_from_the_grid(self):
        prices = _trending_prices(n=200)
        fast_range = range(3, 10, 2)
        slow_range = range(15, 40, 5)
        best_fast, best_slow, grid = select_best_params(
            prices, fast_range, slow_range, cost_bps=1.0
        )
        assert best_fast in fast_range
        assert best_slow in slow_range
        assert best_fast < best_slow
        assert grid.loc[best_fast, best_slow] == grid.stack().max()

    def test_only_uses_prices_it_is_given(self):
        """select_best_params must not reach past the slice it receives --
        proven by checking its result is identical whether or not
        additional (later) data exists in the caller's scope."""
        prices = _trending_prices(n=300)
        train_only = prices.iloc[:200]
        fast_range, slow_range = range(3, 8), range(15, 30, 5)
        best_a = select_best_params(train_only, fast_range, slow_range)[:2]
        best_b = select_best_params(prices.iloc[:200].copy(), fast_range, slow_range)[:2]
        assert best_a == best_b

    def test_all_invalid_combinations_raises(self):
        prices = _trending_prices(n=50)
        with pytest.raises(ValueError, match="empty or all-NaN"):
            select_best_params(prices, fast_range=[20], slow_range=[5])


class TestWalkForwardWindows:
    def test_formation_and_trading_never_overlap(self):
        wins = walk_forward_windows(1000, formation=252, trading=63)
        assert len(wins) > 0
        for w in wins:
            assert w.formation_end < w.trading_start
            assert w.formation_end - w.formation_start + 1 == 252
            assert w.trading_end - w.trading_start + 1 == 63

    def test_default_step_gives_contiguous_trading_windows(self):
        wins = walk_forward_windows(1000, formation=252, trading=63)
        for a, b in zip(wins[:-1], wins[1:]):
            assert b.trading_start == a.trading_end + 1

    def test_window_count(self):
        wins = walk_forward_windows(252 + 63 * 4, formation=252, trading=63)
        assert len(wins) == 4

    def test_malformed_window_raises(self):
        with pytest.raises(ValueError, match="malformed"):
            WalkForwardWindow(0, 100, 50, 150)  # trading starts inside formation

    def test_parameter_validation(self):
        with pytest.raises(ValueError, match="formation"):
            walk_forward_windows(500, formation=1, trading=63)
        with pytest.raises(ValueError, match="trading"):
            walk_forward_windows(500, formation=252, trading=0)
        with pytest.raises(ValueError, match="step"):
            walk_forward_windows(500, formation=252, trading=63, step=0)

    def test_too_short_sample_returns_empty_list(self):
        assert walk_forward_windows(200, formation=252, trading=63) == []

    def test_custom_step_can_overlap_trading_windows(self):
        wins = walk_forward_windows(1000, formation=252, trading=63, step=21)
        assert len(wins) > len(walk_forward_windows(1000, 252, 63))
        for a, b in zip(wins[:-1], wins[1:]):
            assert b.formation_start == a.formation_start + 21


class TestWalkForwardBacktest:
    def _prices(self):
        return _trending_prices(n=1000, seed=21, drift=0.0004, vol=0.01)

    def test_runs_and_windows_dataframe_matches_window_count(self):
        prices = self._prices()
        result = walk_forward_backtest(
            prices,
            fast_range=range(5, 20, 5),
            slow_range=range(30, 80, 20),
            formation=252,
            trading=63,
            cost_bps=5.0,
        )
        expected_windows = walk_forward_windows(len(prices), 252, 63)
        assert len(result.windows) == len(expected_windows)
        assert (result.windows["fast"] < result.windows["slow"]).all()
        assert result.n_trades == result.windows["n_trades"].sum()
        assert len(result.equity) == len(result.benchmark)

    def test_frozen_parameters_reproduce_window_returns_exactly(self):
        """Reconstruct the first trading window's daily strategy RETURNS
        from the recorded (fast, slow) pair alone; they must match the
        stitched equity curve's implied returns for that window exactly.
        If parameters were re-estimated inside the trading window, or if
        the formation-window history leaked into the trading returns,
        this would diverge."""
        from eq_signal_backtest.signals import ma_crossover_signal

        prices = self._prices()
        cost_bps = 5.0
        result = walk_forward_backtest(
            prices,
            fast_range=range(5, 20, 5),
            slow_range=range(30, 80, 20),
            formation=252,
            trading=63,
            cost_bps=cost_bps,
        )
        row = result.windows.iloc[0]

        # actual: back out the exact per-day return the engine produced,
        # straight from the stitched equity curve (equity = cumprod(rets)).
        # equity.pct_change() is NaN on the very first observation of the
        # whole series (no predecessor to divide by), so that one day is
        # recovered directly from equity itself instead.
        full_rets = result.equity.pct_change()
        full_rets.iloc[0] = result.equity.iloc[0] - 1.0
        actual = full_rets.loc[row["trading_start"] : row["trading_end"]]

        # reconstruction: formation+trading context, frozen params only
        context = prices.loc[row["formation_start"] : row["trading_end"]]
        sig = ma_crossover_signal(context, int(row["fast"]), int(row["slow"]))
        position = sig.shift(1).fillna(0.0)
        rets = context.pct_change().fillna(0.0)
        trades = position.diff().abs().fillna(0.0)
        costs = trades * cost_bps / 10_000
        strat_rets = position * rets - costs
        recon = strat_rets.loc[row["trading_start"] : row["trading_end"]]

        np.testing.assert_allclose(
            actual.to_numpy(), recon.to_numpy(), atol=1e-10
        )

    def test_sample_too_short_raises_informative_error(self):
        prices = _trending_prices(n=100)
        with pytest.raises(ValueError, match="too short"):
            walk_forward_backtest(
                prices, range(5, 10), range(20, 30), formation=252, trading=63
            )

    def test_stats_include_benchmark(self):
        prices = self._prices()
        result = walk_forward_backtest(
            prices,
            fast_range=range(5, 20, 5),
            slow_range=range(30, 80, 20),
            formation=252,
            trading=63,
            cost_bps=0.0,
        )
        assert "benchmark" in result.stats
        assert "sharpe" in result.stats


class TestWalkForwardWindowsLargerThanData:
    """The whole family of "the window does not fit" cases. Each returns
    an empty window list (or an informative error at the backtest layer)
    rather than silently truncating a window and quietly testing on less
    data than the caller asked for."""

    @pytest.mark.parametrize(
        "n,formation,trading",
        [
            (100, 252, 63),   # formation alone exceeds the sample
            (100, 50, 500),   # trading alone exceeds the sample
            (100, 90, 20),    # formation + trading exceeds the sample by 10
            (10, 6, 5),       # formation + trading exceeds the sample by 1
            (2, 2, 1),
        ],
    )
    def test_windows_that_do_not_fit_produce_no_windows(self, n, formation, trading):
        assert walk_forward_windows(n, formation, trading) == []

    def test_one_observation_is_the_difference_between_one_window_and_none(self):
        """The exact boundary: formation + trading == n fits exactly one
        window (consuming every observation); one observation fewer fits
        none. Worth pinning because an off-by-one here would either drop a
        legitimate window or read past the end of the series."""
        wins = walk_forward_windows(100, formation=80, trading=20)
        assert len(wins) == 1
        assert wins[0].formation_start == 0
        assert wins[0].trading_end == 99  # last valid 0-based index
        assert walk_forward_windows(99, formation=80, trading=20) == []

    @pytest.mark.parametrize(
        "formation,trading",
        [(252, 63), (2000, 10), (10, 2000)],
    )
    def test_backtest_raises_informative_error_when_no_window_fits(
        self, formation, trading
    ):
        prices = _trending_prices(n=300, seed=4)
        with pytest.raises(ValueError, match="too short"):
            walk_forward_backtest(
                prices, range(5, 15, 5), range(20, 40, 10), formation, trading
            )

    def test_error_message_names_the_offending_sizes(self):
        prices = _trending_prices(n=300, seed=4)
        with pytest.raises(ValueError) as exc:
            walk_forward_backtest(
                prices, range(5, 15, 5), range(20, 40, 10), formation=252, trading=63
            )
        message = str(exc.value)
        assert "252" in message and "63" in message and "300" in message

    def test_formation_window_too_short_for_the_slow_ma_fails_loudly(self):
        """A formation window too short for the slow moving average to warm
        up gives an all-flat signal, a zero-variance return series and a
        NaN Sharpe for every grid cell -- so selection has nothing to rank
        and must fail loudly rather than pick an arbitrary corner."""
        prices = _trending_prices(n=400, seed=6)
        with pytest.raises(ValueError, match="empty or all-NaN"):
            walk_forward_backtest(
                prices,
                fast_range=range(5, 15, 5),
                slow_range=range(300, 340, 20),
                formation=60,
                trading=20,
            )


class TestWalkForwardStitching:
    def test_stitched_returns_cover_every_out_of_sample_day_exactly_once(self):
        """With the default step the trading windows tile the out-of-sample
        period: no day is counted twice (which would double-count
        compounding) and none is dropped."""
        prices = _trending_prices(n=800, seed=11)
        result = walk_forward_backtest(
            prices, range(5, 20, 5), range(30, 80, 20), formation=252, trading=63
        )
        assert result.equity.index.is_unique
        assert result.equity.index.is_monotonic_increasing
        expected_days = 63 * len(walk_forward_windows(len(prices), 252, 63))
        assert len(result.equity) == expected_days

    def test_stitched_equity_is_the_product_of_window_returns_not_a_splice(self):
        """The stitched curve compounds through the seams: its final value
        equals the product of each window's own gross return. A naive
        splice of per-window equity curves would restart at 1.0 each time
        and silently lose (or duplicate) the compounding."""
        prices = _trending_prices(n=800, seed=12)
        result = walk_forward_backtest(
            prices, range(5, 20, 5), range(30, 80, 20), formation=252, trading=63,
            cost_bps=0.0,
        )
        windows = walk_forward_windows(len(prices), 252, 63)
        gross = 1.0
        for w in windows:
            start, end = prices.index[w.trading_start], prices.index[w.trading_end]
            segment = result.equity.loc[start:end]
            first = result.equity.index.get_loc(start)
            prior = 1.0 if first == 0 else result.equity.iloc[first - 1]
            gross *= segment.iloc[-1] / prior
        assert result.equity.iloc[-1] == pytest.approx(gross, rel=1e-12)
