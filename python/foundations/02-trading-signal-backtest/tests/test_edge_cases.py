"""Edge cases required by the portfolio testing contract: all-flat signal,
zero volatility, single-day series, cost_bps=0, and live-data import guard."""

import numpy as np
import pandas as pd
import pytest

from eq_signal_backtest.data import live as live_module
from eq_signal_backtest.engine import run_backtest
from eq_signal_backtest.signals import ma_crossover_signal


def _prices(n=40, seed=5):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, n)
    prices = 100 * np.exp(np.cumsum(rets))
    return pd.Series(prices, index=pd.bdate_range("2023-01-01", periods=n))


class TestAllFlatSignal:
    def test_zero_trades_and_constant_equity(self):
        prices = _prices()
        signal = pd.Series(0.0, index=prices.index)
        res = run_backtest(prices, signal, cost_bps=5.0)
        assert res.n_trades == 0
        assert (res.equity == 1.0).all()

    def test_zero_variance_gives_nan_sharpe(self):
        prices = _prices()
        signal = pd.Series(0.0, index=prices.index)
        res = run_backtest(prices, signal, cost_bps=0.0)
        assert np.isnan(res.stats["sharpe"])
        assert res.stats["max_drawdown"] == pytest.approx(0.0)


class TestSingleDaySeries:
    def test_backtest_does_not_crash_and_stays_flat(self):
        idx = pd.bdate_range("2024-01-01", periods=1)
        prices = pd.Series([100.0], index=idx)
        signal = pd.Series([1.0], index=idx)
        res = run_backtest(prices, signal, cost_bps=5.0)
        assert res.position.iloc[0] == 0.0  # no prior day to lag from
        assert res.n_trades == 0
        assert res.equity.iloc[0] == pytest.approx(1.0)
        assert np.isnan(res.stats["sharpe"])  # std needs >= 2 observations

    def test_ma_crossover_signal_on_two_points_no_crash(self):
        idx = pd.bdate_range("2024-01-01", periods=2)
        prices = pd.Series([100.0, 101.0], index=idx)
        sig = ma_crossover_signal(prices, fast=1, slow=2)
        assert len(sig) == 2


class TestZeroCostReproducesBuyAndHoldWhileOn:
    def test_exact_match_when_always_long(self):
        prices = _prices(n=60)
        signal = pd.Series(1.0, index=prices.index)
        res = run_backtest(prices, signal, cost_bps=0.0)
        pd.testing.assert_series_equal(res.equity, res.benchmark, check_names=False)

    def test_diverges_once_costs_are_nonzero(self):
        prices = _prices(n=60)
        # a fast/slow crossover that actually trades a few times
        signal = ma_crossover_signal(prices, fast=3, slow=10)
        costed = run_backtest(prices, signal, cost_bps=25.0)
        free = run_backtest(prices, signal, cost_bps=0.0)
        if costed.n_trades > 0:
            assert costed.equity.iloc[-1] != pytest.approx(free.equity.iloc[-1])


def _range_bound_prices(n=750, seed=99, kappa=0.05, sigma=0.012):
    """A mean-reverting (Ornstein-Uhlenbeck-like) synthetic price path:
    the regime this strategy class is documented to fail in
    (docs/VALIDATION.md 'Failure mode 1')."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = x[t - 1] + kappa * (0.0 - x[t - 1]) + sigma * rng.standard_normal()
    return pd.Series(
        100 * np.exp(x), index=pd.bdate_range("2020-01-01", periods=n)
    )


class TestChoppyRegimeFailureMode:
    def test_range_bound_regime_produces_whipsaw_underperformance(self):
        """In a range-bound (mean-reverting, non-trending) regime, every
        crossover is a whipsaw: the strategy trades repeatedly, pays
        costs, and captures no sustained move, so it underperforms
        buy & hold -- reproducibly, on a fixed seed. This is the
        regime-dependence failure mode documented in
        docs/VALIDATION.md."""
        prices = _range_bound_prices()
        sig = ma_crossover_signal(prices, fast=20, slow=100)
        res = run_backtest(prices, sig, cost_bps=5.0)
        assert res.n_trades >= 5  # whipsaws: the signal keeps flipping
        assert res.stats["cagr"] < res.stats["benchmark"]["cagr"]
        assert res.stats["sharpe"] < res.stats["benchmark"]["sharpe"]

    def test_costs_make_the_whipsaw_regime_strictly_worse(self):
        prices = _range_bound_prices()
        sig = ma_crossover_signal(prices, fast=20, slow=100)
        costed = run_backtest(prices, sig, cost_bps=5.0)
        free = run_backtest(prices, sig, cost_bps=0.0)
        if costed.n_trades > 0:
            assert costed.stats["cagr"] <= free.stats["cagr"]


class TestLiveDataImportGuard:
    def test_load_prices_without_yfinance_raises_importerror_or_works(self):
        """The live loader must never be exercised over the network in
        tests. If yfinance happens to be installed in this environment we
        only check the function exists and is guarded; if it is not
        installed, calling it must raise a clear ImportError rather than
        an AttributeError/NameError."""
        if live_module._HAS_YF:
            pytest.skip("yfinance is installed in this environment; guard not exercised")
        with pytest.raises(ImportError, match="yfinance"):
            live_module.load_prices("SPY")

    def test_module_import_never_touches_network(self):
        # importing the module must succeed regardless of network access;
        # the guard means yfinance is only imported, never called, at
        # import time.
        import importlib

        importlib.reload(live_module)
        assert hasattr(live_module, "load_prices")


class TestNonFinitePrices:
    """A missing or infinite close is a data-quality event, not a market
    event. Both used to pass through silently and corrupt the result."""

    def test_nan_price_used_to_be_recorded_as_a_flat_day_now_raises(self):
        """``pct_change().fillna(0.0)`` turns the two days around a NaN
        close into "no move at all", swallowing the entire move across the
        gap and inventing a flat day that never happened."""
        idx = pd.bdate_range("2024-01-01", periods=8)
        prices = pd.Series([100.0, 101.0, np.nan, 103.0, 104.0, 103.0, 105.0, 106.0], index=idx)
        signal = pd.Series(1.0, index=idx)
        with pytest.raises(ValueError, match="non-finite"):
            run_backtest(prices, signal, cost_bps=5.0)

    def test_nan_price_is_rejected_by_the_signal_too(self):
        """Caught one layer earlier as well: a NaN close blanks `slow` days
        of moving average, producing a flat signal that is impossible to
        distinguish from a genuine bearish one."""
        idx = pd.bdate_range("2024-01-01", periods=8)
        prices = pd.Series([100.0, 101.0, np.nan, 103.0, 104.0, 103.0, 105.0, 106.0], index=idx)
        with pytest.raises(ValueError, match="non-finite"):
            ma_crossover_signal(prices, fast=2, slow=4)

    def test_infinite_price_raises(self):
        idx = pd.bdate_range("2024-01-01", periods=5)
        prices = pd.Series([100.0, 101.0, np.inf, 103.0, 104.0], index=idx)
        with pytest.raises(ValueError, match="non-finite"):
            run_backtest(prices, pd.Series(1.0, index=idx), cost_bps=5.0)

    def test_exactly_zero_price_raises_instead_of_poisoning_the_curve(self):
        """A zero close makes the *next* day's pct_change infinite, and
        from that point the equity curve is NaN for the rest of the
        sample -- previously with nothing but a RuntimeWarning."""
        idx = pd.bdate_range("2024-01-01", periods=6)
        prices = pd.Series([100.0, 101.0, 0.0, 103.0, 104.0, 105.0], index=idx)
        with pytest.raises(ValueError, match="strictly positive"):
            run_backtest(prices, pd.Series(1.0, index=idx), cost_bps=5.0)

    def test_negative_price_raises(self):
        idx = pd.bdate_range("2024-01-01", periods=4)
        prices = pd.Series([100.0, 101.0, -5.0, 103.0], index=idx)
        with pytest.raises(ValueError, match="strictly positive"):
            run_backtest(prices, pd.Series(1.0, index=idx), cost_bps=5.0)

    def test_empty_prices_raises(self):
        empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        with pytest.raises(ValueError, match="empty"):
            run_backtest(empty, empty, cost_bps=5.0)


class TestSignalValidation:
    def test_misaligned_signal_index_raises(self):
        """Pandas would outer-join the two indices into NaN positions and
        report a plausible-looking (wrong) equity curve."""
        prices = _prices(n=20)
        signal = pd.Series(1.0, index=prices.index[:-3])
        with pytest.raises(ValueError, match="share the exact index"):
            run_backtest(prices, signal, cost_bps=5.0)

    def test_signal_with_nan_values_raises(self):
        prices = _prices(n=20)
        signal = pd.Series(1.0, index=prices.index)
        signal.iloc[5] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            run_backtest(prices, signal, cost_bps=5.0)

    @pytest.mark.parametrize("bad_value", [0.5, -1.0, 2.0])
    def test_non_binary_signal_raises(self, bad_value):
        """The engine is long/flat only: a -1 (short) or 0.5 (half size)
        would be silently multiplied through as if it were supported,
        producing returns from mechanics the cost model does not describe."""
        prices = _prices(n=20)
        signal = pd.Series(1.0, index=prices.index)
        signal.iloc[7] = bad_value
        with pytest.raises(ValueError, match="0.0 or 1.0"):
            run_backtest(prices, signal, cost_bps=5.0)


class TestExtremeTransactionCosts:
    """cost_bps well beyond anything realistic. The engine must degrade
    monotonically to a wipe-out, never into a sign-flipping equity curve."""

    @staticmethod
    def _alternating(n=12):
        idx = pd.bdate_range("2024-01-01", periods=n)
        prices = pd.Series(np.linspace(100.0, 120.0, n), index=idx)
        signal = pd.Series([float(i % 2) for i in range(n)], index=idx)
        return prices, signal

    def test_round_trip_cost_above_one_hundred_percent_wipes_out_to_zero(self):
        """A one-way cost of 200% (400% round trip) used to drive the daily
        return below -100%, sending equity NEGATIVE and then flipping its
        sign on every subsequent day -- which even produced a finite,
        recovered-looking CAGR. It now floors at a total loss."""
        prices, signal = self._alternating()
        res = run_backtest(prices, signal, cost_bps=20_000.0)
        assert (res.equity >= 0).all()
        assert res.equity.iloc[-1] == pytest.approx(0.0, abs=1e-15)
        assert res.stats["cagr"] == pytest.approx(-1.0, abs=1e-12)

    def test_zero_is_an_absorbing_state(self):
        """Once wiped out, the curve stays wiped out: no later gain can
        resurrect a zero equity, which is what happens in reality."""
        prices, signal = self._alternating(n=20)
        res = run_backtest(prices, signal, cost_bps=20_000.0)
        first_zero = int(np.argmax(res.equity.to_numpy() <= 0))
        assert (res.equity.iloc[first_zero:] == 0.0).all()

    def test_costs_are_monotonically_worse(self):
        """Sanity property across the whole plausible-to-absurd range:
        raising the cost can never improve the final equity."""
        prices, signal = self._alternating(n=20)
        finals = [
            run_backtest(prices, signal, cost_bps=c).equity.iloc[-1]
            for c in (0.0, 5.0, 50.0, 500.0, 5_000.0, 20_000.0)
        ]
        assert all(a >= b - 1e-15 for a, b in zip(finals, finals[1:]))

    def test_hundred_percent_round_trip_cost_is_survivable_but_brutal(self):
        """5,000 bps one-way = 50% per leg, 100% round trip: each trade
        halves the book, so equity decays geometrically but stays strictly
        positive. The wipe-out floor must not bind here."""
        prices, signal = self._alternating(n=12)
        res = run_backtest(prices, signal, cost_bps=5_000.0)
        assert (res.equity > 0).all()
        assert res.equity.iloc[-1] < 0.01

    def test_negative_cost_is_rejected(self):
        prices, signal = self._alternating()
        with pytest.raises(ValueError, match="cost_bps"):
            run_backtest(prices, signal, cost_bps=-5.0)

    def test_non_finite_cost_is_rejected(self):
        prices, signal = self._alternating()
        with pytest.raises(ValueError, match="cost_bps"):
            run_backtest(prices, signal, cost_bps=np.nan)


class TestInvalidSignalWindows:
    @pytest.mark.parametrize("fast,slow", [(0, 10), (-5, 10), (0, 1)])
    def test_windows_below_one_are_rejected(self, fast, slow):
        prices = _prices(n=30)
        with pytest.raises(ValueError, match="must be >= 1"):
            ma_crossover_signal(prices, fast, slow)

    def test_non_integer_window_is_rejected(self):
        prices = _prices(n=30)
        with pytest.raises(ValueError, match="must be an int"):
            ma_crossover_signal(prices, 5.5, 20)

    def test_windows_longer_than_the_sample_give_an_all_flat_signal(self):
        """Not an error: both moving averages are NaN throughout, and
        ``NaN > NaN`` is False, so the signal is flat everywhere. A
        strategy that never warms up simply never trades."""
        prices = _prices(n=30)
        sig = ma_crossover_signal(prices, fast=50, slow=200)
        assert (sig == 0.0).all()
        res = run_backtest(prices, sig, cost_bps=5.0)
        assert res.n_trades == 0
        assert (res.equity == 1.0).all()
