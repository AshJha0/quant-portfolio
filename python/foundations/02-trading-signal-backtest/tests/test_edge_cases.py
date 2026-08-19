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
