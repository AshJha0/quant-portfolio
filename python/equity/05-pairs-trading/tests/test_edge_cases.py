"""Edge cases from the documentation contract: zero-variance legs, gaps,
all-cash periods, zero-trade backtests, tiny samples."""

import numpy as np
import pandas as pd
import pytest

from eq_pairs.backtest import ZERO_COSTS, backtest_pair, backtest_portfolio
from eq_pairs.cointegration import adf_test, engle_granger, hedge_ratio
from eq_pairs.data import business_index
from eq_pairs.metrics import summary
from eq_pairs.signals import generate_signals
from eq_pairs.spread import fit_ou_ols
from eq_pairs.universe import correlation_screen, pair_correlations


def _series(vals) -> pd.Series:
    return pd.Series(np.asarray(vals, dtype=float), index=business_index(len(vals)))


class TestZeroVarianceLeg:
    def test_correlation_screen_survives(self):
        idx = business_index(60)
        prices = pd.DataFrame(
            {"A": np.linspace(100, 120, 60), "FLAT": np.full(60, 10.0)}, index=idx
        )
        corr = pair_correlations(prices, [("A", "FLAT")])
        assert np.isnan(corr.iloc[0, 0])
        assert len(correlation_screen(prices, [("A", "FLAT")], min_corr=0.0)) == 0

    def test_hedge_ratio_raises_informatively(self):
        with pytest.raises(ValueError, match="zero variance"):
            hedge_ratio(np.arange(100.0), np.full(100, 5.0))

    def test_adf_raises_informatively(self):
        with pytest.raises(ValueError, match="zero variance"):
            adf_test(np.full(100, 5.0))

    def test_engle_granger_zero_variance_x_raises(self):
        with pytest.raises(ValueError, match="zero variance"):
            engle_granger(np.arange(100.0) + 100.0, np.full(100, 5.0))

    def test_ou_fit_on_constant_spread_raises(self):
        with pytest.raises(ValueError, match="zero variance"):
            fit_ou_ols(np.full(100, 2.5))


class TestMissingData:
    def test_gap_policy_documented_behaviour(self):
        from eq_pairs.backtest import align_pair

        idx = business_index(30)
        y = pd.Series(np.linspace(100.0, 130.0, 30), index=idx)
        x = pd.Series(np.linspace(50.0, 60.0, 30), index=idx)
        y.iloc[10:13] = np.nan  # 3-day halt: within ffill limit
        ya, xa = align_pair(y, x, policy="ffill", limit=5)
        assert len(ya) == 30 and ya.notna().all()
        y.iloc[10:20] = np.nan  # 10-day halt: must NOT be silently filled
        with pytest.raises(ValueError, match="gap too long"):
            align_pair(y, x, policy="ffill", limit=5)
        ya, xa = align_pair(y, x, policy="drop")
        assert len(ya) == 20

    def test_leading_trailing_nans_trimmed(self):
        from eq_pairs.backtest import align_pair

        idx = business_index(20)
        y = pd.Series(np.linspace(100.0, 120.0, 20), index=idx)
        x = pd.Series(np.linspace(50.0, 55.0, 20), index=idx)
        y.iloc[:4] = np.nan  # y lists later
        x.iloc[-3:] = np.nan  # x delists earlier
        ya, xa = align_pair(y, x)
        assert len(ya) == 13
        assert ya.notna().all() and xa.notna().all()


class TestDegenerateBacktests:
    def test_zero_trades_full_pipeline(self):
        y = _series(np.linspace(100, 105, 60))
        x = _series(np.linspace(50, 52, 60))
        target = pd.Series(np.zeros(60), index=y.index)
        res = backtest_pair(y, x, target, beta=2.0)
        stats = summary(res.daily, res.trades, res.ledger, capital=1e6)
        assert stats["n_trades"] == 0
        assert stats["total_net_pnl"] == 0.0
        assert stats["turnover"] == 0.0
        assert np.isnan(stats["hit_rate"])
        assert np.isnan(stats["avg_holding_days"])

    def test_all_cash_period_metrics_are_nan_not_crash(self):
        y = _series(np.linspace(100, 105, 30))
        x = _series(np.full(30, 50.0))
        res = backtest_pair(y, x, pd.Series(np.zeros(30), index=y.index), beta=1.0)
        stats = summary(res.daily, res.trades, res.ledger, capital=1e6)
        assert np.isnan(stats["sharpe"])
        assert np.isnan(stats["sortino"])
        assert stats["max_drawdown"] == 0.0

    def test_single_pair_portfolio_end_to_end(self):
        idx = business_index(120)
        s = np.sin(np.arange(120) / 4.0) * 3.0
        panel = pd.DataFrame({"A": 100.0 + s, "B": np.full(120, 100.0)}, index=idx)
        target = pd.Series(np.select([s > 2, s < -2], [-1, 1], 0), index=idx)
        port = backtest_portfolio(
            panel, {("A", "B"): target}, {("A", "B"): 1.0}, costs=ZERO_COSTS
        )
        assert len(port.pairs) == 1
        att = port.attribution()
        assert len(att) == 1
        stats = summary(
            port.daily.assign(),
            port.pairs[0].trades,
            port.pairs[0].ledger,
            capital=1e6,
        )
        assert stats["n_trades"] >= 1

    def test_backtest_too_short_raises(self):
        y = _series([100.0])
        with pytest.raises(ValueError, match="at least 2 bars"):
            backtest_pair(y, y, pd.Series([0.0], index=y.index), beta=1.0)


class TestSignalEdges:
    def test_all_nan_z_stays_flat(self):
        z = pd.Series(np.full(20, np.nan), index=business_index(20))
        out = generate_signals(z)
        assert (out["position"] == 0).all()

    def test_z_exactly_at_thresholds(self):
        z = pd.Series([0.0, 2.0, 0.0], index=business_index(3))
        out = generate_signals(z)  # >= entry is an entry; >= -exit is an exit
        assert list(out["position"]) == [0, -1, 0]

    def test_summary_capital_validation(self):
        y = _series(np.linspace(100, 105, 30))
        res = backtest_pair(
            y, _series(np.full(30, 50.0)), pd.Series(np.zeros(30), index=y.index),
            beta=1.0,
        )
        with pytest.raises(ValueError, match="capital"):
            summary(res.daily, res.trades, res.ledger, capital=0.0)

    def test_short_series_everywhere(self):
        with pytest.raises(ValueError):
            fit_ou_ols(np.arange(5.0))
        with pytest.raises(ValueError):
            adf_test(np.array([1.0, 2.0, 1.5]), lags=2)
