"""Tests for eq_risk_metrics.performance: drawdown, Sharpe, Sortino."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq_risk_metrics import max_drawdown, sharpe_ratio, sortino_ratio


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=n)


# ---------------------------------------------------------------------
# max_drawdown: hand-built price series with a known peak/trough
# ---------------------------------------------------------------------
def test_max_drawdown_hand_built_series() -> None:
    dates = _dates(7)
    prices = pd.Series([100.0, 120.0, 90.0, 95.0, 60.0, 70.0, 130.0], index=dates)
    result = max_drawdown(prices)
    assert result["max_drawdown"] == pytest.approx(-0.5, abs=1e-12)  # 60/120 - 1
    assert result["peak_date"] == dates[1]  # 120 at t=1
    assert result["trough_date"] == dates[4]  # 60 at t=4
    assert len(result["drawdown_series"]) == len(prices)
    assert result["drawdown_series"].iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown_monotonically_rising_series_is_zero() -> None:
    dates = _dates(5)
    prices = pd.Series([100.0, 101.0, 105.0, 110.0, 120.0], index=dates)
    result = max_drawdown(prices)
    assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    # Drawdown is 0.0 at every point on a monotonically rising series, so
    # idxmin() returns the *first* occurrence -- the very first date.
    assert result["peak_date"] == dates[0]
    assert result["trough_date"] == dates[0]


def test_max_drawdown_single_observation() -> None:
    dates = _dates(1)
    prices = pd.Series([100.0], index=dates)
    result = max_drawdown(prices)
    assert result["max_drawdown"] == 0.0
    assert result["peak_date"] == dates[0]
    assert result["trough_date"] == dates[0]


def test_max_drawdown_peak_is_local_not_global_before_trough() -> None:
    """The reported peak is the running max *at or before* the trough, not
    necessarily the series' overall maximum (which may come later)."""
    dates = _dates(5)
    # Overall max is 200 at the end, but the worst decline runs 150 -> 50.
    prices = pd.Series([100.0, 150.0, 50.0, 80.0, 200.0], index=dates)
    result = max_drawdown(prices)
    assert result["peak_date"] == dates[1]
    assert result["trough_date"] == dates[2]
    assert result["max_drawdown"] == pytest.approx(50.0 / 150.0 - 1.0, abs=1e-12)


# ---------------------------------------------------------------------
# Sharpe / Sortino: constructed return series with known mean/std
# ---------------------------------------------------------------------
def test_sharpe_ratio_hand_computed_zero_rf() -> None:
    dates = _dates(6)
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.015], index=dates)
    rf_annual = 0.0
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1  # == 0.0 exactly
    excess = returns - rf_daily
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(returns, rf_annual) == pytest.approx(expected, rel=1e-12)
    # Cross-check against a fully independent hand computation.
    mean = returns.mean()
    std = np.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1))
    assert sharpe_ratio(returns, rf_annual) == pytest.approx(
        mean / std * np.sqrt(252), rel=1e-10
    )


def test_sortino_ratio_hand_computed() -> None:
    dates = _dates(6)
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.015], index=dates)
    rf_annual = 0.0
    downside = returns[returns < 0]  # rf_daily == 0.0 exactly here
    expected = returns.mean() / downside.std(ddof=1) * np.sqrt(252)
    assert sortino_ratio(returns, rf_annual) == pytest.approx(expected, rel=1e-10)


def test_sortino_ge_sharpe_when_downside_vol_below_total_vol() -> None:
    """When gains are more volatile than losses, Sortino (penalising only
    downside) should read higher than Sharpe (penalising all volatility)."""
    rng = np.random.default_rng(42)
    dates = _dates(500)
    # Small, tight losses; large, varied gains.
    losses = -np.abs(rng.normal(0.002, 0.0005, 250))
    gains = np.abs(rng.normal(0.01, 0.01, 250))
    returns = pd.Series(np.concatenate([losses, gains]), index=dates)
    assert sortino_ratio(returns) > sharpe_ratio(returns)


def test_sharpe_higher_rf_lowers_the_ratio_for_positive_mean_returns() -> None:
    dates = _dates(100)
    rng = np.random.default_rng(9)
    returns = pd.Series(rng.normal(0.001, 0.01, 100), index=dates)
    assert sharpe_ratio(returns, rf_annual=0.0) > sharpe_ratio(returns, rf_annual=0.10)
