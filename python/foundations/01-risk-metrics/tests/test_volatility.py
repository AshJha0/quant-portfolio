"""Tests for eq_risk_metrics.volatility: returns and the three vol estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from eq_risk_metrics import (
    TRADING_DAYS,
    annualised_volatility,
    ewma_volatility,
    log_returns,
    rolling_volatility,
    simple_returns,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=n)


# ---------------------------------------------------------------------
# Return construction
# ---------------------------------------------------------------------
def test_simple_returns_hand_computed() -> None:
    prices = pd.Series([100.0, 110.0, 99.0, 108.9], index=_dates(4))
    rets = simple_returns(prices)
    expected = pd.Series([0.10, -0.10, 0.10], index=_dates(4)[1:])
    pd.testing.assert_series_equal(rets, expected, check_exact=False, atol=1e-12)


def test_log_returns_hand_computed() -> None:
    prices = pd.Series([100.0, 100.0 * np.e, 100.0], index=_dates(3))
    rets = log_returns(prices)
    expected = pd.Series([1.0, -1.0], index=_dates(3)[1:])
    pd.testing.assert_series_equal(rets, expected, check_exact=False, atol=1e-12)


def test_simple_returns_drops_first_nan_and_shrinks_by_one() -> None:
    prices = pd.Series([100.0, 101.0, 99.0, 102.0], index=_dates(4))
    rets = simple_returns(prices)
    assert len(rets) == len(prices) - 1
    assert not rets.isna().any()


# ---------------------------------------------------------------------
# Full-sample annualised volatility
# ---------------------------------------------------------------------
def test_annualised_volatility_hand_computed() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, -0.02], index=_dates(4))
    # mean = 0; sample variance (ddof=1) = sum(x^2) / (n - 1)
    expected_daily_std = np.sqrt((0.0001 + 0.0001 + 0.0004 + 0.0004) / 3)
    expected = expected_daily_std * np.sqrt(TRADING_DAYS)
    assert annualised_volatility(returns) == pytest.approx(expected, rel=1e-12)


def test_annualised_volatility_scales_with_sqrt_trading_days() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.03], index=_dates(5))
    daily_std = returns.std(ddof=1)
    assert annualised_volatility(returns) == pytest.approx(
        daily_std * np.sqrt(252), rel=1e-12
    )


def test_annualised_volatility_of_constant_series_is_zero() -> None:
    returns = pd.Series([0.001] * 10, index=_dates(10))
    # not exactly identical after arithmetic, but should be ~0
    assert annualised_volatility(returns) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------
# Rolling volatility
# ---------------------------------------------------------------------
def test_rolling_volatility_window_two_hand_computed() -> None:
    # For any 2-point sample, std(ddof=1) == |a - b| / sqrt(2) exactly.
    returns = pd.Series([0.01, -0.01, 0.02, -0.02], index=_dates(4))
    roll = rolling_volatility(returns, window=2)
    assert np.isnan(roll.iloc[0])
    expected = [
        abs(-0.01 - 0.01) / np.sqrt(2) * np.sqrt(TRADING_DAYS),
        abs(0.02 - (-0.01)) / np.sqrt(2) * np.sqrt(TRADING_DAYS),
        abs(-0.02 - 0.02) / np.sqrt(2) * np.sqrt(TRADING_DAYS),
    ]
    for got, exp in zip(roll.iloc[1:], expected, strict=False):
        assert got == pytest.approx(exp, rel=1e-12)


def test_rolling_volatility_matches_full_sample_when_window_equals_length() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.03], index=_dates(5))
    roll = rolling_volatility(returns, window=len(returns))
    assert roll.iloc[-1] == pytest.approx(annualised_volatility(returns), rel=1e-12)


# ---------------------------------------------------------------------
# EWMA volatility
# ---------------------------------------------------------------------
def test_ewma_volatility_two_points_hand_computed() -> None:
    # Zero-mean RiskMetrics recursion seeded with mean(r^2) over min(30, n) = 2 points.
    r0, r1 = 0.01, -0.02
    lam = 0.5
    returns = pd.Series([r0, r1], index=_dates(2))
    ewma = ewma_volatility(returns, lam=lam)
    s0 = (r0**2 + r1**2) / 2
    s1 = lam * s0 + (1 - lam) * r0**2
    assert ewma.iloc[0] == pytest.approx(np.sqrt(s0 * TRADING_DAYS), rel=1e-14)
    assert ewma.iloc[1] == pytest.approx(np.sqrt(s1 * TRADING_DAYS), rel=1e-14)


def test_ewma_volatility_no_warmup_nans() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, 0.03, -0.01], index=_dates(5))
    ewma = ewma_volatility(returns)
    assert not ewma.isna().any()
    assert len(ewma) == len(returns)


def test_ewma_reacts_faster_than_full_sample_to_a_late_shock() -> None:
    """A single large shock at the end should move EWMA vol far more than
    the full-sample (unconditional) figure -- the whole point of EWMA."""
    rng = np.random.default_rng(7)
    calm = rng.normal(0, 0.005, 250)
    shock = np.array([0.15, -0.12, 0.10])
    returns = pd.Series(np.concatenate([calm, shock]), index=_dates(253))
    full_sample = annualised_volatility(returns)
    ewma_latest = ewma_volatility(returns, lam=0.94).iloc[-1]
    assert ewma_latest > full_sample


def test_ewma_lambda_near_one_barely_updates() -> None:
    # lambda -> 1: the variance stays at the seed, so a late shock barely moves it.
    returns = pd.Series([0.02, -0.05, 0.10, -0.03], index=_dates(4))
    ewma = ewma_volatility(returns, lam=1 - 1e-9)
    seed = np.sqrt(np.mean(returns.to_numpy() ** 2) * TRADING_DAYS)
    assert ewma.iloc[-1] == pytest.approx(seed, rel=1e-6)
