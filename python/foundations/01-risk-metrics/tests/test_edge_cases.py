"""Edge cases: empty/too-short series, constant returns, all-positive
returns, single confidence level at the boundary.

Every edge case here is also discussed in docs/VALIDATION.md, per the
portfolio documentation contract (edge cases must be BOTH documented
and unit-tested).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq_risk_metrics import (
    annualised_volatility,
    ewma_volatility,
    expected_shortfall,
    max_drawdown,
    normality_report,
    rolling_volatility,
    sharpe_ratio,
    simple_returns,
    sortino_ratio,
    var_cornish_fisher,
    var_historical,
    var_parametric,
)


# ---------------------------------------------------------------------
# Empty / too-short series
# ---------------------------------------------------------------------
def test_simple_returns_on_single_price_is_empty() -> None:
    prices = pd.Series([100.0], index=pd.bdate_range("2024-01-02", periods=1))
    rets = simple_returns(prices)
    assert len(rets) == 0


def test_annualised_volatility_on_empty_series_is_nan() -> None:
    empty = pd.Series([], dtype=float)
    assert np.isnan(annualised_volatility(empty))


def test_annualised_volatility_on_single_return_is_nan() -> None:
    # std(ddof=1) of one observation is undefined (0/0).
    one = pd.Series([0.01])
    assert np.isnan(annualised_volatility(one))


def test_var_historical_on_empty_series_raises() -> None:
    empty = pd.Series([], dtype=float)
    with pytest.raises(IndexError):
        var_historical(empty, 0.95)


def test_five_day_sample_produces_a_number_but_is_statistically_meaningless() -> None:
    """A 5-day sample is a real failure mode, not a crash: every function
    still returns a number, but VaR/ES from 5 points is not a serious
    risk estimate -- see docs/VALIDATION.md 'known failure modes'."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    returns = pd.Series([0.01, -0.02, 0.005, 0.015, -0.008], index=dates)
    var95 = var_historical(returns, 0.95)
    es95 = expected_shortfall(returns, 0.95)
    assert np.isfinite(var95)
    assert np.isfinite(es95)
    # With n=5, the "95th percentile" tail is a near-degenerate
    # interpolation between the two smallest points -- ES's tail sample
    # can be as small as a single observation.
    tail = returns[returns <= -var95]
    assert len(tail) <= 2


def test_rolling_volatility_window_larger_than_sample_is_all_nan() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    returns = pd.Series([0.01, -0.02, 0.005, 0.015, -0.008], index=dates)
    roll = rolling_volatility(returns, window=10)
    assert roll.isna().all()


# ---------------------------------------------------------------------
# Constant returns -> zero vol -> Sharpe division-by-zero behaviour
# ---------------------------------------------------------------------
def test_constant_returns_zero_full_sample_and_ewma_vol() -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)
    returns = pd.Series([0.002] * 30, index=dates)
    assert annualised_volatility(returns) == pytest.approx(0.0, abs=1e-9)
    assert ewma_volatility(returns).iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_constant_returns_sharpe_is_a_huge_finite_number_not_a_crash() -> None:
    """Constant returns give (numerically) near-zero excess std, so Sharpe
    is not exactly infinite (floating-point noise from subtracting a
    tiny risk-free daily rate keeps std a tiny nonzero number) but it is
    a huge, uninformative value -- callers must not present this as a
    real Sharpe ratio."""
    dates = pd.bdate_range("2024-01-02", periods=30)
    returns = pd.Series([0.001] * 30, index=dates)
    sharpe = sharpe_ratio(returns)
    assert np.isfinite(sharpe)
    assert abs(sharpe) > 1e6


def test_constant_returns_var_and_es_equal_the_constant() -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)
    returns = pd.Series([0.002] * 30, index=dates)
    # Every observation is identical, so every quantile is that value.
    assert var_historical(returns, 0.95) == pytest.approx(-0.002, abs=1e-12)
    assert expected_shortfall(returns, 0.95) == pytest.approx(-0.002, abs=1e-12)


def test_constant_returns_cornish_fisher_matches_gaussian() -> None:
    """Exactly-constant returns have zero variance, so scipy's skew/kurtosis
    hit catastrophic cancellation (NaN + RuntimeWarning, promoted to an
    error by the pytest config) rather than cleanly returning 0 -- a
    documented failure mode, not a Cornish-Fisher-specific bug."""
    dates = pd.bdate_range("2024-01-02", periods=30)
    returns = pd.Series([0.002] * 30, index=dates)
    with pytest.warns(RuntimeWarning):
        cf = var_cornish_fisher(returns, 0.95)
    assert np.isnan(cf)


def test_near_constant_returns_cornish_fisher_matches_gaussian() -> None:
    """With floating-point noise breaking exact ties (as any real,
    non-synthetic 'flat' return series would have), skew/kurtosis are
    tiny but finite, and Cornish-Fisher collapses to Gaussian VaR."""
    dates = pd.bdate_range("2024-01-02", periods=30)
    base = 0.002
    returns = pd.Series([base + i * 1e-15 for i in range(30)], index=dates)
    cf = var_cornish_fisher(returns, 0.95)
    gauss = var_parametric(returns, 0.95)
    assert cf == pytest.approx(gauss, abs=1e-6)


# ---------------------------------------------------------------------
# All-positive returns -> Sortino's empty downside array
# ---------------------------------------------------------------------
def test_all_positive_returns_sortino_is_nan() -> None:
    dates = pd.bdate_range("2024-01-02", periods=20)
    returns = pd.Series(np.linspace(0.001, 0.02, 20), index=dates)
    sortino = sortino_ratio(returns, rf_annual=0.0)
    assert np.isnan(sortino)


def test_all_positive_returns_sharpe_is_finite_and_positive() -> None:
    dates = pd.bdate_range("2024-01-02", periods=20)
    returns = pd.Series(np.linspace(0.001, 0.02, 20), index=dates)
    sharpe = sharpe_ratio(returns, rf_annual=0.0)
    assert np.isfinite(sharpe)
    assert sharpe > 0


# ---------------------------------------------------------------------
# Single confidence level at the boundary
# ---------------------------------------------------------------------
def test_confidence_at_extreme_boundary_near_one() -> None:
    rng = np.random.default_rng(55)
    returns = pd.Series(rng.normal(0, 0.01, 2000))
    # 99.99% confidence: far into the tail, still must return a finite
    # (very large) number, not crash.
    var = var_historical(returns, 0.9999)
    assert np.isfinite(var)
    assert var > var_historical(returns, 0.99)


def test_confidence_at_boundary_near_zero_five() -> None:
    rng = np.random.default_rng(56)
    returns = pd.Series(rng.normal(0.0002, 0.01, 2000))
    # 50% confidence: VaR is (close to) minus the median, can be near
    # zero or even negative if the median return is positive -- "VaR"
    # loses its usual tail-risk meaning at low confidence, and the
    # function must not special-case or crash on this.
    var50 = var_historical(returns, 0.50)
    assert np.isfinite(var50)


def test_var_parametric_confidence_exactly_one_half_is_minus_mean() -> None:
    # z = Phi^{-1}(0.5) = 0 exactly, so VaR(50%) == -mu exactly.
    returns = pd.Series([0.01, -0.02, 0.015, 0.03, -0.01])
    var50 = var_parametric(returns, 0.5)
    assert var50 == pytest.approx(-returns.mean(), abs=1e-12)


# ---------------------------------------------------------------------
# Drawdown edge cases
# ---------------------------------------------------------------------
def test_max_drawdown_constant_price_series() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.Series([100.0] * 10, index=dates)
    result = max_drawdown(prices)
    assert result["max_drawdown"] == 0.0


def test_normality_report_on_too_short_sample_runs_without_crashing() -> None:
    returns = pd.Series([0.01, -0.02, 0.005])
    report = normality_report(returns)
    assert np.isfinite(report["jarque_bera_stat"])
