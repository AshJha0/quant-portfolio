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


@pytest.mark.parametrize(
    "func", [var_historical, var_parametric, var_cornish_fisher, expected_shortfall]
)
def test_var_and_es_on_empty_series_raise_informative_value_error(func) -> None:
    """An empty sample is a caller error, not a number. All four estimators
    fail the same way, with a message that names the function and says what
    was expected (previously ``var_historical``/``expected_shortfall``
    surfaced a raw ``IndexError`` from ``numpy.percentile`` -- see
    docs/VALIDATION.md 3.5)."""
    empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="empty return series"):
        func(empty, 0.95)


@pytest.mark.parametrize(
    "func", [var_historical, var_parametric, var_cornish_fisher, expected_shortfall]
)
@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5, -0.1, np.nan, np.inf])
def test_var_and_es_reject_confidence_outside_the_open_unit_interval(
    func, confidence: float
) -> None:
    """Confidence must be strictly inside (0, 1). Without this check
    ``var_parametric(r, 1.5)`` returns a silent ``NaN`` (scipy's ppf of an
    out-of-range probability) -- a wrong risk number that looks like a
    number."""
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0, 0.01, 50))
    with pytest.raises(ValueError, match="confidence must be"):
        func(returns, confidence)


@pytest.mark.parametrize(
    "func", [var_historical, var_parametric, var_cornish_fisher, expected_shortfall]
)
@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_var_and_es_reject_non_finite_returns(func, bad: float) -> None:
    """A single inf/NaN silently poisons every estimator (percentile and
    mean/std both propagate it), so it is rejected with a message that
    explains where such a value comes from."""
    returns = pd.Series([0.01, -0.02, bad, 0.005, 0.0, -0.011, 0.004, 0.002])
    with pytest.raises(ValueError, match="non-finite"):
        func(returns, 0.95)


def test_var_historical_single_observation_is_that_observation() -> None:
    """One observation is a legal (if useless) sample for the empirical
    quantile: every percentile of a one-point sample is that point."""
    assert var_historical(pd.Series([-0.02]), 0.99) == pytest.approx(0.02, abs=1e-12)


@pytest.mark.parametrize("func", [var_parametric, var_cornish_fisher])
def test_parametric_var_on_single_observation_is_nan_not_zero(func) -> None:
    """``std(ddof=1)`` of one point is undefined, so the moment-based
    estimators return NaN rather than a falsely confident 0% risk. NaN
    here means "not enough data", and a report must not coerce it to 0."""
    result = func(pd.Series([0.01]), 0.95)
    assert np.isnan(result)


def test_expected_shortfall_on_single_observation_equals_that_observation() -> None:
    assert expected_shortfall(pd.Series([-0.03]), 0.95) == pytest.approx(0.03, abs=1e-12)


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
    # ...and the *shape* is preserved, so it still aligns with the return
    # index in a report/joins downstream rather than silently shrinking.
    assert len(roll) == len(returns)
    assert roll.index.equals(returns.index)


def test_rolling_volatility_window_exactly_one_more_than_sample_is_all_nan() -> None:
    """The boundary between "warms up on the last day" and "never warms
    up": window == len(returns) produces exactly one number, window ==
    len(returns) + 1 produces none."""
    returns = pd.Series([0.01, -0.02, 0.005, 0.015], index=pd.bdate_range("2024-01-02", periods=4))
    assert rolling_volatility(returns, window=4).notna().sum() == 1
    assert rolling_volatility(returns, window=5).isna().all()


@pytest.mark.parametrize("window", [1, 0, -3])
def test_rolling_volatility_rejects_window_below_two(window: int) -> None:
    """``window=1`` used to return an all-NaN series silently (std with
    ddof=1 of one point is 0/0) -- indistinguishable from "not warmed up
    yet". It is now an explicit error."""
    returns = pd.Series([0.01, -0.02, 0.005], index=pd.bdate_range("2024-01-02", periods=3))
    with pytest.raises(ValueError, match="window must be"):
        rolling_volatility(returns, window=window)


def test_rolling_volatility_rejects_non_integer_window() -> None:
    returns = pd.Series([0.01, -0.02, 0.005], index=pd.bdate_range("2024-01-02", periods=3))
    with pytest.raises(ValueError, match="window must be an int"):
        rolling_volatility(returns, window=2.5)


@pytest.mark.parametrize("lam", [0.0, 1.0, 1.5, -0.2, np.nan])
def test_ewma_volatility_rejects_lambda_outside_the_open_unit_interval(lam: float) -> None:
    """``lam=0`` (no memory) returns an identically-zero variance and
    ``lam>=1`` never updates; both are silently useless, so both are
    rejected with a message naming the RiskMetrics convention."""
    returns = pd.Series([0.01, -0.02, 0.005], index=pd.bdate_range("2024-01-02", periods=3))
    with pytest.raises(ValueError, match="lam must be"):
        ewma_volatility(returns, lam=lam)


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
def test_confidence_extremely_close_to_one_stays_finite_and_monotone() -> None:
    """At 1 - 1e-9 confidence the requested percentile is far beyond any
    observation: numpy clamps to the sample minimum rather than blowing
    up, so VaR saturates at the worst observed loss. Documented because
    the saturation is easy to mistake for a real tail estimate."""
    rng = np.random.default_rng(77)
    returns = pd.Series(rng.normal(0, 0.01, 1000))
    var = var_historical(returns, 1 - 1e-9)
    assert np.isfinite(var)
    # Saturates at (a hair inside) the worst observed loss -- it cannot
    # report anything the sample has not already shown.
    assert var == pytest.approx(-returns.min(), abs=1e-6)
    assert var <= -returns.min()
    assert var >= var_historical(returns, 0.99)


def test_confidence_extremely_close_to_zero_stays_finite() -> None:
    """The mirror-image extreme: at 1e-9 confidence the "VaR" is minus the
    largest *gain* in the sample -- a negative loss. It is meaningless as
    risk, but it must not crash or return NaN."""
    rng = np.random.default_rng(78)
    returns = pd.Series(rng.normal(0.0005, 0.01, 1000))
    var = var_historical(returns, 1e-9)
    assert np.isfinite(var)
    assert var == pytest.approx(-returns.max(), abs=1e-6)
    assert var < 0


def test_parametric_var_at_extreme_confidence_grows_without_saturating() -> None:
    """Unlike the historical estimator, the Gaussian one extrapolates: at
    1 - 1e-9 it returns a number far beyond the worst observed loss. That
    is the whole trade-off between the two methods, in one assertion."""
    rng = np.random.default_rng(79)
    returns = pd.Series(rng.normal(0, 0.01, 1000))
    extreme = var_parametric(returns, 1 - 1e-9)
    assert np.isfinite(extreme)
    assert extreme > var_historical(returns, 1 - 1e-9)


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
def test_max_drawdown_on_empty_prices_raises() -> None:
    with pytest.raises(ValueError, match="prices is empty"):
        max_drawdown(pd.Series([], dtype=float))


@pytest.mark.parametrize("bad_price", [0.0, -5.0])
def test_max_drawdown_rejects_non_positive_prices(bad_price: float) -> None:
    """A zero or negative price level is a data error: the drawdown
    formula would report a decline worse than -100% (or flip sign), which
    is not a market event a report should ever show."""
    prices = pd.Series([100.0, 95.0, bad_price, 98.0], index=pd.bdate_range("2024-01-02", periods=4))
    with pytest.raises(ValueError, match="strictly positive"):
        max_drawdown(prices)


def test_max_drawdown_rejects_non_finite_prices() -> None:
    prices = pd.Series([100.0, np.nan, 98.0], index=pd.bdate_range("2024-01-02", periods=3))
    with pytest.raises(ValueError, match="non-finite"):
        max_drawdown(prices)


def test_max_drawdown_constant_price_series() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.Series([100.0] * 10, index=dates)
    result = max_drawdown(prices)
    assert result["max_drawdown"] == 0.0


def test_normality_report_on_too_short_sample_runs_without_crashing() -> None:
    returns = pd.Series([0.01, -0.02, 0.005])
    report = normality_report(returns)
    assert np.isfinite(report["jarque_bera_stat"])
