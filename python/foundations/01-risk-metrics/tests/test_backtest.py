"""Tests for eq_risk_metrics.backtest: exception counting + Kupiec POF test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from eq_risk_metrics import (
    count_var_exceptions,
    kupiec_pof_test,
    var_historical,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=n)


# ---------------------------------------------------------------------
# Exception counting
# ---------------------------------------------------------------------
def test_count_var_exceptions_hand_checked() -> None:
    returns = pd.Series([-0.05, -0.02, 0.01, -0.021, 0.03, -0.0199], index=_dates(6))
    hits = count_var_exceptions(returns, var=0.02)
    # Exceptions are strictly below -0.02: -0.05 and -0.021 qualify;
    # -0.02 lands exactly on the threshold and does not.
    assert hits.tolist() == [True, False, False, True, False, False]
    assert hits.index.equals(returns.index)


def test_count_var_exceptions_accepts_a_rolling_var_forecast() -> None:
    """The realistic case: VaR is re-estimated daily, so the threshold is a
    Series, and each day is judged against *its own* forecast."""
    returns = pd.Series([-0.03, -0.03, -0.03], index=_dates(3))
    var = pd.Series([0.02, 0.04, 0.02], index=_dates(3))
    assert count_var_exceptions(returns, var).tolist() == [True, False, True]


def test_count_var_exceptions_rejects_misaligned_var_series() -> None:
    returns = pd.Series([-0.03, -0.01], index=_dates(2))
    var = pd.Series([0.02, 0.02, 0.02], index=_dates(3))
    with pytest.raises(ValueError, match="share the index"):
        count_var_exceptions(returns, var)


def test_count_var_exceptions_rejects_negative_var_sign_convention_error() -> None:
    """VaR in this package is a positive loss fraction; a negative value
    means someone passed a raw return quantile and every count would be
    silently inverted."""
    returns = pd.Series([-0.03, -0.01], index=_dates(2))
    with pytest.raises(ValueError, match="POSITIVE loss fraction"):
        count_var_exceptions(returns, -0.02)


def test_count_var_exceptions_rejects_empty_returns() -> None:
    with pytest.raises(ValueError, match="empty"):
        count_var_exceptions(pd.Series([], dtype=float), 0.02)


# ---------------------------------------------------------------------
# Kupiec POF: analytic properties
# ---------------------------------------------------------------------
def test_kupiec_statistic_is_exactly_zero_when_observed_rate_equals_nominal() -> None:
    """The LR statistic compares the null rate against the unrestricted MLE
    (the observed rate). When they coincide the two likelihoods are equal
    and the statistic collapses to 0 (up to the ~1e-17 representation
    error in ``1 - 0.99``), so the model is nowhere near rejected."""
    n, x = 100, 1  # 1% observed, 99% VaR -> nominal 1%
    returns = pd.Series([-0.05] * x + [0.001] * (n - x), index=_dates(n))
    result = kupiec_pof_test(returns, var=0.02, confidence=0.99)
    assert result["n_exceptions"] == x
    assert result["observed_rate"] == pytest.approx(0.01, abs=1e-15)
    assert result["lr_statistic"] == pytest.approx(0.0, abs=1e-6)
    assert result["p_value"] == pytest.approx(1.0, abs=1e-6)
    assert not result["reject_at_5pct"]


def test_kupiec_statistic_matches_a_hand_computed_value() -> None:
    """n=100, x=5, nominal p=1%: LR = 8.2582170028716..., chi2(1) p-value
    0.0040567952567397 (computed independently from the closed form)."""
    n, x = 100, 5
    returns = pd.Series([-0.05] * x + [0.001] * (n - x), index=_dates(n))
    result = kupiec_pof_test(returns, var=0.02, confidence=0.99)
    assert result["lr_statistic"] == pytest.approx(8.258217002871675, rel=1e-12)
    assert result["p_value"] == pytest.approx(0.004056795256739709, rel=1e-10)
    assert result["reject_at_5pct"]
    assert result["expected_exceptions"] == pytest.approx(1.0, abs=1e-12)


def test_kupiec_statistic_equals_the_chi2_survival_of_its_own_statistic() -> None:
    """Internal consistency: the reported p-value is the upper tail of a
    chi-squared(1) at the reported statistic, for an arbitrary sample."""
    rng = np.random.default_rng(31)
    returns = pd.Series(rng.standard_t(df=4, size=750) * 0.01, index=_dates(750))
    result = kupiec_pof_test(returns, var=0.02, confidence=0.99)
    assert result["p_value"] == pytest.approx(
        stats.chi2.sf(result["lr_statistic"], df=1), rel=1e-12
    )


def test_kupiec_zero_exceptions_is_finite_and_rejects_an_absurdly_high_var() -> None:
    """The x=0 corner needs the 0*log(0)=0 convention or the statistic is
    NaN. A VaR of 100% is never breached, and over 1,000 days that is
    itself evidence the model is wrong (too conservative)."""
    rng = np.random.default_rng(32)
    returns = pd.Series(rng.normal(0, 0.01, 1000), index=_dates(1000))
    result = kupiec_pof_test(returns, var=1.0, confidence=0.99)
    assert result["n_exceptions"] == 0
    assert np.isfinite(result["lr_statistic"])
    assert result["lr_statistic"] > 0
    assert result["reject_at_5pct"]
    assert result["observed_rate"] < result["expected_rate"]  # too conservative


def test_kupiec_all_days_exceptions_is_finite() -> None:
    """The mirror corner, x = n: a VaR of 0 is breached on every losing
    day. The statistic must stay finite (the (n-x) term drops out)."""
    returns = pd.Series([-0.01] * 50, index=_dates(50))
    result = kupiec_pof_test(returns, var=0.0, confidence=0.99)
    assert result["n_exceptions"] == 50
    assert np.isfinite(result["lr_statistic"])
    assert result["reject_at_5pct"]


def test_kupiec_rejects_a_badly_understated_var() -> None:
    """The direction that actually matters on a desk: a VaR that is far too
    small produces far too many exceptions and must be rejected."""
    rng = np.random.default_rng(33)
    returns = pd.Series(rng.normal(0, 0.02, 1000), index=_dates(1000))
    result = kupiec_pof_test(returns, var=0.005, confidence=0.99)
    assert result["observed_rate"] > result["expected_rate"]
    assert result["p_value"] < 1e-6
    assert result["reject_at_5pct"]


def test_kupiec_does_not_reject_an_in_sample_historical_var() -> None:
    """Sanity check on the estimator this package ships: historical VaR
    evaluated on the very sample it was fitted to has, by construction, an
    exception rate of about 1 - confidence, so Kupiec should not reject.
    (In-sample coverage is a necessary, not sufficient, condition -- a real
    backtest is out-of-sample; see docs/DESK_GUIDE.md.)"""
    rng = np.random.default_rng(34)
    returns = pd.Series(rng.standard_t(df=5, size=2000) * 0.01, index=_dates(2000))
    var99 = var_historical(returns, 0.99)
    result = kupiec_pof_test(returns, var99, confidence=0.99)
    assert not result["reject_at_5pct"]
    assert result["n_exceptions"] == pytest.approx(20, abs=6)


def test_kupiec_power_is_low_on_a_short_window() -> None:
    """A documented limitation, pinned as a test: on a 250-day window at
    99% you expect 2.5 exceptions, and seeing twice that many is *not*
    enough to reject at 5%. This is why Kupiec alone is a weak control."""
    n, x = 250, 5
    returns = pd.Series([-0.05] * x + [0.001] * (n - x), index=_dates(n))
    result = kupiec_pof_test(returns, var=0.02, confidence=0.99)
    assert result["observed_rate"] == pytest.approx(2 * result["expected_rate"], rel=1e-12)
    assert not result["reject_at_5pct"]


def test_kupiec_is_blind_to_clustering_by_construction() -> None:
    """Unconditional coverage ignores *when* exceptions happen: the same
    exceptions spread out and bunched together give an identical result.
    That blindness is exactly what Christoffersen's independence test
    exists to fix, and why this module documents itself as half a
    backtest."""
    n, x = 500, 5
    spread = [-0.05 if i % 100 == 0 else 0.001 for i in range(n)]
    bunched = [-0.05 if i < x else 0.001 for i in range(n)]
    a = kupiec_pof_test(pd.Series(spread, index=_dates(n)), 0.02, 0.99)
    b = kupiec_pof_test(pd.Series(bunched, index=_dates(n)), 0.02, 0.99)
    assert a["lr_statistic"] == pytest.approx(b["lr_statistic"], rel=1e-15)
    assert a["p_value"] == pytest.approx(b["p_value"], rel=1e-15)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 2.0, np.nan])
def test_kupiec_rejects_invalid_confidence(confidence: float) -> None:
    returns = pd.Series([-0.01, 0.02, 0.005], index=_dates(3))
    with pytest.raises(ValueError, match="confidence must be"):
        kupiec_pof_test(returns, 0.02, confidence)


def test_kupiec_rejects_empty_returns() -> None:
    with pytest.raises(ValueError, match="empty"):
        kupiec_pof_test(pd.Series([], dtype=float), 0.02, 0.99)


def test_kupiec_result_keys_and_ranges() -> None:
    rng = np.random.default_rng(35)
    returns = pd.Series(rng.normal(0, 0.01, 600), index=_dates(600))
    result = kupiec_pof_test(returns, var_historical(returns, 0.95), 0.95)
    assert set(result.keys()) == {
        "n_observations",
        "n_exceptions",
        "observed_rate",
        "expected_rate",
        "expected_exceptions",
        "lr_statistic",
        "p_value",
        "reject_at_5pct",
    }
    assert result["n_observations"] == 600
    assert 0 <= result["n_exceptions"] <= 600
    assert result["lr_statistic"] >= 0.0
    assert 0.0 <= result["p_value"] <= 1.0
    assert isinstance(result["reject_at_5pct"], bool)
