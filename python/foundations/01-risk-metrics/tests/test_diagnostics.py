"""Tests for eq_risk_metrics.diagnostics: normality_report / Jarque-Bera."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq_risk_metrics import normality_report


def test_jarque_bera_does_not_reject_large_normal_sample() -> None:
    rng = np.random.default_rng(100)
    returns = pd.Series(rng.normal(0.0003, 0.01, 5000))
    report = normality_report(returns)
    assert not report["normality_rejected_at_5pct"]
    assert report["jarque_bera_pvalue"] > 0.05
    assert abs(report["skewness"]) < 0.15
    assert abs(report["excess_kurtosis"]) < 0.3


def test_jarque_bera_rejects_fat_tailed_sample() -> None:
    rng = np.random.default_rng(101)
    # Student-t with few degrees of freedom: genuine excess kurtosis.
    returns = pd.Series(rng.standard_t(df=3, size=5000) * 0.01)
    report = normality_report(returns)
    assert report["normality_rejected_at_5pct"]
    assert report["jarque_bera_pvalue"] < 0.05
    assert report["excess_kurtosis"] > 1.0  # fat tails: kurtosis well above 0


def test_jarque_bera_rejects_skewed_sample() -> None:
    rng = np.random.default_rng(102)
    returns = pd.Series(rng.lognormal(mean=0.0, sigma=0.5, size=3000) - 1.5)
    report = normality_report(returns)
    assert report["normality_rejected_at_5pct"]
    assert report["skewness"] > 0.3


def test_normality_report_keys_and_types() -> None:
    rng = np.random.default_rng(103)
    returns = pd.Series(rng.normal(0, 0.01, 200))
    report = normality_report(returns)
    assert set(report.keys()) == {
        "skewness",
        "excess_kurtosis",
        "jarque_bera_stat",
        "jarque_bera_pvalue",
        "normality_rejected_at_5pct",
    }
    assert isinstance(report["normality_rejected_at_5pct"], (bool, np.bool_))
    assert report["jarque_bera_stat"] >= 0.0
    assert 0.0 <= report["jarque_bera_pvalue"] <= 1.0


def test_normality_report_on_exactly_constant_series_raises() -> None:
    """Skewness and kurtosis of a constant sample are 0/0."""
    with pytest.raises(ValueError, match="constant"):
        normality_report(pd.Series([0.01] * 50))