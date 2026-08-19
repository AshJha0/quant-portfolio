"""Tests for eq_risk_metrics.var_es: VaR (3 methods) and Expected Shortfall."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from eq_risk_metrics import (
    expected_shortfall,
    var_cornish_fisher,
    var_historical,
    var_parametric,
)
import eq_risk_metrics.var_es as var_es_module


# ---------------------------------------------------------------------
# Historical VaR: hand-computed on a small, exactly-known sample
# ---------------------------------------------------------------------
def test_var_historical_hand_computed_95() -> None:
    returns = pd.Series([-0.05, -0.03, -0.01, 0.00, 0.01, 0.02, 0.04])
    # 5th percentile via linear interpolation: index = (n-1)*0.05 = 0.3
    # value = sorted[0] + 0.3*(sorted[1]-sorted[0]) = -0.05 + 0.3*0.02 = -0.044
    assert var_historical(returns, 0.95) == pytest.approx(0.044, abs=1e-12)


def test_var_historical_hand_computed_99() -> None:
    returns = pd.Series([-0.05, -0.03, -0.01, 0.00, 0.01, 0.02, 0.04])
    # 1st percentile: index = 6*0.01 = 0.06
    # value = -0.05 + 0.06*0.02 = -0.0488
    assert var_historical(returns, 0.99) == pytest.approx(0.0488, abs=1e-12)


def test_var_historical_single_observation_boundary() -> None:
    # A single-point sample: any percentile of one point is that point.
    returns = pd.Series([-0.02])
    assert var_historical(returns, 0.95) == pytest.approx(0.02, abs=1e-12)
    assert var_historical(returns, 0.99) == pytest.approx(0.02, abs=1e-12)


# ---------------------------------------------------------------------
# Gaussian (parametric) VaR: exact closed form -(mu + sigma*z)
# ---------------------------------------------------------------------
def test_var_parametric_matches_closed_form_exactly() -> None:
    rng = np.random.default_rng(11)
    returns = pd.Series(rng.normal(0.0004, 0.012, 400))
    mu, sigma = returns.mean(), returns.std(ddof=1)
    for confidence in (0.90, 0.95, 0.975, 0.99):
        z = stats.norm.ppf(1 - confidence)
        expected = -(mu + sigma * z)
        assert var_parametric(returns, confidence) == pytest.approx(expected, abs=1e-12)


def test_var_parametric_zero_mean_unit_scale_matches_standard_normal_quantile() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(2000)
    returns = pd.Series((x - x.mean()) / x.std(ddof=1))  # exactly mean 0, std 1
    for confidence, expected_z in ((0.95, 1.6448536269514722), (0.99, 2.3263478740408408)):
        assert var_parametric(returns, confidence) == pytest.approx(expected_z, abs=1e-9)


def test_var_parametric_increases_with_confidence() -> None:
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0.0003, 0.01, 500))
    v95 = var_parametric(returns, 0.95)
    v99 = var_parametric(returns, 0.99)
    assert v99 > v95


# ---------------------------------------------------------------------
# Cornish-Fisher VaR: collapses to Gaussian VaR when skew = kurtosis = 0
# ---------------------------------------------------------------------
def test_cornish_fisher_collapses_to_gaussian_when_skew_kurtosis_zero(monkeypatch) -> None:
    rng = np.random.default_rng(13)
    returns = pd.Series(rng.normal(0.0002, 0.015, 300))
    monkeypatch.setattr(var_es_module.stats, "skew", lambda x: 0.0)
    monkeypatch.setattr(var_es_module.stats, "kurtosis", lambda x: 0.0)
    for confidence in (0.90, 0.95, 0.99):
        cf = var_cornish_fisher(returns, confidence)
        gauss = var_parametric(returns, confidence)
        assert cf == pytest.approx(gauss, abs=1e-12)


def test_cornish_fisher_differs_from_gaussian_with_real_skew_kurtosis() -> None:
    rng = np.random.default_rng(17)
    # Student-t shocks -> genuine excess kurtosis.
    returns = pd.Series(rng.standard_t(df=3, size=2000) * 0.01)
    cf99 = var_cornish_fisher(returns, 0.99)
    gauss99 = var_parametric(returns, 0.99)
    assert cf99 != pytest.approx(gauss99, abs=1e-6)


# ---------------------------------------------------------------------
# Expected Shortfall
# ---------------------------------------------------------------------
def test_expected_shortfall_hand_computed() -> None:
    returns = pd.Series(np.round(np.arange(-0.10, 0.10, 0.01), 10))
    # VaR90 = 0.081 (10th percentile = -0.081); tail = {-0.10, -0.09}
    es = expected_shortfall(returns, 0.90)
    assert es == pytest.approx(0.095, abs=1e-12)


@pytest.mark.parametrize("confidence", [0.90, 0.95, 0.975, 0.99])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_expected_shortfall_greater_or_equal_to_var(seed: int, confidence: float) -> None:
    """Property test: ES >= historical VaR at the same confidence, for any
    sample -- the tail average can never be milder than the tail's edge."""
    rng = np.random.default_rng(seed)
    returns = pd.Series(rng.standard_t(df=4, size=500) * 0.01 + rng.normal(0, 0.001))
    var = var_historical(returns, confidence)
    es = expected_shortfall(returns, confidence)
    assert es >= var - 1e-12


def test_expected_shortfall_coherence_intuition_fat_tails_exceed_thin_tails() -> None:
    """Coherence intuition: a fatter tail (same VaR-defining quantile
    behaviour) produces a strictly larger ES-VaR gap, since ES integrates
    over how bad the tail *beyond* VaR actually is."""
    rng = np.random.default_rng(21)
    thin = pd.Series(rng.normal(0, 0.01, 5000))
    fat = pd.Series(rng.standard_t(df=3, size=5000) * 0.01)
    gap_thin = expected_shortfall(thin, 0.99) - var_historical(thin, 0.99)
    gap_fat = expected_shortfall(fat, 0.99) - var_historical(fat, 0.99)
    assert gap_fat > gap_thin
