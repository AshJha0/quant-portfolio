"""Convention tests: Acerbi-Tasche ES, order-statistic VaR, RiskMetrics EWMA,
downside deviation, Cornish-Fisher domain and NaN refusal in the backtest.

Every expected value here is computed by hand from the definition, not by
calling the function under test.
"""
from __future__ import annotations

import math

import eq_risk_metrics as rm
import numpy as np
import pandas as pd
import pytest


def test_var_historical_is_order_statistic_not_interpolated() -> None:
    # n = 10, confidence 0.85 -> k = 1.5, m = 2: VaR = second largest loss.
    r = pd.Series([-0.10, -0.06, -0.02, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    assert rm.var_historical(r, 0.85) == pytest.approx(0.06, abs=1e-15)


def test_expected_shortfall_is_acerbi_tasche_with_fractional_atom() -> None:
    # k = 1.5: ES = (0.10 + 0.5 * 0.06) / 1.5.
    r = pd.Series([-0.10, -0.06, -0.02, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    assert rm.expected_shortfall(r, 0.85) == pytest.approx((0.10 + 0.5 * 0.06) / 1.5, abs=1e-15)


def test_expected_shortfall_on_ties_uses_the_atom_weight() -> None:
    # Old naive estimator averaged every return <= -VaR, so a tie at the
    # boundary moved ES. Losses 5, 1, 1, 1 ... at k = 2 give ES = (5 + 1) / 2.
    r = pd.Series([-5.0] + [-1.0] * 9 + [0.0] * 10)
    assert rm.expected_shortfall(r, 0.90) == pytest.approx(3.0, abs=1e-15)


def test_ewma_volatility_is_zero_mean_riskmetrics_recursion() -> None:
    r = pd.Series([0.01, -0.02, 0.015, 0.03, -0.01])
    lam = 0.94
    s2 = float(np.mean(r.to_numpy() ** 2))  # seed window min(30, 5) = 5
    expected = [s2]
    for t in range(1, 5):
        s2 = lam * s2 + (1 - lam) * r.iloc[t - 1] ** 2
        expected.append(s2)
    got = rm.ewma_volatility(r, lam=lam).to_numpy()
    np.testing.assert_allclose(got, np.sqrt(np.array(expected) * 252), rtol=1e-14)
    assert got[0] > 0.0


def test_ewma_volatility_seed_uses_only_the_seed_window() -> None:
    r = pd.Series([0.01] * 30 + [0.5] * 10)
    assert rm.ewma_volatility(r).iloc[0] == pytest.approx(0.01 * math.sqrt(252), rel=1e-14)


def test_sortino_uses_downside_deviation_over_all_observations() -> None:
    r = pd.Series([0.02, -0.01, -0.03, 0.04])
    dd = math.sqrt((0.01**2 + 0.03**2) / 4)
    expected = np.mean(r) / dd * math.sqrt(252)
    assert rm.sortino_ratio(r, rf_annual=0.0) == pytest.approx(expected, rel=1e-13)


def test_var_parametric_raises_on_one_observation() -> None:
    with pytest.raises(ValueError):
        rm.var_parametric(pd.Series([-0.01]), 0.95)


def test_kupiec_raises_on_all_nan_var() -> None:
    r = pd.Series(np.linspace(-0.02, 0.02, 50))
    with pytest.raises(ValueError):
        rm.kupiec_pof_test(r, pd.Series(np.nan, index=r.index), 0.99)


def test_kupiec_raises_on_nan_return_day() -> None:
    r = pd.Series(np.linspace(-0.02, 0.02, 50))
    r.iloc[3] = np.nan
    with pytest.raises(ValueError):
        rm.kupiec_pof_test(r, 0.01, 0.99)


@pytest.mark.parametrize(
    ("s", "k", "alpha", "valid"),
    [
        (0.122, -0.427, 0.01, True),  # minimum of g' sits at z = -4, outside [-2.33, 2.33]
        (0.122, -0.427, 0.001, True),
        (0.0, 0.0, 0.01, True),
        (1.0, 1.0, 0.01, True),  # g'(-2.326) = 0.013 > 0, smallest on the interval
        (1.0, 1.0, 0.001, False),  # g'(-3.090) = -0.41
        (-0.0105, 8.0001, 0.01, False),  # vertex at z ~ 0.0018 with g' ~ -2.5e-7
    ],
)
def test_cornish_fisher_valid_closed_form(s: float, k: float, alpha: float, valid: bool) -> None:
    assert rm.cornish_fisher_valid(s, k, alpha) is valid


def test_cornish_fisher_valid_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        rm.cornish_fisher_valid(float("nan"), 0.0, 0.01)
    with pytest.raises(ValueError):
        rm.cornish_fisher_valid(0.0, 0.0, 0.99)


def test_var_cornish_fisher_raises_outside_domain() -> None:
    # One extreme outlier gives skew and kurtosis far outside the domain.
    r = pd.Series([0.0] * 199 + [1.0])
    with pytest.raises(ValueError, match="domain"):
        rm.var_cornish_fisher(r + np.linspace(0, 1e-6, 200), 0.99)
