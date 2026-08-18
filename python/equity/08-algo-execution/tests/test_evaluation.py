"""Evaluation tests: Newey-West vs naive SE, deflated Sharpe identities and
monotonicity, performance stats, capacity monotone in AUM."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from eq_algo import (capacity_curve, deflated_sharpe_ratio, expected_max_sharpe,
                     max_drawdown, newey_west_se, newey_west_tstat,
                     probabilistic_sharpe_ratio, quantile_monotonicity,
                     sharpe_ratio, sortino_ratio)


def _ar1(n=1000, rho=0.6, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.standard_normal()
    return pd.Series(x + 0.05)


def test_newey_west_exceeds_naive_under_autocorrelation():
    x = _ar1()
    naive = x.std(ddof=0) / np.sqrt(len(x))
    assert newey_west_se(x, lags=10) > 1.3 * naive


def test_newey_west_lags_zero_equals_naive():
    x = _ar1(rho=0.0, seed=3)
    naive = x.std(ddof=0) / np.sqrt(len(x))
    assert newey_west_se(x, lags=0) == pytest.approx(naive, rel=1e-12)


def test_newey_west_hand_computed():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    d = x - 2.5
    g0 = (d @ d) / 4
    g1 = (d[1:] @ d[:-1]) / 4
    expected = np.sqrt((g0 + 2 * 0.5 * g1) / 4)  # Bartlett weight (1 - 1/2)
    assert newey_west_se(x, lags=1) == pytest.approx(expected, rel=1e-12)
    assert newey_west_tstat(x, lags=1) == pytest.approx(2.5 / expected, rel=1e-12)


def test_psr_hand_checked_formula():
    sr, n = 0.1, 101
    z = sr * np.sqrt(n - 1) / np.sqrt(1.0 + (3.0 - 1.0) / 4.0 * sr**2)
    assert probabilistic_sharpe_ratio(sr, 0.0, n) == pytest.approx(
        stats.norm.cdf(z), rel=1e-12)
    # skew/kurt enter the denominator
    z2 = (sr - 0.02) * np.sqrt(n - 1) / np.sqrt(1.0 - (-0.5) * sr + (5.0 - 1.0) / 4.0 * sr**2)
    assert probabilistic_sharpe_ratio(sr, 0.02, n, skew=-0.5, kurt=5.0) == \
        pytest.approx(stats.norm.cdf(z2), rel=1e-12)


def test_expected_max_sharpe_properties():
    assert expected_max_sharpe(1, 0.01) == 0.0
    vals = [expected_max_sharpe(n, 0.01) for n in (2, 5, 20, 100, 1000)]
    assert all(b > a for a, b in zip(vals, vals[1:]))  # increasing in N
    assert expected_max_sharpe(10, 0.04) == pytest.approx(
        2.0 * expected_max_sharpe(10, 0.01), rel=1e-12)  # scales with sqrt(V)
    # hand check the Bailey-LdP expression at N=10, V=1
    g = 0.5772156649015329
    expected = (1 - g) * stats.norm.ppf(1 - 1 / 10) + g * stats.norm.ppf(1 - 1 / (10 * np.e))
    assert expected_max_sharpe(10, 1.0) == pytest.approx(expected, rel=1e-12)


def test_dsr_decreases_with_trials_and_equals_psr_at_one():
    rng = np.random.default_rng(11)
    r = pd.Series(rng.standard_normal(500) * 0.01 + 0.0006)
    out = [deflated_sharpe_ratio(r, n_trials=n)["dsr"] for n in (1, 2, 5, 10, 100)]
    assert all(b < a for a, b in zip(out, out[1:]))  # strictly decreasing in N
    d1 = deflated_sharpe_ratio(r, n_trials=1)
    assert d1["dsr"] == pytest.approx(d1["psr0"], rel=1e-12)  # DSR = PSR at N=1
    assert d1["sr_benchmark"] == 0.0


def test_sharpe_sortino_mdd_hand_computed():
    r = pd.Series([0.10, -0.10])
    assert sharpe_ratio(r) == pytest.approx(0.0, abs=1e-12)
    assert max_drawdown(r) == pytest.approx(0.10, abs=1e-12)
    assert sortino_ratio(r) == pytest.approx(0.0, abs=1e-12)
    r2 = pd.Series([0.02, 0.01, -0.01])
    expected_sortino = r2.mean() / np.sqrt((0.01**2) / 3) * np.sqrt(252)
    assert sortino_ratio(r2) == pytest.approx(expected_sortino, rel=1e-12)


def test_quantile_monotonicity_signs():
    assert quantile_monotonicity([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert quantile_monotonicity([4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_capacity_curve_monotone_decreasing_in_aum():
    rng = np.random.default_rng(0)
    gross = pd.Series(rng.standard_normal(500) * 0.004 + 0.001)
    grid = [1e6, 1e7, 1e8, 1e9, 1e10]
    cap = capacity_curve(gross, mean_turnover=0.3, aum_grid=grid,
                         adv_dollars=5e7, n_names=20)
    ns = cap["net_sharpe"].to_numpy()
    assert np.all(np.diff(ns) < 0)                       # strictly decreasing
    assert np.all(np.diff(cap["daily_cost_drag"].to_numpy()) > 0)
    assert np.all(np.diff(cap["participation"].to_numpy()) > 0)


@pytest.mark.parametrize("bad_call", [
    lambda: newey_west_se([1.0], lags=1),
    lambda: newey_west_se([1.0, 2.0], lags=-1),
    lambda: sharpe_ratio([0.01, 0.01]),                       # zero vol
    lambda: sortino_ratio([0.01, 0.02]),                      # no downside
    lambda: max_drawdown([]),
    lambda: quantile_monotonicity([1.0, 2.0]),
    lambda: probabilistic_sharpe_ratio(0.1, 0.0, 1),
    lambda: expected_max_sharpe(0, 0.01),
    lambda: deflated_sharpe_ratio([0.01, 0.02, 0.01], 5),     # too short
    lambda: capacity_curve(pd.Series([0.01, -0.01, 0.02]), 0.3,
                           [1e6], adv_dollars=-1.0, n_names=10),
    lambda: capacity_curve(pd.Series([0.01, -0.01, 0.02]), -0.3,
                           [1e6], adv_dollars=1e7, n_names=10),
])
def test_invalid_inputs_raise(bad_call):
    with pytest.raises(ValueError):
        bad_call()
