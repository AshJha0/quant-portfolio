"""Maximum-Sharpe as a convex QP: sign handling, global optimality, KKT.

The reference in ``_brute_force_long_only`` is independent of the solver:
for long-only weights with a non-binding upper bound, the optimum is the
closed-form tangency portfolio of some support set F with all weights in
F positive. Enumerating every support set and keeping the best feasible
one gives the exact global maximum for small N.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from eq_port import (
    make_tangency_strategy,
    max_sharpe_constrained,
    max_sharpe_qp,
    min_variance_constrained,
    portfolio_vol,
    tangency_weights,
)
from hypothesis import given, settings
from hypothesis import strategies as st


def _sharpe(w, mu, cov, rf=0.0):
    return float((w @ mu - rf) / np.sqrt(w @ cov @ w))


def _brute_force_long_only(mu, cov):
    n = len(mu)
    best, best_w = -np.inf, None
    for k in range(1, n + 1):
        for f in itertools.combinations(range(n), k):
            f = list(f)
            x = np.linalg.solve(cov[np.ix_(f, f)], mu[f])
            if x.sum() <= 0:
                continue
            wf = x / x.sum()
            if np.any(wf < 0):
                continue
            w = np.zeros(n)
            w[f] = wf
            s = _sharpe(w, mu, cov)
            if s > best:
                best, best_w = s, w
    return best, best_w


def _random_problem(seed, n):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    cov = (a @ a.T / n + np.diag(rng.uniform(0.05, 0.5, n))) * 1e-4
    mu = rng.normal(2e-4, 4e-4, n)
    return mu, cov


def test_all_negative_means_returns_min_variance_not_max_vol():
    # Regression for the volatility-maximising bug: with every mean below
    # rf, the old ratio objective put 100% in the most volatile asset.
    cov = np.diag([0.01, 0.04, 0.09]) * 1e-2
    mu = np.array([-0.01, -0.01, -0.01])
    res = max_sharpe_qp(mu, cov, bounds=(0.0, 1.0))
    w_mv = min_variance_constrained(cov, bounds=(0.0, 1.0))
    assert res.fallback
    np.testing.assert_allclose(res.weights, w_mv, atol=1e-9)
    assert portfolio_vol(res.weights, cov) <= portfolio_vol(w_mv, cov) + 1e-12
    assert res.weights[2] < 0.5  # the most volatile asset is not the whole book


def test_means_equal_rf_is_also_a_fallback():
    cov = np.diag([1.0, 2.0]) * 1e-4
    res = max_sharpe_qp(np.array([0.001, 0.001]), cov, rf=0.001)
    assert res.fallback


def test_wrapper_returns_fallback_weights():
    cov = np.diag([1.0, 4.0, 9.0]) * 1e-4
    w = max_sharpe_constrained(np.array([-1e-3, -2e-3, -5e-4]), cov)
    np.testing.assert_allclose(w, min_variance_constrained(cov), atol=1e-9)


def test_inactive_bounds_match_closed_form_tangency():
    mu, cov = _random_problem(3, 6)
    mu = np.abs(mu) + 1e-4
    w_cf = tangency_weights(mu, cov)
    res = max_sharpe_qp(mu, cov, bounds=(-10.0, 10.0))
    assert not res.fallback
    np.testing.assert_allclose(res.weights, w_cf, atol=1e-7)


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10_000), n=st.integers(2, 6))
def test_long_only_matches_support_enumeration(seed, n):
    mu, cov = _random_problem(seed, n)
    res = max_sharpe_qp(mu, cov, bounds=(0.0, 1.0))
    if mu.max() <= 0:
        assert res.fallback
        return
    best, _ = _brute_force_long_only(mu, cov)
    assert not res.fallback
    assert res.weights.min() >= 0.0
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-12)
    # relative Sharpe gap: SLSQP stops at ftol 1e-15 on an O(1) objective,
    # so the optimum is reached to ~1e-7 relative in the ratio
    assert res.sharpe == pytest.approx(best, rel=1e-7)
    assert res.kkt_residual < 1e-6


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), cap=st.floats(0.3, 0.6))
def test_box_cap_kkt_and_dominates_random_feasible_points(seed, cap):
    n = 5
    mu, cov = _random_problem(seed, n)
    mu = mu + 3e-4  # at least one positive excess most of the time
    res = max_sharpe_qp(mu, cov, bounds=(0.0, cap))
    if res.fallback:
        assert mu.max() <= 0
        return
    w = res.weights
    assert w.max() <= cap + 1e-12 and w.min() >= -1e-12
    assert res.kkt_residual < 1e-6
    rng = np.random.default_rng(seed)
    for _ in range(300):
        v = rng.dirichlet(np.ones(n))
        if v.max() > cap:
            continue
        assert _sharpe(v, mu, cov) <= res.sharpe + 1e-9


def test_rejects_nonfinite_rf():
    with pytest.raises(ValueError):
        max_sharpe_qp([0.1, 0.2], np.eye(2), rf=np.nan)


def test_tangency_strategy_logs_fallbacks():
    rng = np.random.default_rng(0)
    neg = pd.DataFrame(rng.normal(-0.002, 0.01, size=(300, 4)))
    pos = pd.DataFrame(rng.normal(0.002, 0.01, size=(300, 4)) + np.arange(4) * 1e-3)
    strat = make_tangency_strategy(shrink_mean=False)
    strat(neg)
    strat(pos)
    assert strat.stats == {"calls": 2, "fallbacks": 1}
