"""Tests for eq_pairs.kalman: exact likelihood oracle, the RLS special case,
and recovery of planted noise variances."""

import numpy as np
import pytest
from eq_pairs.kalman import kalman_hedge_ratio, kalman_loglik
from scipy.stats import multivariate_normal


def _simulate(n, r, q_beta, seed=0, beta0=1.5):
    rng = np.random.default_rng(seed)
    x = 50 + np.cumsum(rng.normal(0, 0.5, n))
    beta = beta0 + np.cumsum(rng.normal(0, np.sqrt(q_beta), n))
    y = beta * x + rng.normal(0, np.sqrt(r), n)
    return y, x, beta


def test_loglik_equals_exact_multivariate_normal_density():
    rng = np.random.default_rng(3)
    n, p0, r = 12, 0.7, 0.3
    q = np.array([0.05, 0.02])
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    H = np.column_stack([np.ones(n), x])
    t = np.arange(1, n + 1)
    cov = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = H[i] @ (p0 * np.eye(2) + min(t[i], t[j]) * np.diag(q)) @ H[j]
    cov += r * np.eye(n)
    exact = multivariate_normal(np.zeros(n), cov).logpdf(y)
    ours = kalman_loglik(y, x, r, q, intercept=True, prior_var=p0, burn=0)
    assert ours == pytest.approx(exact, rel=1e-10)


def test_zero_state_noise_is_expanding_ridge_ols():
    """With Q = 0 the filtered state is the posterior mean of a fixed
    coefficient under the prior N(0, P0 I): ridge regression with penalty
    R / P0 on the data so far. As P0 grows this is expanding OLS (the RLS
    special case with no forgetting)."""
    y, x, _ = _simulate(200, 1.0, 0.0, seed=1)
    p0 = 1e4
    k = kalman_hedge_ratio(y, x, obs_var=1.0, state_var=[0.0, 0.0], prior_var=p0)
    H = np.column_stack([np.ones(200), x])
    for t in (20, 100, 199):
        A = H[: t + 1].T @ H[: t + 1] + (1.0 / p0) * np.eye(2)
        coef = np.linalg.solve(A, H[: t + 1].T @ y[: t + 1])
        assert k.beta.iloc[t] == pytest.approx(coef[1], rel=1e-8)
        assert k.alpha.iloc[t] == pytest.approx(coef[0], rel=1e-6)


def test_mle_recovers_planted_variances_and_tracks_beta():
    r_true, q_true = 4.0, 1e-4
    y, x, beta = _simulate(3000, r_true, q_true, seed=7)
    k = kalman_hedge_ratio(y, x, intercept=False)
    assert k.estimated
    # tolerances: about 3 asymptotic SEs at n = 3000 (R: SE ~ R sqrt(2/n) ~ 0.1)
    assert k.obs_var == pytest.approx(r_true, rel=0.15)
    assert k.state_var[0] == pytest.approx(q_true, rel=0.6)
    # tracks the true path as well as the filter run at the true variances
    oracle = kalman_hedge_ratio(y, x, obs_var=r_true, state_var=[q_true], intercept=False)
    err = np.median(np.abs(k.beta.to_numpy()[500:] - beta[500:]))
    err_oracle = np.median(np.abs(oracle.beta.to_numpy()[500:] - beta[500:]))
    assert err <= 1.1 * err_oracle
    fixed = kalman_loglik(y, x, r_true, [q_true], intercept=False)
    assert k.loglik >= fixed - 1e-6  # the MLE is at least as likely as the truth


def test_standardised_errors_are_unit_variance_at_the_truth():
    y, x, _ = _simulate(4000, 1.0, 1e-5, seed=4)
    k = kalman_hedge_ratio(y, x, obs_var=1.0, state_var=[1e-5], intercept=False)
    z = (k.error / np.sqrt(k.error_var)).to_numpy()[10:]
    assert z.var() == pytest.approx(1.0, abs=0.07)  # SE of a variance ~ sqrt(2/n) = 0.022


def test_input_validation():
    with pytest.raises(ValueError):
        kalman_hedge_ratio(np.ones(5), np.ones(5))
    y, x, _ = _simulate(100, 1.0, 1e-5)
    with pytest.raises(ValueError):
        kalman_hedge_ratio(y, x, obs_var=1.0)
    with pytest.raises(ValueError):
        kalman_hedge_ratio(y, x, obs_var=-1.0, state_var=[0.0, 0.0])
