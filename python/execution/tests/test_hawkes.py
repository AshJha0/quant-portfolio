"""Hawkes MLE recovers known parameters; time rescaling separates models."""

from __future__ import annotations

import numpy as np
from qp_exec.hawkes import (
    compensator_increments_multivariate,
    compensator_increments_univariate,
    fit_multivariate,
    fit_univariate,
    ks_time_rescaling,
    loglik_multivariate,
    loglik_univariate,
    simulate_multivariate,
    simulate_univariate,
)


def test_loglik_matches_brute_force() -> None:
    t = np.array([0.3, 0.5, 1.7, 1.8, 3.1])
    mu, a, b, T = 0.4, 0.5, 1.5, 4.0
    lam = [mu + sum(a * b * np.exp(-b * (ti - tj)) for tj in t if tj < ti) for ti in t]
    comp = mu * T + a * sum(1 - np.exp(-b * (T - tj)) for tj in t)
    want = float(np.sum(np.log(lam)) - comp)
    assert abs(loglik_univariate(t, T, mu, a, b) - want) < 1e-12
    got = loglik_multivariate([t], T, np.array([mu]), np.array([[a]]), b)
    assert abs(got - want) < 1e-12


def test_univariate_mle_recovers_parameters_within_3_se() -> None:
    T = 20000.0
    t = simulate_univariate(0.4, 0.7, 3.0, T, seed=21)
    f = fit_univariate(t, T)
    for est, true, se in zip((f.mu, f.alpha, f.beta), (0.4, 0.7, 3.0), f.stderr, strict=True):
        assert abs(est - true) < 3 * se + 1e-9, (est, true, se)


def test_time_rescaling_accepts_true_model_rejects_poisson() -> None:
    T = 10000.0
    t = simulate_univariate(0.5, 0.6, 2.0, T, seed=5)
    ok = ks_time_rescaling(compensator_increments_univariate(t, 0.5, 0.6, 2.0))
    assert ok.pvalue > 0.05
    rate = t.size / T  # best-fitting Poisson
    bad = ks_time_rescaling(compensator_increments_univariate(t, rate, 0.0, 2.0))
    assert bad.pvalue < 1e-6


def test_multivariate_mle_recovers_cross_excitation() -> None:
    mu = np.array([0.3, 0.2])
    al = np.array([[0.3, 0.4], [0.05, 0.5]])
    T = 8000.0
    ts = simulate_multivariate(mu, al, 2.0, T, seed=8)
    f = fit_multivariate(ts, T, beta0=1.5)
    assert np.max(np.abs(f.alpha - al)) < 0.08
    assert np.max(np.abs(f.mu - mu)) < 0.06
    assert abs(f.beta - 2.0) < 0.3
    inc = compensator_increments_multivariate(ts, f.mu, f.alpha, f.beta)
    assert all(ks_time_rescaling(x).pvalue > 0.01 for x in inc)


def test_nonstationary_alpha_rejected() -> None:
    import pytest
    with pytest.raises(ValueError):
        simulate_multivariate(np.array([1.0]), np.array([[1.0]]), 1.0, 10.0, seed=1)
