"""Shared fixtures: parameter sets and (expensive) session-scoped results."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import eq_surface as es

# Market conventions used across the suite.
S0, R, Q = 100.0, 0.02, 0.01


@pytest.fixture(scope="session")
def mild_heston() -> es.HestonParams:
    """Feller-satisfying set (2*kappa*theta = 0.16 > xi^2 = 0.09)."""
    return es.HestonParams(v0=0.04, kappa=2.0, theta=0.04, rho=-0.5, xi=0.3)


@pytest.fixture(scope="session")
def extreme_heston() -> es.HestonParams:
    """Feller-violating, high vol-of-vol set (stress case for schemes)."""
    return es.HestonParams(v0=0.04, kappa=1.0, theta=0.04, rho=-0.9, xi=1.0)


@pytest.fixture(scope="session")
def good_svi() -> es.SVIParams:
    """A known butterfly-arbitrage-free SVI slice."""
    return es.SVIParams(a=0.02, b=0.4, rho=-0.3, m=0.1, sigma=0.2)


@pytest.fixture(scope="session")
def calib_market():
    """Small clean quote set generated from the DEFAULT_TRUE_HESTON truth."""
    from eq_surface.data import DEFAULT_TRUE_HESTON

    expiries = np.array([0.25, 0.5, 1.0])
    strikes = [S0 * np.exp(np.linspace(-0.35, 0.35, 9) * np.sqrt(T / 0.5)) for T in expiries]
    ivs = es.heston_model_ivs(S0, R, Q, expiries, strikes, DEFAULT_TRUE_HESTON)
    return expiries, strikes, ivs, DEFAULT_TRUE_HESTON


@pytest.fixture(scope="session")
def clean_calibration(calib_market):
    """One shared clean-data calibration (expensive: reused by many tests)."""
    expiries, strikes, ivs, true = calib_market
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = es.calibrate_heston(S0, R, Q, expiries, strikes, ivs, n_starts=2, seed=0)
    return res, true, caught


@pytest.fixture(scope="session")
def noisy_calibration(calib_market):
    """Calibration to vols perturbed by seeded N(0, 0.3 vol points) noise."""
    expiries, strikes, ivs, true = calib_market
    rng = np.random.default_rng(123)
    noisy = [iv + 0.003 * rng.standard_normal(iv.size) for iv in ivs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = es.calibrate_heston(S0, R, Q, expiries, strikes, noisy, n_starts=2, seed=0)
    return res, true
