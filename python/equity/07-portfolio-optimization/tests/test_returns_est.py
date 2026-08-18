"""Tests for eq_port.returns_est: sample/EWMA means, James-Stein
shrinkage, reverse optimization and Black-Litterman."""

import numpy as np
import pandas as pd
import pytest

from eq_port.covariance import psd_repair
from eq_port.mvo import tangency_weights
from eq_port.returns_est import (
    black_litterman,
    ewma_mean,
    implied_equilibrium_returns,
    james_stein_mean,
    sample_mean,
)

RNG = np.random.default_rng(123)


@pytest.fixture()
def panel() -> np.ndarray:
    return RNG.normal(0.0004, 0.01, size=(500, 6))


@pytest.fixture()
def spd_cov() -> np.ndarray:
    a = RNG.normal(size=(6, 6))
    return psd_repair(a @ a.T / 6 + np.eye(6), eps=1e-8)


# ---------------------------------------------------------------- sample/EWMA

def test_sample_mean_matches_numpy(panel):
    np.testing.assert_allclose(sample_mean(panel), panel.mean(axis=0), rtol=0, atol=0)


def test_sample_mean_dataframe(panel):
    df = pd.DataFrame(panel)
    np.testing.assert_allclose(sample_mean(df), panel.mean(axis=0))


def test_ewma_mean_constant_series_is_exact():
    x = np.full((100, 3), 0.002)
    np.testing.assert_allclose(ewma_mean(x, halflife=20), [0.002] * 3, atol=1e-15)


def test_ewma_mean_tilts_to_recent():
    x = np.linspace(0.0, 0.01, 200)[:, None]  # increasing returns
    assert ewma_mean(x, halflife=10)[0] > sample_mean(x)[0]


def test_ewma_mean_matches_manual_weights():
    x = RNG.normal(size=(50, 2))
    lam = 0.5 ** (1.0 / 15.0)
    w = lam ** np.arange(49, -1, -1.0)
    w /= w.sum()
    np.testing.assert_allclose(ewma_mean(x, halflife=15), w @ x, atol=1e-14)


def test_ewma_mean_invalid_halflife_raises():
    with pytest.raises(ValueError, match="halflife"):
        ewma_mean(np.zeros((10, 2)), halflife=0.0)


# ---------------------------------------------------------------- James-Stein

def test_js_is_convex_combination(panel):
    res = james_stein_mean(panel)
    mu = panel.mean(axis=0)
    expected = res.intensity * res.grand_mean + (1 - res.intensity) * mu
    np.testing.assert_allclose(res.mean, expected, atol=1e-15)


def test_js_intensity_in_unit_interval():
    for seed in range(5):
        x = np.random.default_rng(seed).normal(0.001, 0.02, size=(60, 8))
        phi = james_stein_mean(x).intensity
        assert 0.0 <= phi <= 1.0


def test_js_shrinks_toward_grand_mean(panel):
    res = james_stein_mean(panel)
    mu = panel.mean(axis=0)
    # every element moves toward the grand mean, never past it
    assert np.all(np.abs(res.mean - res.grand_mean) <= np.abs(mu - res.grand_mean) + 1e-15)


def test_js_full_shrink_when_no_dispersion():
    x = np.tile(RNG.normal(0, 0.01, size=(200, 1)), (1, 4))  # identical columns
    res = james_stein_mean(x)
    assert res.intensity == 1.0
    np.testing.assert_allclose(res.mean, res.grand_mean)


def test_js_single_observation_returns_grand_mean():
    res = james_stein_mean(np.array([[0.01, 0.03, 0.05]]))
    assert res.intensity == 1.0
    np.testing.assert_allclose(res.mean, 0.03)


def test_js_shrinks_harder_with_shorter_window():
    x = RNG.normal(0.0005, 0.015, size=(2000, 8))
    phi_long = james_stein_mean(x).intensity
    phi_short = james_stein_mean(x[:60]).intensity
    assert phi_short > phi_long


# ------------------------------------------------------- reverse optimization

def test_reverse_optimization_round_trip_identity(spd_cov):
    w_mkt = np.array([0.3, 0.25, 0.15, 0.12, 0.1, 0.08])
    pi = implied_equilibrium_returns(spd_cov, w_mkt, risk_aversion=3.0)
    w_back = tangency_weights(pi, spd_cov, rf=0.0)
    np.testing.assert_allclose(w_back, w_mkt, atol=1e-10)


def test_implied_returns_scale_with_risk_aversion(spd_cov):
    w = np.full(6, 1 / 6)
    pi1 = implied_equilibrium_returns(spd_cov, w, risk_aversion=1.0)
    pi4 = implied_equilibrium_returns(spd_cov, w, risk_aversion=4.0)
    np.testing.assert_allclose(pi4, 4.0 * pi1, rtol=1e-14)


def test_implied_returns_validates_inputs(spd_cov):
    with pytest.raises(ValueError, match="mismatch"):
        implied_equilibrium_returns(spd_cov, np.ones(4))
    with pytest.raises(ValueError, match="risk_aversion"):
        implied_equilibrium_returns(spd_cov, np.full(6, 1 / 6), risk_aversion=-1.0)


# ------------------------------------------------------------ Black-Litterman

def test_bl_no_views_posterior_equals_prior(spd_cov):
    pi = implied_equilibrium_returns(spd_cov, np.full(6, 1 / 6))
    res = black_litterman(pi, spd_cov, tau=0.05)
    np.testing.assert_allclose(res.mean, pi, atol=0)  # exact
    np.testing.assert_allclose(res.cov, spd_cov * 1.05, atol=1e-15)
    np.testing.assert_allclose(res.mean_uncertainty, 0.05 * spd_cov, atol=1e-15)


def test_bl_infinitely_confident_view_holds_exactly(spd_cov):
    pi = implied_equilibrium_returns(spd_cov, np.full(6, 1 / 6))
    p = np.zeros((1, 6))
    p[0, 0], p[0, 1] = 1.0, -1.0  # relative view: asset0 beats asset1
    q = np.array([0.002])
    res = black_litterman(pi, spd_cov, p, q, tau=0.05, omega=np.zeros((1, 1)))
    np.testing.assert_allclose(p @ res.mean, q, atol=1e-14)


def test_bl_posterior_between_prior_and_view(spd_cov):
    pi = implied_equilibrium_returns(spd_cov, np.full(6, 1 / 6))
    p = np.zeros((1, 6))
    p[0, 2] = 1.0  # absolute view on asset 2
    q = np.array([pi[2] + 0.005])
    res = black_litterman(pi, spd_cov, p, q, tau=0.05)
    view_val = float((p @ res.mean)[0])
    assert pi[2] < view_val < q[0]


def test_bl_higher_confidence_moves_closer_to_view(spd_cov):
    pi = implied_equilibrium_returns(spd_cov, np.full(6, 1 / 6))
    p = np.zeros((1, 6))
    p[0, 0] = 1.0
    q = np.array([pi[0] + 0.004])
    loose = black_litterman(pi, spd_cov, p, q, omega=np.array([[1e-4]]))
    tight = black_litterman(pi, spd_cov, p, q, omega=np.array([[1e-8]]))
    assert abs(float((p @ tight.mean)[0]) - q[0]) < abs(float((p @ loose.mean)[0]) - q[0])


def test_bl_posterior_cov_exceeds_prior_cov(spd_cov):
    pi = implied_equilibrium_returns(spd_cov, np.full(6, 1 / 6))
    p = np.zeros((1, 6))
    p[0, 0] = 1.0
    res = black_litterman(pi, spd_cov, p, [0.001], tau=0.05)
    # Sigma + M with M PSD: eigenvalues of (posterior - prior) >= 0
    diff = res.cov - spd_cov
    assert np.linalg.eigvalsh(diff).min() >= -1e-12


def test_bl_validates_inputs(spd_cov):
    pi = np.zeros(6)
    with pytest.raises(ValueError, match="view_returns"):
        black_litterman(pi, spd_cov, np.eye(6)[:1])
    with pytest.raises(ValueError, match="columns"):
        black_litterman(pi, spd_cov, np.ones((1, 4)), [0.001])
    with pytest.raises(ValueError, match="omega"):
        black_litterman(pi, spd_cov, np.eye(6)[:2], [0.001, 0.002], omega=np.eye(3))
    with pytest.raises(ValueError, match="tau"):
        black_litterman(pi, spd_cov, tau=0.0)
    with pytest.raises(ValueError, match="mismatch"):
        black_litterman(np.zeros(4), spd_cov)
