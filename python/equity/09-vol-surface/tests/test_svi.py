"""SVI parameterisation, analytic derivatives, Durrleman condition, fitting."""

from __future__ import annotations

import numpy as np
import pytest

from eq_surface.smile import (
    SVIParams,
    check_butterfly,
    durrleman_g,
    fit_quadratic_delta,
    fit_svi,
    svi_d2w_dk2,
    svi_dw_dk,
    svi_implied_vol,
    svi_total_variance,
)

#: Axel Vogt's classic butterfly-arbitrageable raw-SVI slice: w(k) > 0
#: everywhere yet the Durrleman g goes negative (Gatheral & Jacquier 2014).
VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)


def test_total_variance_hand_checked(good_svi):
    p = good_svi  # a=0.02, b=0.4, rho=-0.3, m=0.1, sigma=0.2
    # At k = m: w = a + b*sigma = 0.02 + 0.4*0.2 = 0.10
    assert svi_total_variance(0.1, p) == pytest.approx(0.10, abs=1e-14)
    # At k = m + 0.3: w = a + b*(rho*0.3 + sqrt(0.09 + 0.04))
    expected = 0.02 + 0.4 * (-0.3 * 0.3 + np.sqrt(0.09 + 0.04))
    assert svi_total_variance(0.4, p) == pytest.approx(expected, abs=1e-14)
    # At k = m - 0.4: w = a + b*(rho*(-0.4) + sqrt(0.16 + 0.04))
    expected = 0.02 + 0.4 * (0.3 * 0.4 + np.sqrt(0.16 + 0.04))
    assert svi_total_variance(-0.3, p) == pytest.approx(expected, abs=1e-14)


def test_analytic_first_derivative_matches_numerical(good_svi):
    k = np.linspace(-1.2, 1.2, 41)
    h = 1e-6
    num = (np.asarray(svi_total_variance(k + h, good_svi)) - np.asarray(svi_total_variance(k - h, good_svi))) / (2 * h)
    ana = np.asarray(svi_dw_dk(k, good_svi))
    assert np.max(np.abs(ana - num)) < 1e-8


def test_analytic_second_derivative_matches_numerical(good_svi):
    k = np.linspace(-1.2, 1.2, 41)
    w = lambda x: np.asarray(svi_total_variance(x, good_svi))

    def second(h):
        return (w(k + h) - 2 * w(k) + w(k - h)) / (h * h)

    h = 1e-3  # Richardson-extrapolated second difference: O(h^4) truncation
    num = (4.0 * second(h / 2) - second(h)) / 3.0
    ana = np.asarray(svi_d2w_dk2(k, good_svi))
    assert np.max(np.abs(ana - num)) < 1e-8


def test_wing_slopes_asymptotic(good_svi):
    """w'(k) -> b(rho +- 1) in the wings (raw-SVI linear wings)."""
    p = good_svi
    assert svi_dw_dk(200.0, p) == pytest.approx(p.b * (p.rho + 1.0), abs=1e-6)
    assert svi_dw_dk(-200.0, p) == pytest.approx(p.b * (p.rho - 1.0), abs=1e-6)


def test_durrleman_nonnegative_for_good_smile(good_svi):
    ok, min_g, viol = check_butterfly(good_svi, -2.0, 2.0, 801)
    assert ok
    assert min_g >= 0.0
    assert viol.size == 0


def test_planted_butterfly_arbitrage_is_flagged():
    """The Vogt slice has positive w everywhere but violates Durrleman."""
    w = np.asarray(svi_total_variance(np.linspace(-1.5, 1.5, 301), VOGT))
    assert np.all(w > 0.0)  # not a parameter-validity failure...
    ok, min_g, viol = check_butterfly(VOGT, -1.5, 1.5, 601)
    assert not ok  # ...but a genuine density violation
    assert min_g < 0.0
    assert viol.size > 0


def test_durrleman_g_scalar_and_vector_agree(good_svi):
    ks = np.array([-0.5, 0.0, 0.7])
    vec = np.asarray(durrleman_g(ks, good_svi))
    for i, k in enumerate(ks):
        assert durrleman_g(float(k), good_svi) == pytest.approx(vec[i], abs=1e-14)


def test_fit_recovers_known_params_from_exact_smile(good_svi):
    k = np.linspace(-0.8, 0.8, 25)
    w = np.asarray(svi_total_variance(k, good_svi))
    fit = fit_svi(k, w, T=0.5, n_restarts=8, seed=0)
    got = fit.params.as_array()
    want = good_svi.as_array()
    assert np.max(np.abs(got - want)) < 1e-4
    # Fitted curve reproduces the input essentially exactly.
    w_fit = np.asarray(svi_total_variance(k, fit.params))
    assert np.max(np.abs(w_fit - w)) < 1e-8
    assert fit.rmse_w < 1e-8
    assert fit.arb_free


def test_fit_is_deterministic_given_seed(good_svi):
    k = np.linspace(-0.6, 0.6, 15)
    w = np.asarray(svi_total_variance(k, good_svi)) * (1 + 0.01 * np.sin(9 * k))
    f1 = fit_svi(k, w, T=0.5, n_restarts=5, seed=7)
    f2 = fit_svi(k, w, T=0.5, n_restarts=5, seed=7)
    assert np.array_equal(f1.params.as_array(), f2.params.as_array())


def test_fit_drops_nan_quotes(good_svi):
    k = np.linspace(-0.8, 0.8, 20)
    w = np.asarray(svi_total_variance(k, good_svi))
    w[3] = np.nan
    fit = fit_svi(k, w, T=0.5, seed=0)
    assert fit.n_points == 19
    assert fit.rmse_w < 1e-7


def test_fit_too_few_points_rejected_informatively(good_svi):
    k = np.array([0.0])
    w = np.array([0.04])
    with pytest.raises(ValueError, match="at least 5"):
        fit_svi(k, w, T=0.5)
    with pytest.raises(ValueError, match="at least 5"):
        fit_svi(np.linspace(-1, 1, 4), np.full(4, 0.04), T=0.5)


def test_fit_nonpositive_variance_raises():
    with pytest.raises(ValueError, match="positive"):
        fit_svi(np.linspace(-1, 1, 6), np.array([0.04, 0.03, -0.01, 0.03, 0.04, 0.05]), T=0.5)


def test_invalid_svi_params_raise():
    with pytest.raises(ValueError, match="b must be"):
        SVIParams(a=0.02, b=-0.1, rho=0.0, m=0.0, sigma=0.1)
    with pytest.raises(ValueError, match="rho"):
        SVIParams(a=0.02, b=0.1, rho=1.0, m=0.0, sigma=0.1)
    with pytest.raises(ValueError, match="sigma"):
        SVIParams(a=0.02, b=0.1, rho=0.0, m=0.0, sigma=0.0)
    with pytest.raises(ValueError, match="non-positive total variance"):
        SVIParams(a=-0.5, b=0.1, rho=0.0, m=0.0, sigma=0.1)


def test_svi_implied_vol_requires_positive_T(good_svi):
    with pytest.raises(ValueError, match="positive"):
        svi_implied_vol(0.0, good_svi, T=0.0)
    iv = svi_implied_vol(0.1, good_svi, T=0.5)
    assert iv == pytest.approx(np.sqrt(0.10 / 0.5), abs=1e-12)


def test_quadratic_delta_baseline_fits_and_underperforms_svi(good_svi):
    """The naive baseline runs, but SVI beats it on an SVI-generated smile."""
    T = 0.5
    k = np.linspace(-0.5, 0.5, 21)
    vols = np.asarray(svi_implied_vol(k, good_svi, T))
    quad = fit_quadratic_delta(k, vols, T)
    assert np.isfinite(quad.rmse_vol)
    svi_fit = fit_svi(k, vols**2 * T, T, seed=0)
    assert svi_fit.rmse_vol < quad.rmse_vol
    # callable interface
    assert np.isfinite(quad(0.5))


def test_quadratic_delta_invalid_inputs():
    with pytest.raises(ValueError, match="at least 3"):
        fit_quadratic_delta(np.array([0.0, 0.1]), np.array([0.2, 0.21]), 0.5)
    with pytest.raises(ValueError, match="T must be positive"):
        fit_quadratic_delta(np.zeros(5), np.full(5, 0.2), 0.0)
