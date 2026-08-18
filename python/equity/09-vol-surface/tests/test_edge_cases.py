"""Cross-cutting edge cases (documentation contract item 6).

Each case here is also described in docs/METHODOLOGY.md (assumptions register)
or docs/VALIDATION.md (failure modes).
"""

from __future__ import annotations

import numpy as np
import pytest

import eq_surface as es
from eq_surface.smile import SVIParams

S, R, Q = 100.0, 0.02, 0.01


def test_single_expiry_surface_end_to_end():
    """A one-pillar surface is legal and follows the extrapolation policy."""
    p = SVIParams(a=0.015, b=0.08, rho=-0.4, m=0.0, sigma=0.2)
    surf = es.VolSurface(np.array([0.5]), [p], S, R, Q)
    v_short = surf.vol_k(0.0, 0.1)
    v_pillar = surf.vol_k(0.0, 0.5)
    v_long = surf.vol_k(0.0, 3.0)
    assert v_short == pytest.approx(v_pillar, abs=1e-12)
    assert v_long == pytest.approx(v_pillar, abs=1e-12)


def test_single_strike_expiry_rejected_informatively():
    """One quote cannot pin down a 5-parameter smile: clear error, no garbage."""
    with pytest.raises(ValueError, match="at least 5 valid quotes"):
        es.fit_svi(np.array([0.0]), np.array([0.02]), T=0.25)


def test_t_to_zero_chain_implied_vol_round_trip():
    """T -> 0: the solver still inverts what little time value remains."""
    T = 1e-4
    price = es.bs_price(S, 100.0, T, R, Q, 0.2)
    assert price > 0.0
    assert es.implied_vol(price, S, 100.0, T, R, Q) == pytest.approx(0.2, abs=1e-8)


def test_t_to_zero_heston_price_converges_to_intrinsic(mild_heston):
    T = 1e-4
    itm = float(es.heston_call_gl(S, 90.0, T, R, Q, mild_heston))
    otm = float(es.heston_call_gl(S, 110.0, T, R, Q, mild_heston))
    assert itm == pytest.approx(10.0, abs=0.01)
    assert otm < 1e-6


def test_negative_variance_inputs_raise():
    with pytest.raises(ValueError):
        es.HestonParams(v0=-0.01, kappa=1.0, theta=0.04, rho=0.0, xi=0.3)
    with pytest.raises(ValueError):
        es.HestonParams(v0=0.04, kappa=1.0, theta=-0.04, rho=0.0, xi=0.3)
    with pytest.raises(ValueError, match="positive"):
        es.fit_svi(np.linspace(-0.5, 0.5, 6), np.array([0.02, 0.02, 0.0, 0.02, 0.02, 0.02]), T=0.5)
    with pytest.raises(ValueError, match="volatility"):
        es.bs_price(S, 100.0, 1.0, R, Q, -0.2)


def test_xi_zero_consistency_fourier_vs_mc():
    """xi = 0 collapses Heston to deterministic variance in every component."""
    p = es.HestonParams(v0=0.05, kappa=1.5, theta=0.03, rho=0.0, xi=0.0)
    fourier = es.heston_call(S, 100.0, 1.0, R, Q, p, method="damped")
    iv2 = p.theta + (p.v0 - p.theta) * (1 - np.exp(-p.kappa)) / p.kappa
    bs = es.bs_price(S, 100.0, 1.0, R, Q, np.sqrt(iv2))
    assert fourier == pytest.approx(bs, abs=1e-8)
    mc = es.heston_mc_price(S, 100.0, 1.0, R, Q, p, n_paths=100_000, n_steps=16,
                            scheme="euler_ft", seed=2)
    assert abs(mc.price - fourier) < 3.0 * mc.stderr


def test_rho_boundary_end_to_end():
    """rho = +-1: pricing, MC and Greeks all remain finite and consistent."""
    for rho in (-1.0, 1.0):
        p = es.HestonParams(v0=0.04, kappa=2.0, theta=0.04, rho=rho, xi=0.4)
        price = float(es.heston_call_gl(S, 105.0, 0.5, R, Q, p))
        assert 0.0 < price < S
        g = es.heston_greeks(S, 105.0, 0.5, R, Q, p, richardson=True)
        assert np.isfinite(g.delta) and g.gamma > 0.0


def test_deep_wing_iv_is_nan_not_garbage(mild_heston):
    """Model price of a hopeless wing quote inverts to nan, never to a number."""
    price = float(es.heston_call_gl(S, 500.0, 0.05, R, Q, mild_heston))
    ivs = es.implied_vol_vector(np.array([price]), S, np.array([500.0]), 0.05, R, Q)
    assert np.isnan(ivs[0])


def test_negative_rates_supported_throughout(mild_heston):
    r_neg = -0.01
    c = es.heston_call_p1p2(S, 100.0, 1.0, r_neg, Q, mild_heston)
    g = float(es.heston_call_gl(S, 100.0, 1.0, r_neg, Q, mild_heston))
    assert c == pytest.approx(g, abs=1e-6)
    iv = es.implied_vol(c, S, 100.0, 1.0, r_neg, Q)
    assert 0.1 < iv < 0.4


def test_zero_dividend_and_high_dividend():
    p = es.HestonParams(v0=0.04, kappa=2.0, theta=0.04, rho=-0.5, xi=0.3)
    c0 = float(es.heston_call_gl(S, 100.0, 1.0, R, 0.0, p))
    c5 = float(es.heston_call_gl(S, 100.0, 1.0, R, 0.05, p))
    assert c0 > c5  # dividends drag the forward down


def test_calibration_single_expiry_smile_runs():
    """Calibrating to one expiry is allowed (under-identified but must not crash)."""
    import warnings

    true = es.HestonParams(v0=0.04, kappa=2.0, theta=0.05, rho=-0.6, xi=0.4)
    T = np.array([0.5])
    K = [np.linspace(80.0, 120.0, 9)]
    ivs = es.heston_model_ivs(S, R, Q, T, K, true)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = es.calibrate_heston(S, R, Q, T, K, ivs, n_starts=1, seed=0)
    assert res.rmse_vol_points < 0.5
    assert res.condition_number > 1e2  # severely under-identified
