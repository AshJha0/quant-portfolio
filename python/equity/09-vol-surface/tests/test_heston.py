"""Heston characteristic function and Fourier pricing."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from eq_surface.black_scholes import bs_price
from eq_surface.heston import (
    FellerWarning,
    HestonParams,
    feller_condition,
    heston_call,
    heston_call_damped,
    heston_call_gl,
    heston_call_p1p2,
    heston_cf,
    heston_put,
)

S, R, Q = 100.0, 0.02, 0.01


def test_cf_at_zero_is_one(mild_heston, extreme_heston):
    for p in (mild_heston, extreme_heston):
        for T in [0.05, 0.5, 2.0]:
            assert heston_cf(0.0, T, S, R, Q, p) == pytest.approx(1.0 + 0.0j, abs=1e-12)


def test_cf_martingale_identity(mild_heston, extreme_heston):
    """phi(-i) = E[S_T] = forward: the discounted spot is a martingale."""
    for p in (mild_heston, extreme_heston):
        for T in [0.1, 1.0, 3.0]:
            F = S * np.exp((R - Q) * T)
            assert heston_cf(-1j, T, S, R, Q, p) == pytest.approx(F, rel=1e-10)


def test_cf_conjugate_symmetry(mild_heston):
    """phi(-u) = conj(phi(u)) for real u (real-valued density)."""
    for u in [0.3, 2.0, 17.5]:
        a = heston_cf(u, 0.7, S, R, Q, mild_heston)
        b = heston_cf(-u, 0.7, S, R, Q, mild_heston)
        assert b == pytest.approx(np.conj(a), abs=1e-12)


def test_little_trap_cf_continuous_in_u():
    """Long expiry + high vol-of-vol: the original formulation would branch-jump."""
    p = HestonParams(v0=0.09, kappa=1.0, theta=0.09, rho=-0.8, xi=1.0)
    u = np.linspace(0.01, 80.0, 32000)
    phi = np.asarray(heston_cf(u, 10.0, S, R, Q, p))
    # continuity: successive values move by O(du * |phi'|); a principal-branch
    # jump in the original formulation would show up as an O(1) discontinuity.
    jumps = np.abs(np.diff(phi))
    assert np.max(jumps) < 0.05
    assert np.all(np.isfinite(phi))


def test_degenerate_limit_matches_black_scholes():
    """xi -> 0 with v0 = theta collapses to BS with sigma = sqrt(v0)."""
    p = HestonParams(v0=0.04, kappa=1.5, theta=0.04, rho=0.0, xi=0.0)
    for T in [0.25, 1.0]:
        for K in [80.0, 100.0, 125.0]:
            bs = bs_price(S, K, T, R, Q, 0.2, "call")
            assert heston_call_p1p2(S, K, T, R, Q, p) == pytest.approx(bs, abs=1e-8)
            assert float(heston_call_gl(S, K, T, R, Q, p)) == pytest.approx(bs, abs=1e-8)


def test_xi_small_time_dependent_variance_matches_bs_total_variance():
    """xi ~ 0 with v0 != theta: BS with the integrated deterministic variance."""
    p = HestonParams(v0=0.09, kappa=2.0, theta=0.04, rho=0.0, xi=0.0)
    T = 1.0
    iv2 = p.theta * T + (p.v0 - p.theta) * (1 - np.exp(-p.kappa * T)) / p.kappa
    sigma_eff = np.sqrt(iv2 / T)
    bs = bs_price(S, 100.0, T, R, Q, sigma_eff, "call")
    assert heston_call_p1p2(S, 100.0, T, R, Q, p) == pytest.approx(bs, abs=1e-8)


def test_p1p2_vs_damped_agree(mild_heston, extreme_heston):
    """Two independent Fourier routes agree to 1e-6 across a strike/expiry grid."""
    for p in (mild_heston, extreme_heston):
        for T in [1.0 / 12.0, 0.5, 2.0]:
            for K in [70.0, 100.0, 140.0]:
                a = heston_call_p1p2(S, K, T, R, Q, p)
                b = heston_call_damped(S, K, T, R, Q, p)
                assert a == pytest.approx(b, abs=1e-6), (p, T, K)


def test_gl_vs_damped_agree_wide_grid(mild_heston, extreme_heston):
    """Fast Gauss-Legendre path validated against adaptive quadrature."""
    Ks = np.array([55.0, 70.0, 85.0, 100.0, 115.0, 135.0, 160.0])
    for p in (mild_heston, extreme_heston):
        for T in [1.0 / 52.0, 0.25, 1.0]:
            gl = np.asarray(heston_call_gl(S, Ks, T, R, Q, p))
            for j, K in enumerate(Ks):
                b = heston_call_damped(S, float(K), T, R, Q, p)
                assert gl[j] == pytest.approx(b, abs=1e-6), (p, T, K)


def test_put_call_parity(mild_heston):
    for T in [0.1, 1.0]:
        for K in [80.0, 100.0, 120.0]:
            c = heston_call(S, K, T, R, Q, mild_heston)
            p_ = heston_put(S, K, T, R, Q, mild_heston)
            parity = S * np.exp(-Q * T) - K * np.exp(-R * T)
            assert c - p_ == pytest.approx(parity, abs=1e-8)


def test_price_monotone_in_v0(mild_heston):
    prices = []
    for v0 in [0.01, 0.02, 0.04, 0.08, 0.16]:
        p = HestonParams(v0, mild_heston.kappa, mild_heston.theta, mild_heston.rho, mild_heston.xi)
        prices.append(float(heston_call_gl(S, 100.0, 0.5, R, Q, p)))
    assert np.all(np.diff(prices) > 0.0)


def test_price_monotone_and_convex_in_strike(mild_heston):
    Ks = np.linspace(60.0, 150.0, 31)
    prices = np.asarray(heston_call_gl(S, Ks, 0.5, R, Q, mild_heston))
    assert np.all(np.diff(prices) < 0.0)  # decreasing in K
    assert np.all(np.diff(prices, 2) > -1e-10)  # convex in K


def test_deep_itm_limit_is_discounted_intrinsic(mild_heston):
    T = 0.25
    lower = S * np.exp(-Q * T) - 10.0 * np.exp(-R * T)
    price = heston_call_p1p2(S, 10.0, T, R, Q, mild_heston)
    assert price == pytest.approx(lower, abs=1e-6)


def test_deep_otm_limit_is_zero(mild_heston):
    price = heston_call_p1p2(S, 1000.0, 0.1, R, Q, mild_heston)
    assert 0.0 <= price < 1e-8
    price_gl = float(heston_call_gl(S, 1000.0, 0.1, R, Q, mild_heston))
    assert 0.0 <= price_gl < 1e-8


def test_t_zero_returns_intrinsic(mild_heston):
    assert heston_call_p1p2(S, 90.0, 0.0, R, Q, mild_heston) == 10.0
    assert float(heston_call_gl(S, 110.0, 0.0, R, Q, mild_heston)) == 0.0


def test_rho_boundaries_handled():
    """|rho| = 1 exactly: methods stay finite and mutually consistent."""
    for rho in (-1.0, 1.0):
        p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, rho=rho, xi=0.5)
        a = heston_call_p1p2(S, 100.0, 0.5, R, Q, p)
        b = float(heston_call_gl(S, 100.0, 0.5, R, Q, p))
        assert np.isfinite(a) and np.isfinite(b)
        assert a == pytest.approx(b, abs=1e-6)
        assert 0.0 < a < S


def test_feller_condition_checker():
    ok = HestonParams(v0=0.04, kappa=2.0, theta=0.04, rho=-0.5, xi=0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # must NOT warn
        ratio = feller_condition(ok)
    assert ratio == pytest.approx(2 * 2.0 * 0.04 / 0.09)
    bad = HestonParams(v0=0.04, kappa=1.0, theta=0.04, rho=-0.5, xi=1.0)
    with pytest.warns(FellerWarning, match="Feller"):
        ratio = feller_condition(bad)
    assert ratio == pytest.approx(0.08)
    zero_xi = HestonParams(v0=0.04, kappa=1.0, theta=0.04, rho=0.0, xi=0.0)
    assert feller_condition(zero_xi) == np.inf


def test_invalid_params_raise():
    with pytest.raises(ValueError, match="v0"):
        HestonParams(v0=0.0, kappa=1.0, theta=0.04, rho=0.0, xi=0.5)
    with pytest.raises(ValueError, match="v0"):
        HestonParams(v0=-0.04, kappa=1.0, theta=0.04, rho=0.0, xi=0.5)
    with pytest.raises(ValueError, match="kappa"):
        HestonParams(v0=0.04, kappa=-1.0, theta=0.04, rho=0.0, xi=0.5)
    with pytest.raises(ValueError, match="theta"):
        HestonParams(v0=0.04, kappa=1.0, theta=-0.04, rho=0.0, xi=0.5)
    with pytest.raises(ValueError, match="rho"):
        HestonParams(v0=0.04, kappa=1.0, theta=0.04, rho=-1.5, xi=0.5)
    with pytest.raises(ValueError, match="xi"):
        HestonParams(v0=0.04, kappa=1.0, theta=0.04, rho=0.0, xi=-0.1)


def test_invalid_pricing_inputs_raise(mild_heston):
    with pytest.raises(ValueError, match="spot"):
        heston_call_p1p2(-1.0, 100.0, 1.0, R, Q, mild_heston)
    with pytest.raises(ValueError, match="strike"):
        heston_call_gl(S, -100.0, 1.0, R, Q, mild_heston)
    with pytest.raises(ValueError, match="non-negative"):
        heston_call_damped(S, 100.0, -1.0, R, Q, mild_heston)
    with pytest.raises(ValueError, match="alpha"):
        heston_call_damped(S, 100.0, 1.0, R, Q, mild_heston, alpha=0.0)
    with pytest.raises(ValueError, match="method"):
        heston_call(S, 100.0, 1.0, R, Q, mild_heston, method="fft2")
