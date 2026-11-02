"""Round 3 oracle tests: SSVI/eSSVI, Dupire local vol, Bates."""

import numpy as np
import pytest
from eq_surface.bates import JumpParams, bates_call, bates_put, merton_call
from eq_surface.black_scholes import bs_price
from eq_surface.dupire import (
    LocalVolArbitrageError,
    local_vol_from_prices,
    local_vol_from_total_variance,
    local_vol_mc_price,
)
from eq_surface.heston import HestonParams, heston_call
from eq_surface.smile import durrleman_g, svi_total_variance
from eq_surface.ssvi import (
    ESSVIParams,
    SSVIParams,
    calendar_arbitrage_free,
    fit_ssvi,
    ssvi_slice_as_svi,
    ssvi_total_variance,
)

K_GRID = np.linspace(-1.5, 1.5, 61)


# ----------------------------------------------------------------- SSVI
def test_ssvi_slice_equals_raw_svi():
    """Gatheral-Jacquier 2014 Section 4: an SSVI slice is a raw SVI slice."""
    theta, rho, phi = 0.04, -0.6, 2.5
    w_ssvi = ssvi_total_variance(K_GRID, theta, rho, phi)
    w_svi = svi_total_variance(K_GRID, ssvi_slice_as_svi(theta, rho, phi))
    np.testing.assert_allclose(w_ssvi, w_svi, rtol=0, atol=1e-15)


def test_ssvi_atm_is_theta():
    p = SSVIParams(rho=-0.5, eta=1.2, gamma=0.4)
    for th in (0.01, 0.05, 0.3):
        assert p.total_variance(0.0, th) == pytest.approx(th, abs=1e-15)


def test_ssvi_no_arbitrage_conditions_give_positive_density():
    """Theorem 4.2 conditions hold -> Durrleman g >= 0 on the slice."""
    p = SSVIParams(rho=-0.6, eta=1.2, gamma=0.45)
    assert p.static_arbitrage_free()
    thetas = np.array([0.01, 0.03, 0.06, 0.1, 0.2])
    for th in thetas:
        assert p.butterfly_free(th)
        svi = ssvi_slice_as_svi(th, p.rho, float(p.phi(th)))
        assert np.all(durrleman_g(np.linspace(-4, 4, 801), svi) >= -1e-12)
    assert calendar_arbitrage_free(p, thetas)


def test_essvi_rho_interpolates_and_fit_recovers():
    true = ESSVIParams(rho_lo=-0.7, rho_hi=-0.3, theta_lo=0.01, theta_hi=0.12,
                       eta=1.0, gamma=0.4)
    thetas = np.array([0.01, 0.03, 0.06, 0.12])
    assert true.rho_of(0.01) == pytest.approx(-0.7)
    assert true.rho_of(0.12) == pytest.approx(-0.3)
    ks = [np.linspace(-0.5, 0.5, 21)] * 4
    ws = [true.total_variance(k, th) for k, th in zip(ks, thetas, strict=False)]
    fit = fit_ssvi(ks, ws, thetas, extended=True)
    assert fit.rmse_total_variance < 1e-7
    assert fit.params.rho_of(0.01) == pytest.approx(-0.7, abs=1e-3)
    assert fit.params.rho_of(0.12) == pytest.approx(-0.3, abs=1e-3)


def test_ssvi_fit_recovers_params():
    true = SSVIParams(rho=-0.4, eta=0.9, gamma=0.35)
    thetas = np.array([0.02, 0.05, 0.1])
    ks = [np.linspace(-0.6, 0.6, 25)] * 3
    ws = [true.total_variance(k, th) for k, th in zip(ks, thetas, strict=False)]
    fit = fit_ssvi(ks, ws, thetas)
    assert fit.params.rho == pytest.approx(-0.4, abs=1e-5)
    assert fit.params.eta == pytest.approx(0.9, abs=1e-5)
    assert fit.params.gamma == pytest.approx(0.35, abs=1e-5)


def test_calendar_check_flags_crossing():
    p = SSVIParams(rho=-0.5, eta=1.0, gamma=0.4)
    assert not calendar_arbitrage_free(p, np.array([0.05, 0.04]))


# --------------------------------------------------------------- Dupire
def test_dupire_flat_vol_returns_sigma():
    sig = 0.25
    lv = local_vol_from_total_variance(lambda k, T: sig * sig * T + 0 * k, K_GRID, 1.0)
    np.testing.assert_allclose(lv, sig, rtol=1e-6)


def test_dupire_term_structure_only():
    """w = a T + b T^2 -> sigma_loc^2 = dw/dT = a + 2 b T, any k."""
    a, b = 0.04, 0.01
    lv = local_vol_from_total_variance(lambda k, T: a * T + b * T * T + 0 * k, K_GRID, 2.0)
    np.testing.assert_allclose(lv, np.sqrt(a + 2 * b * 2.0), rtol=1e-6)


def _ssvi_surface():
    p = SSVIParams(rho=-0.5, eta=1.0, gamma=0.4)
    return lambda k, T: p.total_variance(k, 0.04 * T)


def test_dupire_two_routes_agree_on_ssvi():
    """Gatheral eq. 1.10 (from w) and Dupire's price formula give one answer."""
    w = _ssvi_surface()
    S, r, q, T = 100.0, 0.03, 0.01, 1.0

    def call(K, t):
        F = S * np.exp((r - q) * t)
        k = np.log(K / F)
        return bs_price(S, K, t, r, q, float(np.sqrt(w(np.array([k]), t)[0] / t)))

    for K in (80.0, 100.0, 125.0):
        k = np.log(K / (S * np.exp((r - q) * T)))
        a = local_vol_from_total_variance(w, k, T)
        b = local_vol_from_prices(call, K, T, r, q)
        assert a == pytest.approx(b, rel=2e-4)


def test_dupire_raises_on_calendar_arbitrage():
    with pytest.raises(LocalVolArbitrageError):
        local_vol_from_total_variance(lambda k, T: 0.04 * (2.0 - T) + 0 * k, 0.0, 1.0)


def test_dupire_mc_reprices_surface():
    """Convergence oracle: the local-vol MC reprices the input implied vols."""
    w = _ssvi_surface()
    S, r, q, T = 100.0, 0.02, 0.0, 0.5
    K = np.array([85.0, 100.0, 115.0])
    px, se = local_vol_mc_price(w, S, K, T, r, q, n_paths=40_000, n_steps=100, seed=11)
    F = S * np.exp((r - q) * T)
    for Ki, p, s in zip(K, px, se, strict=False):
        ref = bs_price(S, Ki, T, r, q, float(np.sqrt(w(np.array([np.log(Ki / F)]), T)[0] / T)))
        assert abs(p - ref) < 4.0 * s + 0.02, (Ki, p, ref, s)


# ---------------------------------------------------------------- Bates
HP = HestonParams(v0=0.04, kappa=1.5, theta=0.05, xi=0.5, rho=-0.6)


def test_bates_without_jumps_is_heston():
    j = JumpParams(lam=0.0, mu_j=-0.1, delta=0.1)
    for K in (80.0, 100.0, 120.0):
        assert bates_call(100, K, 1.0, 0.03, 0.01, HP, j) == pytest.approx(
            heston_call(100, K, 1.0, 0.03, 0.01, HP), abs=1e-7)


def test_bates_deterministic_vol_is_merton():
    """xi = 0, v0 = theta -> Merton (1976) closed-form series."""
    p = HestonParams(v0=0.04, kappa=1.0, theta=0.04, xi=0.0, rho=0.0)
    j = JumpParams(lam=0.8, mu_j=-0.15, delta=0.2)
    for K in (70.0, 100.0, 140.0):
        for T in (0.25, 2.0):
            assert bates_call(100, K, T, 0.03, 0.01, p, j) == pytest.approx(
                merton_call(100, K, T, 0.03, 0.01, 0.2, j), abs=1e-8)


def test_bates_put_call_parity_and_forward():
    j = JumpParams(lam=0.5, mu_j=-0.1, delta=0.15)
    S, r, q, T = 100.0, 0.03, 0.01, 1.5
    c = bates_call(S, 100.0, T, r, q, HP, j)
    p = bates_put(S, 100.0, T, r, q, HP, j)
    assert c - p == pytest.approx(S * np.exp(-q * T) - 100 * np.exp(-r * T), abs=1e-12)
    # K -> 0 call is the discounted forward: the jump compensator is right
    assert bates_call(S, 1e-4, T, r, q, HP, j) == pytest.approx(
        S * np.exp(-q * T) - 1e-4 * np.exp(-r * T), rel=1e-7)


def test_bates_mc_convergence():
    """Independent check: Euler MC of Bates dynamics within 4 SE of Lewis price."""
    rng = np.random.default_rng(3)
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.02, 0.0
    j = JumpParams(lam=1.0, mu_j=-0.1, delta=0.1)
    n, steps = 100_000, 200
    dt = T / steps
    x = np.zeros(n)
    v = np.full(n, HP.v0)
    for _ in range(steps):
        z1 = rng.standard_normal(n)
        z2 = HP.rho * z1 + np.sqrt(1 - HP.rho**2) * rng.standard_normal(n)
        vp = np.maximum(v, 0.0)
        nj = rng.poisson(j.lam * dt, n)
        jumps = nj * j.mu_j + np.sqrt(nj) * j.delta * rng.standard_normal(n)
        x += (r - q - j.lam * j.kbar - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1 + jumps
        v += HP.kappa * (HP.theta - vp) * dt + HP.xi * np.sqrt(vp * dt) * z2
    pay = np.exp(-r * T) * np.maximum(S * np.exp(x) - K, 0.0)
    ref = bates_call(S, K, T, r, q, HP, j)
    assert abs(pay.mean() - ref) < 4 * pay.std() / np.sqrt(n) + 0.03
