"""Heston: characteristic function identities, Fourier cross-validation."""

import math

import numpy as np
import pytest

from fx_surface import (
    HestonParams,
    gk_forward,
    gk_price,
    heston_cf,
    heston_digital,
    heston_price,
    price_cos,
    price_gil_pelaez,
)

S, RD, RF = 1.10, 0.045, 0.033
PARAMS = HestonParams(v0=0.0064, kappa=1.8, theta=0.008, xi=0.45, rho=-0.35)


def test_cf_at_zero_is_one():
    for T in (0.02, 0.5, 2.0):
        val = complex(heston_cf(0.0, T, PARAMS, RD - RF, math.log(S)))
        assert val == pytest.approx(1.0 + 0.0j, abs=1e-14)


def test_cf_martingale_identity():
    """phi(-i) = E[S_T] = F: the discounted spot is a martingale."""
    for T in (0.1, 1.0, 2.0):
        val = complex(heston_cf(-1j, T, PARAMS, RD - RF, math.log(S)))
        F = gk_forward(S, T, RD, RF)
        assert val.real == pytest.approx(F, rel=1e-13)
        assert abs(val.imag) < 1e-13


def test_cf_conjugate_symmetry():
    u = np.linspace(0.5, 50, 25)
    a = heston_cf(u, 1.0, PARAMS, RD - RF, math.log(S))
    b = heston_cf(-u, 1.0, PARAMS, RD - RF, math.log(S))
    np.testing.assert_allclose(a, np.conj(b), rtol=1e-12)


def test_little_trap_long_maturity_stable():
    """The little-trap CF is continuous in u even at very long T."""
    u = np.linspace(0.01, 100, 500)
    vals = heston_cf(u, 15.0, PARAMS, RD - RF, math.log(S))
    assert np.all(np.isfinite(vals))
    assert np.all(np.abs(vals) <= 1.0 + 1e-12)
    # no branch-cut jumps: |phi| varies smoothly
    assert np.max(np.abs(np.diff(np.abs(vals)))) < 0.05


@pytest.mark.parametrize("T", [0.02, 0.25, 1.0, 2.0])
@pytest.mark.parametrize("K", [0.90, 1.00, 1.10, 1.20, 1.40])
def test_two_fourier_methods_agree(T, K):
    a = price_gil_pelaez(S, K, T, RD, RF, PARAMS, 1)
    b = float(price_cos(S, K, T, RD, RF, PARAMS, 1))
    assert abs(a - b) < 1e-6


@pytest.mark.parametrize("method", ["cos", "gil_pelaez"])
@pytest.mark.parametrize("K", [0.95, 1.10, 1.30])
def test_put_call_parity(method, K):
    T = 0.75
    c = float(heston_price(S, K, T, RD, RF, PARAMS, +1, method=method))
    p = float(heston_price(S, K, T, RD, RF, PARAMS, -1, method=method))
    F = gk_forward(S, T, RD, RF)
    assert c - p == pytest.approx(math.exp(-RD * T) * (F - K), abs=1e-8)


def test_xi_to_zero_degenerates_to_gk():
    """xi -> 0: variance is deterministic, price -> GK at the effective
    vol sqrt(mean integrated variance)."""
    p = HestonParams(v0=0.0064, kappa=1.8, theta=0.008, xi=1e-4, rho=0.0)
    for T in (0.25, 1.0):
        w = p.theta * T + (p.v0 - p.theta) * (1 - math.exp(-p.kappa * T)) / p.kappa
        sig_eff = math.sqrt(w / T)
        for K in (1.0, 1.1, 1.2):
            h = float(price_cos(S, K, T, RD, RF, p, 1))
            g = gk_price(S, K, T, RD, RF, sig_eff, 1)
            assert h == pytest.approx(g, abs=1e-6)


def test_deep_itm_otm_limits():
    T = 1.0
    F = gk_forward(S, T, RD, RF)
    # ~10-sigma strikes: the COS absolute noise floor is ~1e-6 here
    deep_itm = float(price_cos(S, 0.4, T, RD, RF, PARAMS, 1))
    assert deep_itm == pytest.approx(math.exp(-RD * T) * (F - 0.4), abs=5e-6)
    deep_otm = float(price_cos(S, 3.0, T, RD, RF, PARAMS, 1))
    assert 0.0 <= deep_otm < 5e-6


def test_vectorised_cos_matches_scalar():
    Ks = np.array([0.95, 1.05, 1.15, 1.25])
    vec = price_cos(S, Ks, 0.5, RD, RF, PARAMS, 1)
    for K, v in zip(Ks, vec):
        assert float(price_cos(S, float(K), 0.5, RD, RF, PARAMS, 1)) == pytest.approx(v, abs=1e-14)


def test_digital_matches_strike_derivative_and_bounds():
    T, K = 0.5, 1.12
    dig = heston_digital(S, K, T, RD, RF, PARAMS, 1)
    h = 5e-5
    fd = -(
        float(price_cos(S, K + h, T, RD, RF, PARAMS, 1))
        - float(price_cos(S, K - h, T, RD, RF, PARAMS, 1))
    ) / (2 * h)
    assert dig == pytest.approx(fd, abs=5e-5)
    assert 0.0 < dig < math.exp(-RD * T)
    # call + put digitals = discounted 1
    dig_p = heston_digital(S, K, T, RD, RF, PARAMS, -1)
    assert dig + dig_p == pytest.approx(math.exp(-RD * T), abs=1e-12)


def test_rho_extremes_price_finite_and_parity():
    for rho in (-1.0, 1.0):
        p = HestonParams(v0=0.01, kappa=2.0, theta=0.01, xi=0.5, rho=rho)
        c = float(price_cos(S, 1.15, 1.0, RD, RF, p, 1))
        pt = float(price_cos(S, 1.15, 1.0, RD, RF, p, -1))
        F = gk_forward(S, 1.0, RD, RF)
        assert np.isfinite(c) and c > 0
        assert c - pt == pytest.approx(math.exp(-RD) * (F - 1.15), abs=1e-10)


def test_feller_condition_reporting():
    good = HestonParams(v0=0.01, kappa=3.0, theta=0.02, xi=0.3, rho=-0.5)
    bad = HestonParams(v0=0.01, kappa=1.0, theta=0.01, xi=0.6, rho=-0.5)
    assert good.feller_satisfied and good.feller_ratio == pytest.approx(2 * 3 * 0.02 / 0.09)
    assert not bad.feller_satisfied


def test_params_validation():
    with pytest.raises(ValueError, match="v0 and theta"):
        HestonParams(v0=-0.01, kappa=1.0, theta=0.01, xi=0.3, rho=0.0)
    with pytest.raises(ValueError, match="kappa"):
        HestonParams(v0=0.01, kappa=0.0, theta=0.01, xi=0.3, rho=0.0)
    with pytest.raises(ValueError, match="xi"):
        HestonParams(v0=0.01, kappa=1.0, theta=0.01, xi=0.0, rho=0.0)
    with pytest.raises(ValueError, match="rho"):
        HestonParams(v0=0.01, kappa=1.0, theta=0.01, xi=0.3, rho=-1.5)
    with pytest.raises(ValueError, match="method"):
        heston_price(S, 1.1, 1.0, RD, RF, PARAMS, 1, method="fft")
    with pytest.raises(ValueError, match="cp"):
        price_cos(S, 1.1, 1.0, RD, RF, PARAMS, 0)
