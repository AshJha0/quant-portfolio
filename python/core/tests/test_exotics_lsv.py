"""Particle-method LSV reprices vanillas off the Dupire surface (W16 acceptance)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qp_core.exotics.barriers import vanilla_price
from qp_core.exotics.lsv import (
    HestonParams,
    calibrate_lsv_particles,
    dupire_local_vol,
    lsv_price_vanillas,
)
from scipy.optimize import brentq

RHO, ETA, GAM = -0.4, 1.0, 0.5  # power-law SSVI, eta (1 + |rho|) = 1.4 <= 2: arbitrage-free
S0, R, Q, T = 100.0, 0.02, 0.0, 1.0


def ssvi_w(k: np.ndarray, t: float) -> np.ndarray:
    th = 0.04 * t
    phi = ETA / (th**GAM * (1 + th) ** (1 - GAM))
    pk = phi * np.asarray(k, dtype=float)
    return 0.5 * th * (1 + RHO * pk + np.sqrt((pk + RHO) ** 2 + 1 - RHO * RHO))


def surface_call(K: float) -> tuple[float, float]:
    F = S0 * math.exp((R - Q) * T)
    vol = math.sqrt(float(ssvi_w(np.array([math.log(K / F)]), T)[0]) / T)
    return vanilla_price("call", S0, K, T, R, R - Q, vol), vol


def test_dupire_flat_surface_is_flat():
    lv = dupire_local_vol(lambda k, t: 0.04 * t + 0 * np.asarray(k), np.linspace(-1, 1, 11), 0.5)
    assert np.allclose(lv, 0.2, atol=1e-8)
    with pytest.raises(ValueError, match="arbitrage"):
        dupire_local_vol(lambda k, t: 0.04 * (2.0 - t) + 0 * np.asarray(k), np.zeros(1), 0.5)


@pytest.fixture(scope="module")
def calibrated():
    h = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.6)
    return calibrate_lsv_particles(ssvi_w, S0, R, Q, T, h, n_particles=100_000, n_steps=100, seed=0)


def test_lsv_reprices_vanillas_within_3se_and_02_vol(calibrated):
    K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    p, se = lsv_price_vanillas(calibrated, K, T, n_paths=200_000, seed=1)
    for Ki, pi, si in zip(K, p, se, strict=True):
        ref, vol = surface_call(Ki)
        iv = brentq(lambda s, Ki=Ki, pi=pi: vanilla_price("call", S0, Ki, T, R, R - Q, s) - pi, 0.01, 2.0)
        assert abs(pi - ref) < 3.0 * si, (Ki, pi, ref, si)
        assert abs(iv - vol) < 0.002, (Ki, iv, vol)


def test_pure_local_vol_limit_xi_zero():
    """xi = 0 and v0 = theta: LSV is Dupire local vol and the leverage is sigma_Dup / sqrt(v0)."""
    h = HestonParams(v0=0.04, kappa=1.0, theta=0.04, xi=0.0, rho=0.0)
    cal = calibrate_lsv_particles(ssvi_w, S0, R, Q, T, h, n_particles=20_000, n_steps=50, seed=3)
    i = 25
    fwd = math.log(S0) + (R - Q) * cal.times[i]
    lv = dupire_local_vol(ssvi_w, cal.x_grids[i] - fwd, float(cal.times[i]), dT=1e-4)
    assert np.allclose(cal.leverage[i], lv / 0.2, rtol=1e-10)
    K = np.array([90.0, 100.0, 110.0])
    p, se = lsv_price_vanillas(cal, K, T, n_paths=100_000, seed=4)
    for Ki, pi, si in zip(K, p, se, strict=True):
        assert abs(pi - surface_call(Ki)[0]) < 3.0 * si + 0.01


def test_lsv_input_checks(calibrated):
    with pytest.raises(ValueError):
        lsv_price_vanillas(calibrated, np.array([100.0]), 2.0)
    with pytest.raises(ValueError):
        HestonParams(v0=0.04, kappa=1.0, theta=0.04, xi=0.3, rho=1.0)
