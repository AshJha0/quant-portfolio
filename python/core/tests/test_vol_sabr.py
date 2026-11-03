"""qp_core.vol.sabr: published values, QuantLib oracle, calibration, parity with eq_options."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qp_core.vol.sabr import black76_price, sabr_calibrate, sabr_density_check, sabr_vol

ARGS = (0.03, 5.0, 0.02, 0.5, -0.3, 0.4)  # F, T, alpha, beta, rho, nu (module docstring)


def test_docstring_worked_example():
    F, T, a, b, r, n = ARGS
    for v in ("hagan", "obloj"):
        assert sabr_vol(F, F, T, a, b, r, n, variant=v) == pytest.approx(0.121209, abs=5e-7)
    assert sabr_vol(F, 0.01, T, a, b, r, n, variant="hagan") == pytest.approx(0.28182, abs=5e-6)
    assert sabr_vol(F, 0.01, T, a, b, r, n, variant="obloj") == pytest.approx(0.28346, abs=5e-6)


def test_beta1_nu0_is_flat_alpha():
    for k in (0.5, 1.0, 2.0):
        assert sabr_vol(1.0, k, 3.0, 0.25, 1.0, 0.0, 0.0) == pytest.approx(0.25, rel=1e-15)


def test_atm_matches_hagan_atm_formula():
    F, T, a, b, rho, nu = ARGS
    atm = a / F ** (1 - b) * (1 + T * ((1 - b) ** 2 * a * a / (24 * F ** (2 - 2 * b))
                                       + rho * b * nu * a / (4 * F ** (1 - b))
                                       + (2 - 3 * rho * rho) * nu * nu / 24))
    for variant in ("hagan", "obloj"):
        assert sabr_vol(F, F, T, a, b, rho, nu, variant=variant) == pytest.approx(atm, rel=1e-14)


def test_bit_identical_to_eq_options_copy():
    eq = pytest.importorskip("eq_options.sabr")
    F, T, a, b, r, n = ARGS
    for k in np.linspace(0.002, 0.1, 37):
        for v in ("hagan", "obloj"):
            assert sabr_vol(F, k, T, a, b, r, n, variant=v) == eq.sabr_vol(F, k, T, a, b, r, n, variant=v)


def test_black76_put_call_parity_and_limits():
    c = black76_price(0.03, 0.025, 2.0, 0.01, 0.2, "call")
    p = black76_price(0.03, 0.025, 2.0, 0.01, 0.2, "put")
    assert c - p == pytest.approx(math.exp(-0.02) * 0.005, abs=1e-16)
    assert black76_price(0.03, 0.025, 0.0, 0.01, 0.2) == pytest.approx(0.005)
    with pytest.raises(ValueError):
        black76_price(0.03, 0.025, 1.0, 0.0, 0.2, "straddle")


def test_density_flags_low_strike_arbitrage():
    k, d = sabr_density_check(0.03, 10.0, 0.02, 0.5, -0.3, 0.6, np.linspace(0.0005, 0.1, 200))
    assert d.min() < 0.0 and k[d < 0].max() < 0.03


@pytest.mark.parametrize("variant", ["hagan", "obloj"])
def test_calibration_recovers_parameters(variant):
    F, T, a, b, r, n = ARGS
    ks = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06])
    vols = [sabr_vol(F, k, T, a, b, r, n, variant=variant) for k in ks]
    fit = sabr_calibrate(F, T, ks, vols, beta=b, variant=variant)
    assert (fit.alpha, fit.rho, fit.nu) == pytest.approx((a, r, n), abs=1e-6)
    assert fit.rmse < 1e-10


def test_calibration_validation():
    with pytest.raises(ValueError):
        sabr_calibrate(0.03, 1.0, [0.02, 0.03], [0.2, 0.2], beta=0.5)
    with pytest.raises(ValueError):
        sabr_calibrate(0.03, 1.0, [0.02, 0.03, 0.04], [0.2, -0.1, 0.2], beta=0.5)
