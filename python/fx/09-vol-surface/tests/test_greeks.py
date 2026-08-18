"""Heston FD Greeks: signs vs BS-world, both rhos, FD stability."""

import numpy as np
import pytest

from fx_surface import HestonParams, gk_greeks, heston_greeks_fd

S, RD, RF = 1.10, 0.045, 0.033
PARAMS = HestonParams(v0=0.0064, kappa=1.8, theta=0.008, xi=0.45, rho=-0.35)
T = 0.75
ATM_VOL = 0.082  # roughly the model ATM vol for the comparison


@pytest.fixture(scope="module")
def atm_greeks():
    return heston_greeks_fd(S, 1.11, T, RD, RF, PARAMS, 1)


def test_gamma_positive(atm_greeks):
    assert atm_greeks["gamma"] > 0


def test_vega_positive(atm_greeks):
    assert atm_greeks["vega"] > 0


def test_delta_in_unit_interval(atm_greeks):
    assert 0.0 < atm_greeks["delta"] < 1.0


def test_both_rhos_signs_call(atm_greeks):
    """Call: domestic rho positive (forward rises), foreign rho negative."""
    assert atm_greeks["rho_d"] > 0
    assert atm_greeks["rho_f"] < 0
    bs = gk_greeks(S, 1.11, T, RD, RF, ATM_VOL, 1)
    assert np.sign(atm_greeks["rho_d"]) == np.sign(bs["rho_d"])
    assert np.sign(atm_greeks["rho_f"]) == np.sign(bs["rho_f"])


def test_both_rhos_signs_put():
    g = heston_greeks_fd(S, 1.11, T, RD, RF, PARAMS, -1)
    assert g["rho_d"] < 0
    assert g["rho_f"] > 0
    assert -1.0 < g["delta"] < 0.0


def test_theta_negative_for_otm(atm_greeks):
    assert atm_greeks["theta"] < 0


def test_vanna_sign_matches_bs_wings():
    """High strikes: vanna > 0; low strikes: vanna < 0 (both worlds)."""
    for K, sign in ((1.22, +1), (0.98, -1)):
        h = heston_greeks_fd(S, K, T, RD, RF, PARAMS, 1)
        b = gk_greeks(S, K, T, RD, RF, ATM_VOL, 1)
        assert np.sign(h["vanna"]) == sign
        assert np.sign(b["vanna"]) == sign


def test_volga_positive_in_wings_matches_bs():
    for K in (0.98, 1.25):
        h = heston_greeks_fd(S, K, T, RD, RF, PARAMS, 1)
        b = gk_greeks(S, K, T, RD, RF, ATM_VOL, 1)
        assert h["volga"] > 0
        assert b["volga"] > 0


def test_heston_greeks_near_bs_when_xi_small():
    """xi ~ 0 and flat variance: FD Greeks converge to analytic GK."""
    p = HestonParams(v0=0.0064, kappa=1.8, theta=0.0064, xi=1e-4, rho=0.0)
    sig = 0.08
    h = heston_greeks_fd(S, 1.12, T, RD, RF, p, 1)
    b = gk_greeks(S, 1.12, T, RD, RF, sig, 1)
    assert h["delta"] == pytest.approx(b["delta"], abs=2e-4)
    assert h["gamma"] == pytest.approx(b["gamma"], rel=2e-3)
    assert h["vega"] == pytest.approx(b["vega"], rel=2e-3)
    assert h["vanna"] == pytest.approx(b["vanna"], rel=5e-2, abs=5e-3)
    assert h["volga"] == pytest.approx(b["volga"], rel=5e-2, abs=5e-3)
    assert h["rho_d"] == pytest.approx(b["rho_d"], rel=1e-3)
    assert h["rho_f"] == pytest.approx(b["rho_f"], rel=1e-3)
    assert h["theta"] == pytest.approx(b["theta"], rel=5e-3)


def test_fd_step_stability():
    """Halving all FD steps moves each Greek by < 0.5% (rel) - the
    differences sit well above the COS noise floor."""
    K = 1.13
    g1 = heston_greeks_fd(S, K, T, RD, RF, PARAMS, 1)
    g2 = heston_greeks_fd(S, K, T, RD, RF, PARAMS, 1,
                          dS_rel=5e-4, dvol=5e-4, dr=5e-6, dT=5e-5)
    for k in ("delta", "gamma", "vega", "vanna", "volga", "rho_d", "rho_f", "theta"):
        denom = max(abs(g1[k]), 1e-6)
        assert abs(g1[k] - g2[k]) / denom < 5e-3, k


def test_put_call_greek_relations(atm_greeks):
    """Same gamma/vega/vanna/volga for put and call (parity)."""
    gp = heston_greeks_fd(S, 1.11, T, RD, RF, PARAMS, -1)
    for k in ("gamma", "vega", "vanna", "volga"):
        assert gp[k] == pytest.approx(atm_greeks[k], rel=1e-6, abs=1e-8)
    # delta_call - delta_put = e^{-rf T} (spot deltas from FD spot bump)
    assert atm_greeks["delta"] - gp["delta"] == pytest.approx(
        np.exp(-RF * T), abs=1e-6
    )
