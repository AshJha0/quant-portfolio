"""Heston Monte Carlo: full-truncation Euler and Andersen QE."""

from __future__ import annotations

import numpy as np
import pytest

from eq_surface.black_scholes import bs_price
from eq_surface.heston import heston_call_gl
from eq_surface.heston_mc import heston_mc_price, simulate_heston_terminal

S, R, Q = 100.0, 0.02, 0.01
N_PATHS = 150_000


def test_euler_fine_steps_within_3se_of_fourier(mild_heston):
    ref = float(heston_call_gl(S, 100.0, 1.0, R, Q, mild_heston))
    res = heston_mc_price(S, 100.0, 1.0, R, Q, mild_heston, n_paths=N_PATHS,
                          n_steps=64, scheme="euler_ft", seed=11)
    assert abs(res.price - ref) < 3.0 * res.stderr


def test_qe_within_3se_at_coarse_steps_where_euler_is_biased(extreme_heston):
    """The comparative bias test: 8 steps/year on a Feller-violating set.

    Euler's truncation bias is tens of standard errors; QE stays within 3 SE
    at the same step count.
    """
    ref = float(heston_call_gl(S, 100.0, 1.0, R, Q, extreme_heston))
    eu = heston_mc_price(S, 100.0, 1.0, R, Q, extreme_heston, n_paths=N_PATHS,
                         n_steps=8, scheme="euler_ft", seed=42)
    qe = heston_mc_price(S, 100.0, 1.0, R, Q, extreme_heston, n_paths=N_PATHS,
                         n_steps=8, scheme="qe", seed=42)
    assert abs(eu.price - ref) > 3.0 * eu.stderr  # Euler IS biased at 8 steps
    assert abs(qe.price - ref) < 3.0 * qe.stderr  # QE is not
    # and QE's absolute bias is strictly smaller
    assert abs(qe.price - ref) < abs(eu.price - ref)


def test_qe_beats_euler_bias_on_mild_set_too(mild_heston):
    ref = float(heston_call_gl(S, 100.0, 1.0, R, Q, mild_heston))
    eu = heston_mc_price(S, 100.0, 1.0, R, Q, mild_heston, n_paths=N_PATHS,
                         n_steps=8, scheme="euler_ft", seed=5)
    qe = heston_mc_price(S, 100.0, 1.0, R, Q, mild_heston, n_paths=N_PATHS,
                         n_steps=8, scheme="qe", seed=5)
    assert abs(qe.price - ref) <= abs(eu.price - ref) + 2.0 * qe.stderr


def test_seeded_reproducibility(mild_heston):
    a = heston_mc_price(S, 100.0, 0.5, R, Q, mild_heston, n_paths=20_000, n_steps=16, seed=7)
    b = heston_mc_price(S, 100.0, 0.5, R, Q, mild_heston, n_paths=20_000, n_steps=16, seed=7)
    c = heston_mc_price(S, 100.0, 0.5, R, Q, mild_heston, n_paths=20_000, n_steps=16, seed=8)
    assert a.price == b.price and a.stderr == b.stderr
    assert a.price != c.price


def test_variance_stays_nonnegative_both_schemes(extreme_heston):
    """Even on a Feller-violating set, stored variance is never negative."""
    for scheme in ("euler_ft", "qe"):
        _, v_T = simulate_heston_terminal(S, 1.0, R, Q, extreme_heston,
                                          n_paths=50_000, n_steps=16, scheme=scheme, seed=3)
        assert np.all(v_T >= 0.0), scheme
        assert np.any(v_T == 0.0) or np.all(v_T > 0.0)  # sanity: array is well-formed


def test_martingale_property_qe(mild_heston):
    """E[S_T] must equal the forward within MC noise."""
    s_T, _ = simulate_heston_terminal(S, 1.0, R, Q, mild_heston,
                                      n_paths=200_000, n_steps=32, scheme="qe", seed=21)
    fwd = S * np.exp((R - Q) * 1.0)
    se = s_T.std(ddof=1) / np.sqrt(s_T.size)
    assert abs(s_T.mean() - fwd) < 3.0 * se


def test_put_pricing_within_3se(mild_heston):
    from eq_surface.heston import heston_put

    ref = heston_put(S, 110.0, 0.5, R, Q, mild_heston)
    res = heston_mc_price(S, 110.0, 0.5, R, Q, mild_heston, n_paths=N_PATHS,
                          n_steps=32, scheme="qe", seed=13, kind="put")
    assert abs(res.price - ref) < 3.0 * res.stderr


def test_xi_zero_qe_matches_deterministic_variance_price():
    from eq_surface.heston import HestonParams, heston_call_p1p2

    p = HestonParams(v0=0.09, kappa=2.0, theta=0.04, rho=0.0, xi=0.0)
    ref = heston_call_p1p2(S, 100.0, 1.0, R, Q, p)
    res = heston_mc_price(S, 100.0, 1.0, R, Q, p, n_paths=N_PATHS, n_steps=32,
                          scheme="qe", seed=9)
    assert abs(res.price - ref) < 3.0 * res.stderr
    # variance path is deterministic: terminal v is a constant
    _, v_T = simulate_heston_terminal(S, 1.0, R, Q, p, 100, 8, "qe", seed=1)
    assert np.allclose(v_T, v_T[0])


def test_rho_minus_one_boundary_simulates(mild_heston):
    from eq_surface.heston import HestonParams

    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, rho=-1.0, xi=0.4)
    ref = float(heston_call_gl(S, 100.0, 0.5, R, Q, p))
    res = heston_mc_price(S, 100.0, 0.5, R, Q, p, n_paths=N_PATHS, n_steps=32,
                          scheme="qe", seed=17)
    assert abs(res.price - ref) < 3.0 * res.stderr


def test_generator_instance_accepted(mild_heston):
    rng = np.random.default_rng(99)
    s_T, _ = simulate_heston_terminal(S, 0.5, R, Q, mild_heston, 1000, 8, "qe", seed=rng)
    assert s_T.shape == (1000,)
    assert np.all(s_T > 0.0)


def test_invalid_inputs_raise(mild_heston):
    with pytest.raises(ValueError, match="paths"):
        simulate_heston_terminal(S, 1.0, R, Q, mild_heston, 1, 8)
    with pytest.raises(ValueError, match="step"):
        simulate_heston_terminal(S, 1.0, R, Q, mild_heston, 100, 0)
    with pytest.raises(ValueError, match="T must be positive"):
        simulate_heston_terminal(S, 0.0, R, Q, mild_heston, 100, 8)
    with pytest.raises(ValueError, match="spot"):
        simulate_heston_terminal(-S, 1.0, R, Q, mild_heston, 100, 8)
    with pytest.raises(ValueError, match="scheme"):
        simulate_heston_terminal(S, 1.0, R, Q, mild_heston, 100, 8, scheme="milstein")
    with pytest.raises(ValueError, match="strike"):
        heston_mc_price(S, -1.0, 1.0, R, Q, mild_heston)
    with pytest.raises(ValueError, match="kind"):
        heston_mc_price(S, 100.0, 1.0, R, Q, mild_heston, kind="chooser")
