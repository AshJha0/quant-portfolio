"""Monte Carlo vs Fourier: 3-standard-error consistency, determinism."""

import math

import numpy as np
import pytest

from fx_surface import HestonParams, gk_forward, mc_price, price_cos, simulate_terminal

S, RD, RF = 1.10, 0.045, 0.033
PARAMS = HestonParams(v0=0.0064, kappa=1.8, theta=0.008, xi=0.45, rho=-0.35)


@pytest.mark.parametrize("K,cp", [(1.05, 1), (1.12, 1), (1.12, -1)])
def test_euler_fine_steps_within_3se(K, cp):
    T = 1.0
    ref = float(price_cos(S, K, T, RD, RF, PARAMS, cp))
    price, se = mc_price(S, K, T, RD, RF, PARAMS, cp, n_paths=120_000,
                         n_steps=250, scheme="euler_ft", seed=1234)
    assert abs(price - ref) < 3.0 * se, f"|{price}-{ref}| vs 3*{se}"


@pytest.mark.parametrize("K,cp", [(1.05, 1), (1.12, 1), (1.20, -1)])
def test_qe_coarse_steps_within_3se(K, cp):
    """QE stays unbiased on a coarse grid (24 steps for 1y) where Euler
    would need hundreds."""
    T = 1.0
    ref = float(price_cos(S, K, T, RD, RF, PARAMS, cp))
    price, se = mc_price(S, K, T, RD, RF, PARAMS, cp, n_paths=120_000,
                         n_steps=24, scheme="qe", seed=99)
    assert abs(price - ref) < 3.0 * se


def test_qe_feller_violated_regime():
    """Strong vol-of-vol (Feller badly violated): QE still matches Fourier."""
    p = HestonParams(v0=0.01, kappa=1.0, theta=0.015, xi=0.8, rho=-0.6)
    assert not p.feller_satisfied
    T, K = 0.5, 1.10
    ref = float(price_cos(S, K, T, RD, RF, p, 1))
    price, se = mc_price(S, K, T, RD, RF, p, 1, n_paths=150_000,
                         n_steps=50, scheme="qe", seed=7)
    assert abs(price - ref) < 3.0 * se


def test_terminal_spot_martingale():
    T = 1.0
    ST = simulate_terminal(S, T, RD, RF, PARAMS, n_paths=200_000, n_steps=24,
                          scheme="qe", seed=2024)
    F = gk_forward(S, T, RD, RF)
    se = ST.std(ddof=1) / math.sqrt(len(ST))
    assert abs(ST.mean() - F) < 3.0 * se


def test_seed_reproducibility():
    a = simulate_terminal(S, 0.5, RD, RF, PARAMS, 1000, 10, "qe", seed=5)
    b = simulate_terminal(S, 0.5, RD, RF, PARAMS, 1000, 10, "qe", seed=5)
    c = simulate_terminal(S, 0.5, RD, RF, PARAMS, 1000, 10, "qe", seed=6)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    # generator objects work too
    d = simulate_terminal(S, 0.5, RD, RF, PARAMS, 1000, 10, "qe",
                          seed=np.random.default_rng(5))
    np.testing.assert_array_equal(a, d)


def test_paths_positive_and_finite():
    for scheme in ("euler_ft", "qe"):
        ST = simulate_terminal(S, 2.0, RD, RF, PARAMS, 20_000, 48, scheme, seed=3)
        assert np.all(ST > 0) and np.all(np.isfinite(ST))


def test_mc_validation_errors():
    with pytest.raises(ValueError, match="scheme"):
        simulate_terminal(S, 1.0, RD, RF, PARAMS, 100, 10, "milstein", 0)
    with pytest.raises(ValueError, match="positive"):
        simulate_terminal(S, 1.0, RD, RF, PARAMS, 0, 10, "qe", 0)
    with pytest.raises(ValueError, match="cp"):
        mc_price(S, 1.1, 1.0, RD, RF, PARAMS, cp=2)
    with pytest.raises(ValueError, match="strike"):
        mc_price(S, -1.0, 1.0, RD, RF, PARAMS)
