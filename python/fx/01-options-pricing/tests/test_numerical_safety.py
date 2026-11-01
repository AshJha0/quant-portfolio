"""Regression tests for wrong numbers and silent failures at the domain edges."""

import math

import numpy as np
import pytest

import fx_options.deltas as deltas_mod
from fx_options import (analytic_greeks, cip_forward, forward_points, gk_price,
                        mc_price, strike_from_delta)
from fx_options.deltas import _log_ncdf, _pa_fold_bracket, _pa_peak_strike


def test_sigma_zero_large_carry_is_finite_and_exact() -> None:
    """(r_d - r_f) T = 800 overflows the forward; the price must not be NaN."""
    assert gk_price(1.0, 1.0, 1000.0, 0.8, 0.0, 0.0, "call") == 1.0
    assert gk_price(1.0, 1.0, 1000.0, 0.8, 0.0, 0.0, "put") == 0.0


def test_overflow_raises_value_error() -> None:
    with pytest.raises(ValueError):
        gk_price(1.0, 1.0, 1000.0, -0.8, 0.0, 0.2, "put")
    with pytest.raises(ValueError):
        cip_forward(1.0, 1000.0, 0.8, 0.0)
    with pytest.raises(ValueError):
        forward_points(1.0, 1.0, 0.03, 0.01, pip_factor=math.inf)


def test_numpy_scalars_accepted() -> None:
    ref = gk_price(1.1, 1.1, 0.5, 0.04, 0.02, 0.1, "call")
    assert gk_price(np.float64(1.1), np.float32(1.1), np.float64(0.5),
                    np.float64(0.04), np.float64(0.02), np.float64(0.1),
                    "call") == pytest.approx(ref, rel=1e-7)  # float32 strike
    assert gk_price(np.int64(150), np.int64(150), 1, 0.01, 0.05, 0.1, "call") == \
        gk_price(150.0, 150.0, 1.0, 0.01, 0.05, 0.1, "call")
    assert mc_price(1.1, 1.1, 0.5, 0.04, 0.02, 0.1, "call", n_paths=np.int64(1000),
                    rng=1).std_error > 0


@pytest.mark.parametrize("v", [1e-6, 0.1, 1.0, 19.0, 21.0, 36.0, 40.0, 100.0])
def test_pa_fold_bracket_valid_for_any_total_vol(v: float) -> None:
    """The bracket must straddle the fold root for every v = sigma sqrt T.

    The previous fixed bracket [-20, 20] failed for v > 20.
    """
    lo, hi = _pa_fold_bracket(v)
    log_v = math.log(v)

    def h(x: float) -> float:
        return _log_ncdf(x) + log_v + 0.5 * x * x + 0.9189385332046728

    assert h(lo) < 0.0 < h(hi)


@pytest.mark.parametrize("sigma,T", [(25.0, 1.0), (5.0, 30.0), (0.1, 1.0)])
def test_pa_peak_strike_large_total_vol(sigma: float, T: float) -> None:
    """The fold satisfies N(d2) v = n(d2); check it in log form."""
    F = 1.1
    v = sigma * math.sqrt(T)
    k = _pa_peak_strike(F, T, sigma)
    d2 = (math.log(F / k) - 0.5 * v * v) / v
    resid = _log_ncdf(d2) + math.log(v) + 0.5 * d2 * d2 + 0.9189385332046728
    assert abs(resid) < 1e-9


@pytest.mark.parametrize("x", [-37.0000001, -40.0, -100.0, -1e4])
def test_log_ncdf_series_branch_against_mpmath(x: float) -> None:
    import mpmath as mp
    mp.mp.dps = 40
    # Truncation bound 945/x^10 (2.1e-13 at the switch) plus rounding.
    assert abs(_log_ncdf(x) - float(mp.log(mp.ncdf(x)))) <= 3e-13 + 4e-16 * x * x


def test_brent_non_convergence_raises(monkeypatch) -> None:
    class Info:
        converged = False
        flag = "convergence error"

    monkeypatch.setattr(deltas_mod, "brentq",
                        lambda *a, **k: (1.0, Info()))
    with pytest.raises(ValueError, match="did not converge"):
        strike_from_delta(-0.25, 1.1, 0.5, 0.04, 0.02, 0.1, "put", "spot_pa")


def test_mc_needs_independent_samples() -> None:
    args = (1.1, 1.1, 0.5, 0.04, 0.02, 0.1, "call")
    with pytest.raises(ValueError):
        mc_price(*args, n_paths=2, rng=1, antithetic=True, control_variate=False)
    with pytest.raises(ValueError):
        mc_price(*args, n_paths=4, rng=1, antithetic=True, control_variate=True)
    assert mc_price(*args, n_paths=6, rng=1).std_error > 0


def test_mc_overflowing_carry_raises() -> None:
    with pytest.raises(ValueError):
        mc_price(1.0, 1.0, 1000.0, 0.8, 0.0, 0.2, "call", n_paths=1000, rng=1)


def test_jpy_gamma_relative_scale() -> None:
    """Greeks for USDJPY: gamma is a small number on a large spot."""
    g = analytic_greeks(147.5, 150.0, 0.5, 0.005, 0.0525, 0.1075, "call")
    assert 0.0 < g.gamma < 0.1
