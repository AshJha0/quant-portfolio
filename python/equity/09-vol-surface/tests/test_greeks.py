"""Heston Greeks: FD stability, Richardson consistency, BS comparisons."""

from __future__ import annotations

import numpy as np
import pytest

from eq_surface.black_scholes import bs_delta
from eq_surface.greeks import bs_equivalent_greeks, heston_greeks, smile_adjusted_delta

S, R, Q = 100.0, 0.02, 0.01


def test_fd_delta_stable_across_bump_sizes(mild_heston):
    """The Fourier pricer is smooth: FD deltas agree across bump decades."""
    deltas = [
        heston_greeks(S, 100.0, 0.5, R, Q, mild_heston, rel_bump=h).delta
        for h in (3e-3, 1e-3, 3e-4)
    ]
    assert max(deltas) - min(deltas) < 1e-4
    # a 30x coarser bump still agrees to central-difference truncation order
    coarse = heston_greeks(S, 100.0, 0.5, R, Q, mild_heston, rel_bump=1e-2).delta
    assert abs(coarse - deltas[-1]) < 1e-3


def test_richardson_consistency(mild_heston):
    """Richardson at coarse bump matches plain FD at fine bump (higher order)."""
    rich = heston_greeks(S, 100.0, 0.5, R, Q, mild_heston, rel_bump=1e-2, richardson=True)
    fine = heston_greeks(S, 100.0, 0.5, R, Q, mild_heston, rel_bump=1e-4, richardson=False)
    assert rich.delta == pytest.approx(fine.delta, abs=1e-6)
    assert rich.gamma == pytest.approx(fine.gamma, abs=1e-4)
    assert rich.vega_v0 == pytest.approx(fine.vega_v0, abs=1e-3)


def test_heston_delta_close_to_bs_atm_short_dated(mild_heston):
    """Short-dated ATM: stochastic vol barely moves delta away from BS.

    Compared at the option's own implied vol; the residual gap is the smile-
    dynamics term (vega * dsigma/dS), which shrinks with sqrt(T).
    """
    T = 1.0 / 52.0
    g = heston_greeks(S, 100.0, T, R, Q, mild_heston, richardson=True)
    iv = bs_equivalent_greeks(S, 100.0, T, R, Q, mild_heston)["implied_vol"]
    bs_d = bs_delta(S, 100.0, T, R, Q, iv, "call")
    assert g.delta == pytest.approx(bs_d, abs=0.02)


def test_gamma_positive_vega_positive(mild_heston, extreme_heston):
    for p in (mild_heston, extreme_heston):
        for K in (90.0, 100.0, 115.0):
            g = heston_greeks(S, K, 0.5, R, Q, p, richardson=True)
            assert g.gamma > 0.0, (p, K)
            assert g.vega_v0 > 0.0, (p, K)


def test_delta_bounds_and_rate_rho_sign(mild_heston):
    g_itm = heston_greeks(S, 70.0, 0.5, R, Q, mild_heston)
    g_otm = heston_greeks(S, 140.0, 0.5, R, Q, mild_heston)
    for g in (g_itm, g_otm):
        assert 0.0 <= g.delta <= np.exp(-Q * 0.5) + 1e-10
        assert g.rho_rate > 0.0  # calls gain when rates rise
    assert g_itm.delta > g_otm.delta


def test_bs_equivalent_greeks(mild_heston):
    out = bs_equivalent_greeks(S, 100.0, 0.5, R, Q, mild_heston)
    assert 0.1 < out["implied_vol"] < 0.4
    g = heston_greeks(S, 100.0, 0.5, R, Q, mild_heston, richardson=True)
    # ATM: BS-equivalent delta close to Heston FD delta
    assert out["delta"] == pytest.approx(g.delta, abs=0.1)
    assert out["gamma"] > 0.0 and out["vega"] > 0.0


def test_bs_equivalent_deep_wing_graceful(mild_heston):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # deep-wing inversion warns by design
        out = bs_equivalent_greeks(S, 5000.0, 0.05, R, Q, mild_heston)
    assert np.isnan(out["implied_vol"]) and np.isnan(out["delta"])


def test_smile_adjusted_delta_signs():
    # negative skew slope -> sticky-moneyness delta above sticky-strike
    out = smile_adjusted_delta(S, 100.0, 0.5, R, Q, sigma=0.2, dsigma_dk=-0.10)
    assert out["adjustment"] > 0.0
    assert out["delta_sticky_moneyness"] > out["delta_sticky_strike"]
    # flat smile -> no adjustment
    flat = smile_adjusted_delta(S, 100.0, 0.5, R, Q, sigma=0.2, dsigma_dk=0.0)
    assert flat["adjustment"] == 0.0
    # positive slope (call skew) -> negative adjustment
    pos = smile_adjusted_delta(S, 100.0, 0.5, R, Q, sigma=0.2, dsigma_dk=0.10)
    assert pos["adjustment"] < 0.0


def test_smile_adjusted_delta_magnitude_hand_checked():
    from eq_surface.black_scholes import bs_vega

    slope = -0.10
    out = smile_adjusted_delta(S, 100.0, 0.5, R, Q, 0.2, slope)
    expected_adj = bs_vega(S, 100.0, 0.5, R, Q, 0.2) * (-slope / S)
    assert out["adjustment"] == pytest.approx(expected_adj, abs=1e-12)


def test_invalid_inputs_raise(mild_heston):
    with pytest.raises(ValueError, match="T must be positive"):
        heston_greeks(S, 100.0, 0.0, R, Q, mild_heston)
    with pytest.raises(ValueError, match="rel_bump"):
        heston_greeks(S, 100.0, 0.5, R, Q, mild_heston, rel_bump=0.5)
    with pytest.raises(ValueError, match="sigma"):
        smile_adjusted_delta(S, 100.0, 0.5, R, Q, sigma=-0.2, dsigma_dk=0.0)
