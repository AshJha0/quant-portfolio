"""Arbitrage-violating surface inputs, extreme wings, and the T -> 0 limit.

These are the domain-specific edge cases required by CONVENTIONS.md item 6
for a volatility-surface project: no-arbitrage invariants must be *checked*
(and the checker must actually fire), the wings must stay finite and
well-ordered, and the short end must not blow up.
"""

import warnings

import numpy as np
import pytest

from eq_surface.black_scholes import bs_price, bs_vega, implied_vol
from eq_surface.smile import (SVIParams, check_butterfly, durrleman_g, fit_svi,
                              svi_d2w_dk2, svi_dw_dk, svi_implied_vol,
                              svi_total_variance)
from eq_surface.surface import VolSurface, check_calendar


def _slice(a=0.02, b=0.1, rho=-0.4, m=0.0, sigma=0.1) -> SVIParams:
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def _surface(enforce=False) -> VolSurface:
    return VolSurface(
        np.array([0.1, 0.5, 1.0, 2.0]),
        [_slice(0.02, 0.10, -0.4, 0.0, 0.10),
         _slice(0.05, 0.12, -0.4, 0.0, 0.12),
         _slice(0.09, 0.15, -0.4, 0.0, 0.15),
         _slice(0.16, 0.18, -0.4, 0.0, 0.18)],
        spot=100.0, rate=0.02, div_yield=0.01, enforce_calendar=enforce,
    )


# ---------------------------------------------------------------------------
# T -> 0
# ---------------------------------------------------------------------------

def test_short_end_vol_is_flat_as_T_goes_to_zero():
    """Regression: an absolute floor `w >= 1e-12` clamped legitimate total
    variance at tiny T, so sqrt(w/T) reported an ATM vol of 1.0 (100 vol
    points) at T = 1e-12 instead of the flat first-pillar vol."""
    s = _surface()
    ref = s.vol_k(0.0, 0.1)  # first pillar
    for T in (1e-14, 1e-12, 1e-10, 1e-6, 1e-3, 0.05, 0.1):
        v = s.vol_k(0.0, T)
        assert np.isfinite(v)
        assert v == pytest.approx(ref, rel=1e-12), f"short-end vol broke at T={T}"


def test_short_end_total_variance_goes_to_zero_linearly():
    s = _surface()
    w1 = s.total_variance(0.0, 1e-6)
    w2 = s.total_variance(0.0, 2e-6)
    assert w2 == pytest.approx(2.0 * w1, rel=1e-10)
    assert s.total_variance(0.0, 1e-12) > 0.0


def test_T_zero_and_negative_rejected():
    s = _surface()
    with pytest.raises(ValueError, match="T must be positive"):
        s.total_variance(0.0, 0.0)
    with pytest.raises(ValueError, match="T must be positive"):
        s.vol_k(0.0, -0.1)
    with pytest.raises(ValueError, match="T must be positive"):
        svi_implied_vol(0.0, _slice(), 0.0)


def test_option_price_converges_to_intrinsic_as_T_to_zero():
    """Deep ITM call must converge to S - K (no discounting left)."""
    S, K = 100.0, 80.0
    prev = np.inf
    for T in (1e-2, 1e-4, 1e-6, 1e-8):
        p = bs_price(S, K, T, r=0.02, q=0.0, sigma=0.3, kind="call")
        assert p >= max(S - K, 0.0) - 1e-9
        prev = p
    assert prev == pytest.approx(S - K, abs=1e-5)


# ---------------------------------------------------------------------------
# Wings
# ---------------------------------------------------------------------------

def test_svi_wings_are_asymptotically_linear_in_k():
    """Raw SVI has linear wings: w'(k) -> b(rho +/- 1). This is exactly the
    Lee moment-formula slope bound and is what keeps the wings arbitrage-free."""
    p = _slice(b=0.12, rho=-0.4)
    for k, expected in ((-200.0, p.b * (p.rho - 1.0)), (200.0, p.b * (p.rho + 1.0))):
        assert float(svi_dw_dk(k, p)) == pytest.approx(expected, rel=1e-6)
    # Curvature dies in the wings as |k|^-3: doubling k must cut w'' by 8x.
    c1 = float(svi_d2w_dk2(200.0, p))
    c2 = float(svi_d2w_dk2(400.0, p))
    assert c1 > 0.0 and c2 > 0.0
    assert c1 / c2 == pytest.approx(8.0, rel=1e-4)
    assert c1 == pytest.approx(p.b * p.sigma**2 / 200.0**3, rel=1e-5)


def test_lee_slope_bound_respected_in_wings():
    """Lee (2004): total-variance wing slope must not exceed 2 in |k|, or
    the moment formula is violated. Raw SVI with b(1+|rho|) <= 2 satisfies it."""
    p = _slice(b=0.12, rho=-0.4)
    assert p.b * (1.0 + abs(p.rho)) <= 2.0
    k = np.linspace(-50.0, 50.0, 501)
    assert np.all(np.abs(np.asarray(svi_dw_dk(k, p))) <= 2.0 + 1e-12)


def test_deep_wing_total_variance_stays_positive_and_finite():
    p = _slice()
    k = np.array([-500.0, -100.0, -20.0, 0.0, 20.0, 100.0, 500.0])
    w = np.asarray(svi_total_variance(k, p))
    assert np.all(np.isfinite(w))
    assert np.all(w > 0.0)


def test_surface_wing_vols_finite_far_outside_quoted_strikes():
    s = _surface()
    for T in (0.1, 1.0, 5.0):
        vols = np.asarray(s.vol(np.array([1.0, 10.0, 100.0, 1_000.0, 10_000.0]), T))
        assert np.all(np.isfinite(vols))
        assert np.all(vols > 0.0)


def test_deep_wing_implied_vol_returns_nan_not_garbage():
    """When vega underflows there is no vol information left in the price;
    the solver must say so rather than return an arbitrary bracket end."""
    S, K, T = 100.0, 1_000.0, 0.01
    px = bs_price(S, K, T, 0.0, 0.0, 0.2, "call")
    assert bs_vega(S, K, T, 0.0, 0.0, 0.2) < 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iv = implied_vol(px, S, K, T, 0.0, 0.0, "call")
    assert np.isnan(iv)


# ---------------------------------------------------------------------------
# Butterfly (strike) arbitrage
# ---------------------------------------------------------------------------

def test_butterfly_checker_fires_on_arbitrage_violating_quotes():
    """A kinked (V-shaped) total-variance smile has a negative density; the
    Durrleman check must flag it AND the fitter must warn."""
    k = np.linspace(-0.4, 0.4, 11)
    w_bad = 0.04 + 0.5 * np.abs(k)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = fit_svi(k, w_bad, T=1.0)
    assert res.arb_free is False
    assert res.min_g < 0.0
    assert any("Durrleman" in str(r.message) for r in rec)


def test_butterfly_checker_passes_a_benign_slice():
    ok, min_g, viol = check_butterfly(_slice())
    assert ok is True
    assert min_g > 0.0
    assert viol.size == 0


def test_durrleman_g_is_one_for_a_flat_smile():
    """Flat total variance (w' = w'' = 0) has g(k) == 1 identically: a
    lognormal density, the canonical arbitrage-free reference."""
    flat = SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    g = np.asarray(durrleman_g(np.linspace(-2.0, 2.0, 51), flat))
    assert np.allclose(g, 1.0, atol=1e-12)


def test_svi_params_reject_negative_total_variance():
    with pytest.raises(ValueError, match="non-positive total variance"):
        SVIParams(a=-0.02, b=1.5, rho=-0.9, m=0.0, sigma=0.02)
    with pytest.raises(ValueError, match="rho"):
        SVIParams(a=0.04, b=0.1, rho=-1.0, m=0.0, sigma=0.1)
    with pytest.raises(ValueError, match="sigma"):
        SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.0)
    with pytest.raises(ValueError, match="b must be"):
        SVIParams(a=0.04, b=-0.1, rho=0.0, m=0.0, sigma=0.1)


def test_call_price_convex_in_strike_for_arb_free_slice():
    """No butterfly arbitrage <=> C(K) convex. Check it on actual prices
    reconstructed from the fitted slice, not just via the g-function."""
    s = _surface()
    T = 1.0
    F = s.forward(T)
    K = np.linspace(0.5 * F, 1.8 * F, 200)
    vols = np.asarray(s.vol(K, T))
    C = np.array([bs_price(100.0, k_, T, 0.02, 0.01, v, "call")
                  for k_, v in zip(K, vols)])
    second = np.diff(C, 2)
    assert np.all(second > -1e-10), f"negative density, min d2C = {second.min():.3e}"
    # Calls are also decreasing in strike.
    assert np.all(np.diff(C) < 0.0)


# ---------------------------------------------------------------------------
# Calendar arbitrage
# ---------------------------------------------------------------------------

def test_calendar_checker_detects_decreasing_total_variance():
    hi = _slice(a=0.09)
    lo = _slice(a=0.02)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        s = VolSurface(np.array([0.25, 1.0]), [hi, lo], 100.0, 0.02, 0.0)
    assert s.calendar.is_free is False
    assert s.calendar.worst_violation < 0.0
    assert len(s.calendar.violations) > 0
    assert any("calendar arbitrage" in str(r.message) for r in rec)


def test_calendar_enforcement_makes_total_variance_monotone_in_T():
    hi = _slice(a=0.09)
    lo = _slice(a=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = VolSurface(np.array([0.25, 1.0]), [hi, lo], 100.0, 0.02, 0.0,
                       enforce_calendar=True)
    assert np.all(np.diff(s.w_grid, axis=0) >= -1e-15)
    # And the property survives query-level interpolation/extrapolation.
    ks = np.array([-0.8, -0.2, 0.0, 0.3, 0.9])
    Ts = np.linspace(0.02, 4.0, 120)
    W = np.array([np.asarray(s.total_variance(ks, T)) for T in Ts])
    assert np.all(np.diff(W, axis=0) >= -1e-14)


def test_arb_free_surface_total_variance_monotone_in_T_everywhere():
    """Property test on the healthy surface: w(k, T) non-decreasing in T for
    every k, including both extrapolation regions."""
    s = _surface()
    assert s.calendar.is_free
    ks = np.linspace(-1.2, 1.2, 25)
    Ts = np.concatenate([np.linspace(1e-6, 0.1, 20), np.linspace(0.1, 6.0, 80)])
    W = np.array([np.asarray(s.total_variance(ks, T)) for T in Ts])
    assert np.all(np.diff(W, axis=0) >= -1e-14)


def test_extrapolated_slope_never_decreases_total_variance():
    """Beyond the last pillar the slope is floored at zero, so even a surface
    whose last two pillars decrease cannot extrapolate into arbitrage."""
    hi = _slice(a=0.20)
    lo = _slice(a=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = VolSurface(np.array([1.0, 2.0]), [hi, lo], 100.0, 0.0, 0.0)
    w_last = s.total_variance(0.0, 2.0)
    for T in (2.5, 5.0, 20.0):
        assert s.total_variance(0.0, T) >= w_last - 1e-14


def test_check_calendar_helper_on_hand_built_grids():
    k = np.array([-0.1, 0.0, 0.1])
    good = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02]])
    bad = np.array([[0.02, 0.02, 0.02], [0.01, 0.03, 0.02]])
    assert check_calendar(np.array([0.5, 1.0]), good, k).is_free
    res = check_calendar(np.array([0.5, 1.0]), bad, k)
    assert not res.is_free
    assert res.worst_violation == pytest.approx(-0.01, abs=1e-15)
    assert len(res.violations) == 1


# ---------------------------------------------------------------------------
# Degenerate surface construction
# ---------------------------------------------------------------------------

def test_surface_rejects_degenerate_pillar_sets():
    p = _slice()
    with pytest.raises(ValueError, match="strictly increasing"):
        VolSurface(np.array([1.0, 1.0]), [p, p], 100.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        VolSurface(np.array([2.0, 1.0]), [p, p], 100.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        VolSurface(np.array([0.0, 1.0]), [p, p], 100.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="non-empty"):
        VolSurface(np.array([]), [], 100.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="slices"):
        VolSurface(np.array([1.0, 2.0]), [p], 100.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="spot"):
        VolSurface(np.array([1.0]), [p], 0.0, 0.0, 0.0)


def test_surface_rejects_non_positive_strikes():
    s = _surface()
    with pytest.raises(ValueError, match="strikes must be positive"):
        s.vol(np.array([100.0, 0.0]), 1.0)
    with pytest.raises(ValueError, match="strikes must be positive"):
        s.vol(-50.0, 1.0)


def test_single_pillar_surface_is_flat_in_T():
    """One pillar: w scales linearly in T both sides, i.e. flat vol forever."""
    s = VolSurface(np.array([1.0]), [_slice()], 100.0, 0.0, 0.0)
    v0 = s.vol_k(0.1, 1.0)
    for T in (1e-6, 0.01, 1.0, 3.0, 30.0):
        assert s.vol_k(0.1, T) == pytest.approx(v0, rel=1e-12)


def test_vol_monotone_decreasing_in_strike_on_the_downside_skew():
    """Equity index skew: with rho < 0 the smile falls as strike rises over
    the put wing. A sign flip here means the skew convention broke."""
    s = _surface()
    F = s.forward(1.0)
    K = np.linspace(0.6 * F, 0.95 * F, 40)
    vols = np.asarray(s.vol(K, 1.0))
    assert np.all(np.diff(vols) < 0.0)
