"""Total-variance surface: interpolation, calendar arbitrage, extrapolation."""

from __future__ import annotations

import numpy as np
import pytest

from eq_surface.smile import SVIParams, svi_total_variance
from eq_surface.surface import VolSurface, check_calendar

S, R, Q = 100.0, 0.02, 0.01


def make_slices():
    """Two calendar-consistent slices (total variance grows with T)."""
    p1 = SVIParams(a=0.010, b=0.05, rho=-0.4, m=0.0, sigma=0.15)
    p2 = SVIParams(a=0.030, b=0.08, rho=-0.4, m=0.0, sigma=0.25)
    return np.array([0.25, 1.0]), [p1, p2]


def test_interpolation_exact_at_pillars():
    Ts, slices = make_slices()
    surf = VolSurface(Ts, slices, S, R, Q)
    for T, p in zip(Ts, slices):
        F = surf.forward(T)
        for K in [80.0, 100.0, 115.0]:
            k = np.log(K / F)
            expected = np.sqrt(float(svi_total_variance(k, p)) / T)
            assert surf.vol(K, T) == pytest.approx(expected, abs=1e-12)


def test_total_variance_linear_in_T_hand_checked():
    Ts, slices = make_slices()
    surf = VolSurface(Ts, slices, S, R, Q)
    k = 0.05
    w1 = float(svi_total_variance(k, slices[0]))
    w2 = float(svi_total_variance(k, slices[1]))
    # T = 0.625 is 50% of the way from 0.25 to 1.0
    T = 0.625
    lam = (T - 0.25) / (1.0 - 0.25)
    expected = (1 - lam) * w1 + lam * w2
    assert surf.total_variance(k, T) == pytest.approx(expected, abs=1e-14)


def test_calendar_arbitrage_detected_on_planted_decrease():
    # Second slice with LOWER total variance -> calendar arbitrage.
    p1 = SVIParams(a=0.040, b=0.05, rho=-0.4, m=0.0, sigma=0.15)
    p2 = SVIParams(a=0.010, b=0.05, rho=-0.4, m=0.0, sigma=0.15)
    with pytest.warns(UserWarning, match="calendar arbitrage"):
        surf = VolSurface(np.array([0.25, 1.0]), [p1, p2], S, R, Q)
    assert not surf.calendar.is_free
    assert surf.calendar.worst_violation < -0.02
    assert len(surf.calendar.violations) > 0


def test_calendar_enforcement_monotonises_total_variance():
    p1 = SVIParams(a=0.040, b=0.05, rho=-0.4, m=0.0, sigma=0.15)
    p2 = SVIParams(a=0.010, b=0.05, rho=-0.4, m=0.0, sigma=0.15)
    with pytest.warns(UserWarning, match="running-max"):
        surf = VolSurface(np.array([0.25, 1.0]), [p1, p2], S, R, Q, enforce_calendar=True)
    for k in [-0.5, 0.0, 0.4]:
        w_short = surf.total_variance(k, 0.25)
        w_long = surf.total_variance(k, 1.0)
        assert w_long >= w_short - 1e-12


def test_check_calendar_direct():
    Ts = np.array([0.5, 1.0])
    k_grid = np.array([-0.1, 0.0, 0.1])
    w = np.array([[0.02, 0.02, 0.02], [0.03, 0.015, 0.03]])  # dip at k=0
    res = check_calendar(Ts, w, k_grid)
    assert not res.is_free
    assert res.worst_violation == pytest.approx(-0.005)
    assert len(res.violations) == 1
    assert res.violations[0][2] == 0.0  # at k = 0


def test_short_end_extrapolation_flat_vol():
    """T below first pillar: w scales as T/T1 -> implied vol equals pillar vol."""
    Ts, slices = make_slices()
    surf = VolSurface(Ts, slices, S, R, Q)
    k = -0.1
    vol_pillar = surf.vol_k(k, 0.25)
    assert surf.vol_k(k, 0.10) == pytest.approx(vol_pillar, abs=1e-12)
    assert surf.total_variance(k, 0.10) == pytest.approx(
        surf.total_variance(k, 0.25) * 0.10 / 0.25, abs=1e-14
    )


def test_long_end_extrapolation_linear_slope():
    Ts, slices = make_slices()
    surf = VolSurface(Ts, slices, S, R, Q)
    k = 0.0
    w1 = surf.total_variance(k, 0.25)
    w2 = surf.total_variance(k, 1.0)
    slope = (w2 - w1) / 0.75
    assert surf.total_variance(k, 2.0) == pytest.approx(w2 + slope * 1.0, abs=1e-13)


def test_long_end_extrapolation_never_decreases():
    # Slices whose wings cross (slice-to-slice slope negative in a wing):
    # the construction warns about the crossing, and long-end extrapolation
    # floors the slope at zero so w never decreases beyond the last pillar.
    p1 = SVIParams(a=0.020, b=0.30, rho=-0.8, m=0.0, sigma=0.10)
    p2 = SVIParams(a=0.045, b=0.05, rho=-0.2, m=0.0, sigma=0.30)
    with pytest.warns(UserWarning, match="calendar"):
        surf = VolSurface(np.array([0.5, 1.0]), [p1, p2], S, R, Q)
    for k in [-1.2, 0.0, 1.2]:
        assert surf.total_variance(k, 5.0) >= surf.total_variance(k, 1.0) - 1e-14


def test_single_expiry_surface_valid():
    p1 = SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    surf = VolSurface(np.array([0.5]), [p1], S, R, Q)
    F = surf.forward(0.5)
    # at the pillar: exact
    assert surf.vol(F, 0.5) == pytest.approx(np.sqrt(float(svi_total_variance(0.0, p1)) / 0.5), abs=1e-12)
    # short end: flat vol
    assert surf.vol_k(0.0, 0.2) == pytest.approx(surf.vol_k(0.0, 0.5), abs=1e-12)
    # long end: flat-vol continuation (slope w1/T1)
    assert surf.vol_k(0.0, 2.0) == pytest.approx(surf.vol_k(0.0, 0.5), abs=1e-12)


def test_forward_moneyness_handling():
    Ts, slices = make_slices()
    surf = VolSurface(Ts, slices, S, R, Q)
    T = 0.6
    F = S * np.exp((R - Q) * T)
    assert surf.forward(T) == pytest.approx(F, abs=1e-12)
    # vol at K = F must equal vol at k = 0
    assert surf.vol(F, T) == pytest.approx(surf.vol_k(0.0, T), abs=1e-14)


def test_vol_vectorised_over_strikes():
    Ts, slices = make_slices()
    surf = VolSurface(Ts, slices, S, R, Q)
    Ks = np.array([80.0, 100.0, 120.0])
    vols = surf.vol(Ks, 0.5)
    assert vols.shape == (3,)
    assert np.all(np.isfinite(vols))
    assert vols[0] > vols[2]  # negative skew


def test_invalid_construction_raises():
    p = SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.2)
    with pytest.raises(ValueError, match="non-empty"):
        VolSurface(np.array([]), [], S, R, Q)
    with pytest.raises(ValueError, match="positive"):
        VolSurface(np.array([-0.5]), [p], S, R, Q)
    with pytest.raises(ValueError, match="strictly increasing"):
        VolSurface(np.array([1.0, 0.5]), [p, p], S, R, Q)
    with pytest.raises(ValueError, match="slices"):
        VolSurface(np.array([0.5, 1.0]), [p], S, R, Q)
    with pytest.raises(ValueError, match="spot"):
        VolSurface(np.array([0.5]), [p], -1.0, R, Q)


def test_invalid_queries_raise():
    Ts, slices = make_slices()
    surf = VolSurface(Ts, slices, S, R, Q)
    with pytest.raises(ValueError, match="T must be positive"):
        surf.vol(100.0, 0.0)
    with pytest.raises(ValueError, match="strikes must be positive"):
        surf.vol(-100.0, 0.5)


# -- butterfly re-check at interpolated expiries and after enforcement -------
def test_butterfly_report_covers_pillars_and_midpoints():
    Ts, slices = make_slices()
    surf = VolSurface(Ts, slices, S, R, Q)
    rep = surf.butterfly_report()
    assert [r["T"] for r in rep] == [0.25, 0.625, 1.0]
    # Smooth arbitrage-free slices: g > 0 up to the O(h^2) FD error.
    assert all(r["min_g"] > -1e-6 for r in rep)


def test_butterfly_report_flags_arbitrageable_pillar_and_its_neighbourhood():
    """Vogt's slice has negative density near k = 0.5 (Gatheral-Jacquier).
    The surface check must flag it at the pillar and at an interpolated
    expiry where that slice dominates the blend."""
    vogt = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    later = SVIParams(a=-0.035, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    surf = VolSurface(np.array([1.0, 2.0]), [vogt, later], S, R, Q)
    rep = {r["T"]: r for r in surf.butterfly_report(np.array([1.0, 1.1]))}
    assert rep[1.0]["min_g"] < 0.0
    assert rep[1.1]["min_g"] < 0.0


def test_enforced_surface_is_c1_in_k_and_rechecked():
    p1 = SVIParams(a=0.040, b=0.05, rho=-0.4, m=0.0, sigma=0.15)
    p2 = SVIParams(a=0.010, b=0.05, rho=-0.4, m=0.0, sigma=0.15)
    with pytest.warns(UserWarning, match="running-max"):
        surf = VolSurface(np.array([0.25, 1.0]), [p1, p2], S, R, Q, enforce_calendar=True)
    assert surf.butterfly_after_enforcement is not None
    assert len(surf.butterfly_after_enforcement) == 3  # 2 pillars + 1 midpoint
    # PCHIP: one-sided slopes at an interior node agree to O(delta); the old
    # piecewise-linear interpolant had a slope jump of O(grid curvature).
    k0, d = float(surf.k_grid[120]), 1e-7
    w = lambda x: float(surf.total_variance(x, 1.0))
    right = (w(k0 + d) - w(k0)) / d
    left = (w(k0) - w(k0 - d)) / d
    assert abs(right - left) < 1e-4
