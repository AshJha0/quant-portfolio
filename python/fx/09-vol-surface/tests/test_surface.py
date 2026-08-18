"""Surface: delta-space interpolation, calendar checks, query consistency."""

import math

import numpy as np
import pytest

from fx_surface import SmileQuotes, gk_delta
from fx_surface.data import eurusd_market
from fx_surface.data.synthetic import MarketSlice, FXMarketData
from fx_surface.surface import FXVolSurface, build_slice, build_surface


def test_surface_exact_at_pillar_strikes(eurusd_surface):
    for sl in eurusd_surface.slices:
        for p, K in sl.strikes.items():
            assert eurusd_surface.vol(K, sl.T) == pytest.approx(sl.vols[p], abs=1e-8)


def test_delta_interpolation_exact_at_pillars(eurusd):
    """With the surface delta coordinate equal to the quotes' native
    convention, vol_delta at a pillar expiry returns the quoted vol."""
    slices = [
        build_slice(ms.label, ms.T, eurusd.S, ms.r_d, ms.r_f, ms.quotes, ms.convention)
        for ms in eurusd.slices[:5]  # spot-delta tenors
    ]
    surf = FXVolSurface(slices, delta_convention="spot")
    for sl in surf.slices:
        assert surf.vol_delta(0.25, sl.T, +1) == pytest.approx(sl.vols["25c"], abs=1e-7)
        assert surf.vol_delta(0.25, sl.T, -1) == pytest.approx(sl.vols["25p"], abs=1e-7)
        assert surf.vol_delta(0.10, sl.T, -1) == pytest.approx(sl.vols["10p"], abs=1e-7)
        assert surf.vol_atm(sl.T) == pytest.approx(sl.vols["atm"], abs=1e-7)


def test_vol_strike_vs_vol_delta_round_trip(eurusd_surface):
    surf = eurusd_surface
    for K, T in [(1.05, 0.2), (1.10, 0.6), (1.16, 1.4), (1.25, 0.8)]:
        v = surf.vol(K, T)
        r_d, r_f = surf.rates(T)
        cp = 1 if K >= surf.forward(T) else -1
        d = gk_delta(surf.S, K, T, r_d, r_f, v, cp, surf.delta_convention)
        assert surf.vol_delta(abs(d), T, cp) == pytest.approx(v, abs=1e-9)


def test_total_variance_interp_between_pillars(eurusd_surface):
    s0, s1 = eurusd_surface.slices[2], eurusd_surface.slices[3]
    T_mid = 0.5 * (s0.T + s1.T)
    v0 = eurusd_surface.vol_atm(s0.T)
    v1 = eurusd_surface.vol_atm(s1.T)
    vm = eurusd_surface.vol_atm(T_mid)
    w0, w1, wm = v0 * v0 * s0.T, v1 * v1 * s1.T, vm * vm * T_mid
    assert wm == pytest.approx(w0 + (w1 - w0) * (T_mid - s0.T) / (s1.T - s0.T), abs=1e-12)


def test_flat_extrapolation_outside_pillars(eurusd_surface):
    first, last = eurusd_surface.slices[0], eurusd_surface.slices[-1]
    assert eurusd_surface.vol_atm(first.T / 2) == pytest.approx(
        eurusd_surface.vol_atm(first.T), abs=1e-12
    )
    assert eurusd_surface.vol_atm(last.T * 2) == pytest.approx(
        eurusd_surface.vol_atm(last.T), abs=1e-12
    )


def test_preset_surfaces_calendar_free(eurusd_surface, usdjpy_surface):
    assert eurusd_surface.is_calendar_arbitrage_free()
    assert usdjpy_surface.is_calendar_arbitrage_free()


def test_planted_calendar_arbitrage_detected():
    base = eurusd_market()
    slices = []
    for ms in base.slices:
        q = ms.quotes
        if ms.label == "1m":  # blow up 1m vol so 3m total variance falls below
            q = SmileQuotes(0.20, q.rr25, q.bf25, q.rr10, q.bf10)
        slices.append(MarketSlice(ms.label, ms.T, ms.r_d, ms.r_f, q, ms.convention))
    market = FXMarketData(base.pair, base.S, tuple(slices))
    surf = build_surface(market)
    report = surf.calendar_arbitrage_report()
    assert len(report) > 0
    assert any(v["coordinate"] == "atm" for v in report)
    assert all(v["w_to"] < v["w_from"] for v in report)


def test_single_expiry_surface():
    base = eurusd_market()
    ms = base.slices[2]
    sl = build_slice(ms.label, ms.T, base.S, ms.r_d, ms.r_f, ms.quotes, ms.convention)
    surf = FXVolSurface([sl])
    # queries at any T fall back to the single slice (flat in T at fixed delta)
    assert surf.vol_atm(0.05) == pytest.approx(surf.vol_atm(2.0), abs=1e-12)
    v = surf.vol(1.12, 1.0)
    assert 0.05 < v < 0.12
    assert surf.is_calendar_arbitrage_free()


def test_surface_validation():
    with pytest.raises(ValueError, match="at least one"):
        FXVolSurface([])
    base = eurusd_market()
    ms = base.slices[0]
    sl = build_slice(ms.label, ms.T, base.S, ms.r_d, ms.r_f, ms.quotes, ms.convention)
    with pytest.raises(ValueError, match="duplicate"):
        FXVolSurface([sl, sl])
    surf = FXVolSurface([sl])
    with pytest.raises(ValueError, match="positive"):
        surf.vol(-1.0, 0.5)
    with pytest.raises(ValueError, match="T must be positive"):
        surf.vol_atm(0.0)
    with pytest.raises(ValueError, match="delta magnitude"):
        surf.vol_delta(1.5, 0.5, 1)


def test_rates_interpolation(eurusd_surface):
    s0, s1 = eurusd_surface.slices[0], eurusd_surface.slices[1]
    rd_mid, rf_mid = eurusd_surface.rates(0.5 * (s0.T + s1.T))
    assert min(s0.r_d, s1.r_d) <= rd_mid <= max(s0.r_d, s1.r_d)
    assert min(s0.r_f, s1.r_f) <= rf_mid <= max(s0.r_f, s1.r_f)
    # exact at pillars
    assert eurusd_surface.rates(s0.T) == (s0.r_d, s0.r_f)


def test_smile_shape_along_delta_axis(usdjpy_surface):
    """USDJPY: strong negative RR => put wing well above call wing."""
    T = 1.0
    v10p = usdjpy_surface.vol_delta(0.10, T, -1)
    v25p = usdjpy_surface.vol_delta(0.25, T, -1)
    atm = usdjpy_surface.vol_atm(T)
    v25c = usdjpy_surface.vol_delta(0.25, T, +1)
    assert v10p > v25p > atm
    assert v25p - v25c > 0.015  # ~ RR25 magnitude
