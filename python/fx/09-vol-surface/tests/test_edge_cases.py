"""Edge cases from the documentation contract: T->0, flat smiles,
negative rates, extreme vols, rho at the bounds, degenerate inputs."""

import math

import numpy as np
import pytest

from fx_surface import (
    HestonParams,
    SmileQuotes,
    atm_dns_strike,
    gk_delta,
    gk_forward,
    gk_price,
    implied_vol,
    price_cos,
    solve_pillar_strikes,
    strike_from_delta,
    vols_from_quotes,
)
from fx_surface.surface import build_slice

S, RD, RF, SIG = 1.10, 0.045, 0.033, 0.09


def test_short_expiry_price_approaches_intrinsic():
    T = 1e-6
    for K in (1.05, 1.15):
        c = gk_price(S, K, T, RD, RF, SIG, 1)
        assert c == pytest.approx(max(S - K, 0.0), abs=1e-6)


def test_short_expiry_implied_vol_round_trip():
    T = 1e-4
    price = gk_price(S, 1.101, T, RD, RF, 0.10, 1)
    assert implied_vol(price, S, 1.101, T, RD, RF, 1) == pytest.approx(0.10, abs=1e-8)


def test_zero_or_negative_expiry_raises():
    with pytest.raises(ValueError):
        gk_price(S, 1.1, 0.0, RD, RF, SIG, 1)
    with pytest.raises(ValueError):
        gk_price(S, 1.1, -0.5, RD, RF, SIG, 1)
    with pytest.raises(ValueError):
        strike_from_delta(0.25, 1, SIG, S, 0.0, RD, RF, "spot")


def test_short_dated_strike_solving_stable():
    T = 1.0 / 365.0
    for conv in ("spot", "spot_pa"):
        for cp in (1, -1):
            K = strike_from_delta(0.25, cp, 0.08, S, T, RD, RF, conv)
            d = gk_delta(S, K, T, RD, RF, 0.08, cp, conv)
            assert abs(abs(d) - 0.25) < 1e-8


def test_flat_smile_zero_rr_bf_pipeline():
    """Zero RR and BF: smile collapses to flat; SVI degenerates cleanly
    and the surface returns the ATM vol everywhere."""
    q = SmileQuotes(0.08, 0.0, 0.0, 0.0, 0.0)
    sl = build_slice("3m", 0.25, S, RD, RF, q, "spot", "svi")
    assert sl.smile.params.b == 0.0
    for K in (0.9, 1.1, 1.3):
        assert float(sl.smile.vol(K)) == pytest.approx(0.08, abs=1e-10)
    ok, _ = sl.smile.is_butterfly_arbitrage_free()
    assert ok


def test_negative_rates_both_legs_strike_solving():
    rd, rf = -0.0075, -0.005
    vols = vols_from_quotes(SmileQuotes(0.07, -0.004, 0.002, -0.007, 0.006))
    for conv in ("spot", "forward", "spot_pa"):
        strikes = solve_pillar_strikes(vols, S, 1.0, rd, rf, conv)
        ks = [strikes[p] for p in ("10p", "25p", "atm", "25c", "10c")]
        assert all(a < b for a, b in zip(ks, ks[1:]))


def test_negative_rates_heston_parity():
    rd, rf = -0.0075, -0.005
    p = HestonParams(v0=0.005, kappa=1.5, theta=0.006, xi=0.3, rho=-0.2)
    F = gk_forward(S, 1.0, rd, rf)
    c = float(price_cos(S, 1.1, 1.0, rd, rf, p, 1))
    pt = float(price_cos(S, 1.1, 1.0, rd, rf, p, -1))
    assert c - pt == pytest.approx(math.exp(-rd) * (F - 1.1), abs=1e-10)


def test_em_high_vol_preset_full_build(em_market):
    """35% ATM, 30% domestic rates, +6% RR: everything still works."""
    from fx_surface.surface import build_surface

    surf = build_surface(em_market, "svi")
    assert surf.is_calendar_arbitrage_free()
    for sl in surf.slices:
        ks = [sl.strikes[p] for p in ("10p", "25p", "atm", "25c", "10c")]
        assert all(a < b for a, b in zip(ks, ks[1:]))
        ok, g_min = sl.smile.is_butterfly_arbitrage_free(k_lo=-1.0, k_hi=1.5)
        assert ok, f"{sl.label}: {g_min}"
    # positive RR: call wing above put wing
    v25c = surf.vol_delta(0.25, 1.0, +1)
    v25p = surf.vol_delta(0.25, 1.0, -1)
    assert v25c - v25p > 0.03


def test_em_atm_strike_far_above_spot(em_market):
    """30% carry: the 1y forward (and ATM strike) sits ~30% above spot -
    conventions must not silently anchor ATM at spot."""
    sl = [s for s in em_market.slices if s.label == "1y"][0]
    F = gk_forward(em_market.S, sl.T, sl.r_d, sl.r_f)
    assert F / em_market.S > 1.25
    vols = vols_from_quotes(sl.quotes)
    K_atm = atm_dns_strike(F, vols["atm"], sl.T)
    assert K_atm > F > em_market.S


def test_extreme_vol_implied_round_trip():
    price = gk_price(8.5, 11.0, 1.0, 0.30, 0.045, 0.35, 1)
    assert implied_vol(price, 8.5, 11.0, 1.0, 0.30, 0.045, 1) == pytest.approx(0.35, abs=1e-9)


def test_deep_itm_implied_vol_near_bound():
    """Deep ITM: price is nearly all intrinsic; the solver must stay
    stable where vega is tiny."""
    K = 0.75
    price = gk_price(S, K, 0.5, RD, RF, 0.10, 1)
    assert implied_vol(price, S, K, 0.5, RD, RF, 1) == pytest.approx(0.10, abs=1e-6)


def test_rho_bounds_mc_and_fourier():
    p = HestonParams(v0=0.0064, kappa=2.0, theta=0.008, xi=0.4, rho=-1.0)
    from fx_surface import mc_price

    ref = float(price_cos(S, 1.12, 0.5, RD, RF, p, 1))
    price, se = mc_price(S, 1.12, 0.5, RD, RF, p, 1, n_paths=60_000, n_steps=24,
                         scheme="qe", seed=11)
    assert abs(price - ref) < 3 * se


def test_vol_to_zero_and_large_vol_prices():
    lo = gk_price(S, 1.10, 0.5, RD, RF, 1e-4, 1)
    hi = gk_price(S, 1.10, 0.5, RD, RF, 20.0, 1)
    F = gk_forward(S, 0.5, RD, RF)
    assert lo == pytest.approx(math.exp(-RD * 0.5) * max(F - 1.10, 0.0), abs=1e-6)
    assert hi == pytest.approx(S * math.exp(-RF * 0.5), rel=1e-6)  # -> spot bound


def test_atm_delta_neutrality_all_expiries(usdjpy_surface):
    """The ATM strike is genuinely delta-neutral under the native pa
    convention at every tenor."""
    for sl in usdjpy_surface.slices:
        dc = gk_delta(sl.S, sl.strikes["atm"], sl.T, sl.r_d, sl.r_f,
                      sl.vols["atm"], +1, sl.convention)
        dp = gk_delta(sl.S, sl.strikes["atm"], sl.T, sl.r_d, sl.r_f,
                      sl.vols["atm"], -1, sl.convention)
        assert abs(dc + dp) < 1e-10
