"""Surface reads each slice in its quote convention and reproduces pillars.

Regression tests for: the forward-delta default mis-marking premium-
adjusted pairs (USDJPY 2y ATM off by 30 bp), pa call delta solving
crashing on the rising branch, the fixed-delta calendar check presented
as a no-arbitrage test, and solvers returning the bracket edge.
"""

import dataclasses
import math
import warnings

import pytest

from fx_surface import (
    SmileQuotes,
    SurfaceConvergenceError,
    broker_strangle_from_smile,
    gk_price,
    implied_vol,
    smile_quotes_from_broker,
)
from fx_surface.data.synthetic import (
    FXMarketData,
    em_high_vol_market,
    eurusd_market,
    usdjpy_market,
)
from fx_surface.surface import FXVolSurface, build_slice, build_surface

PILLARS = [("atm", None, +1), ("25p", 0.25, -1), ("25c", 0.25, +1),
           ("10p", 0.10, -1), ("10c", 0.10, +1)]
PRESETS = {"eurusd": eurusd_market, "usdjpy": usdjpy_market, "em": em_high_vol_market}
CONVENTIONS = ("spot", "forward", "spot_pa", "forward_pa")


def _surface(market, model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # SVI misfit warning is tested below
        return build_surface(market, smile_model=model)


def _assert_pillars(surf, model):
    """The pillar delta must land on the pillar strike: the surface equals
    the slice smile there to 1e-10, and equals the quote to 1e-10 plus the
    smile's own miss at that strike (zero for exact fits)."""
    labels = PILLARS if model == "svi" else PILLARS[:3]
    for sl in surf.slices:
        for lab, d, cp in labels:
            v = surf.vol_delta(d, sl.T, cp)
            if sl.pillar_fit_error < 1e-12:
                assert abs(v - sl.vols[lab]) < 1e-10, (sl.label, lab)
            else:
                # Inexact SVI fit (USDJPY): the smile misses the quotes by
                # up to pillar_fit_error, and the smile-consistent pillar
                # strike moves with it, so the read-back is bounded by a
                # small multiple of that miss rather than by 1e-10.
                assert abs(v - sl.vols[lab]) < 3.0 * sl.pillar_fit_error, (sl.label, lab)


@pytest.mark.parametrize("model", ["svi", "vv"])
@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_reproduces_pillars(preset, model):
    _assert_pillars(_surface(PRESETS[preset](), model), model)


@pytest.mark.parametrize(
    "preset,conv",
    [("eurusd", c) for c in CONVENTIONS] + [("em", "spot"), ("em", "forward")],
)
def test_every_convention_reproduces_pillars(preset, conv):
    """EM pa variants are left out: the high-carry EM preset quoted in a
    pa convention gives smiles raw SVI cannot interpolate (the miss is
    reported, see test_svi_misfit_is_reported)."""
    base = PRESETS[preset]()
    market = FXMarketData(
        base.pair, base.S,
        tuple(dataclasses.replace(ms, convention=conv) for ms in base.slices),
    )
    surf = _surface(market, "svi")
    _assert_pillars(surf, "svi")
    for sl in surf.slices:  # these SVI fits are exact, so quotes come back
        assert sl.pillar_fit_error < 1e-10


def test_usdjpy_default_surface_atm_matches_pa_quote():
    """The old forward-delta default returned 10.348% for the 10.650% 2y ATM."""
    surf = _surface(usdjpy_market(), "vv")  # VV is exact at ATM
    for sl in surf.slices:
        assert surf.vol_atm(sl.T) == pytest.approx(sl.vols["atm"], abs=1e-10)


def test_pa_call_delta_with_pa_coordinate_does_not_crash():
    surf = FXVolSurface(_surface(usdjpy_market(), "vv").slices, delta_convention="spot_pa")
    sl = [s for s in surf.slices if s.convention == "spot_pa"][-1]
    assert surf.vol_delta(0.25, sl.T, +1) == pytest.approx(sl.vols["25c"], abs=1e-10)


@pytest.mark.parametrize("K,T", [(12.0, 1.5), (14.0, 2.0)])
def test_em_forward_pa_strike_queries_solve(K, T):
    surf = FXVolSurface(_surface(em_high_vol_market(), "svi").slices,
                        delta_convention="forward_pa")
    v = surf.vol(K, T)
    assert math.isfinite(v) and 0.05 < v < 1.0


def test_unattainable_pa_call_delta_raises():
    surf = FXVolSurface(_surface(usdjpy_market(), "svi").slices, delta_convention="spot_pa")
    with pytest.raises(ValueError, match="unattainable"):
        surf.vol_delta(0.95, 1.0, +1)


def test_svi_misfit_is_reported():
    """Raw SVI cannot follow the USDJPY call wing that keeps falling past
    10 delta; the miss is reported rather than hidden."""
    with pytest.warns(UserWarning, match="SVI cannot reproduce"):
        surf = build_surface(usdjpy_market(), smile_model="svi")
    assert max(sl.pillar_fit_error for sl in surf.slices) > 1e-5
    assert any("rho" in sl.smile.active_bounds for sl in surf.slices)


def test_vol_raises_when_fixed_point_does_not_converge():
    surf = _surface(eurusd_market(), "svi")
    with pytest.raises(SurfaceConvergenceError):
        surf.vol(1.30, 0.7, max_iter=1)


def test_unknown_surface_convention_rejected():
    with pytest.raises(ValueError, match="unknown delta convention"):
        FXVolSurface(_surface(eurusd_market(), "svi").slices, delta_convention="fwd")


def test_calendar_check_is_at_fixed_moneyness():
    surf = _surface(eurusd_market(), "svi")
    assert surf.calendar_arbitrage_report() == []
    rep = surf.calendar_arbitrage_report(k_grid=[0.0])
    assert rep == []
    with pytest.warns(DeprecationWarning):
        surf.calendar_arbitrage_report(deltas=(0.25,))


# -- implied vol solver -------------------------------------------------
def test_implied_vol_refuses_price_at_intrinsic():
    S, K, T, rd, rf = 1.10, 0.80, 1.0, 0.04, 0.02
    intrinsic = S * math.exp(-rf * T) - K * math.exp(-rd * T)
    with pytest.raises(ValueError, match="below bracket"):
        implied_vol(intrinsic, S, K, T, rd, rf, +1)
    assert math.isnan(implied_vol(intrinsic, S, K, T, rd, rf, +1, on_fail="nan"))


def test_implied_vol_refuses_price_below_lo_vol_price():
    S, K, T, rd, rf = 1.10, 1.10, 1.0, 0.04, 0.02
    p_lo = gk_price(S, K, T, rd, rf, 1e-3, +1)
    with pytest.raises(ValueError, match="below bracket"):
        implied_vol(0.5 * p_lo + 0.5 * max(S * math.exp(-rf) - K * math.exp(-rd), 0.0),
                    S, K, T, rd, rf, +1, lo=1e-3)


# -- broker strangles ---------------------------------------------------
def _slice(ms, S, quotes=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return build_slice(ms.label, ms.T, S, ms.r_d, ms.r_f, quotes or ms.quotes, ms.convention)


def test_broker_strangle_equals_smile_strangle_without_skew():
    """With RR = 0 on a symmetric-in-delta smile the market-strangle strikes
    are the smile pillar strikes, so the two butterflies coincide."""
    m = eurusd_market()
    ms = m.slices[3]
    q = SmileQuotes(ms.quotes.atm, 0.0, ms.quotes.bf25, 0.0, ms.quotes.bf10)
    sl = _slice(ms, m.S, q)
    for d, bf in ((0.25, q.bf25), (0.10, q.bf10)):
        b = broker_strangle_from_smile(sl.smile.vol, q.atm, d, m.S, ms.T, ms.r_d,
                                       ms.r_f, ms.convention)
        assert b == pytest.approx(bf, abs=2e-6)  # forward vs spot delta asymmetry only


@pytest.mark.parametrize("preset", ["eurusd", "usdjpy"])
def test_broker_quotes_convert_back_to_smile_quotes(preset):
    m = PRESETS[preset]()
    for ms in m.slices[1:5]:
        sl = _slice(ms, m.S)
        args = (m.S, ms.T, ms.r_d, ms.r_f, ms.convention)
        b25 = broker_strangle_from_smile(sl.smile.vol, ms.quotes.atm, 0.25, *args)
        b10 = broker_strangle_from_smile(sl.smile.vol, ms.quotes.atm, 0.10, *args)
        q = smile_quotes_from_broker(ms.quotes.atm, ms.quotes.rr25, b25,
                                     ms.quotes.rr10, b10, *args)
        # SVI is exact on EURUSD; USDJPY SVI misses the quotes by up to
        # ~4e-4 vol, which bounds how well the smile BF can be recovered.
        tol = 1e-10 if sl.pillar_fit_error < 1e-10 else 5.0 * sl.pillar_fit_error
        assert q.bf25 == pytest.approx(ms.quotes.bf25, abs=tol)
        assert q.bf10 == pytest.approx(ms.quotes.bf10, abs=tol)


def test_broker_and_smile_butterflies_differ_under_skew():
    m = usdjpy_market()
    ms = m.slices[4]
    sl = _slice(ms, m.S)
    b10 = broker_strangle_from_smile(sl.smile.vol, ms.quotes.atm, 0.10, m.S, ms.T,
                                     ms.r_d, ms.r_f, ms.convention)
    assert abs(b10 - ms.quotes.bf10) > 1e-4  # a material fraction of a vol point
