"""Float-float cross-currency swaps: constant notional and MtM reset."""

import pytest
from fx_rates.curve import DiscountCurve
from fx_rates.fxforward import MarketState
from fx_rates.xccy import BasisSwap
from fx_rates.xccy_float import FloatFloatXccy, XccyCurves, par_base_spread

TIMES = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


def _curve(z0, slope):
    return DiscountCurve.from_zero_rates(TIMES, [z0 + slope * t for t in TIMES])


EUR_PROJ, EUR_DISC = _curve(0.025, 0.001), _curve(0.022, 0.0012)
USD_PROJ, USD_DISC = _curve(0.045, -0.001), _curve(0.043, -0.0009)
C = XccyCurves(1.085, EUR_PROJ, EUR_DISC, USD_PROJ, USD_DISC)


def test_single_curve_zero_spread_legs_are_par():
    """Projection = discount and zero spread -> each leg (with exchanges) is worth 0."""
    c = XccyCurves(1.1, EUR_DISC, EUR_DISC, USD_DISC, USD_DISC)
    for mtm in (False, True):
        s = FloatFloatXccy(1e6, 5.0, 4, mtm_reset=mtm)
        assert s.base_leg_pv(c) == pytest.approx(0.0, abs=1e-8)
        assert s.quote_leg_pv(c) == pytest.approx(0.0, abs=1e-8)


def test_constant_notional_matches_existing_basis_swap():
    """Independent code path: fx_rates.xccy.BasisSwap on a basis-quoted market."""
    m = MarketState(1.085, USD_PROJ, EUR_PROJ, basis_spreads=((2.0, -0.002), (5.0, -0.0025),
                                                               (10.0, -0.003)),
                    pair=("EUR", "USD"))
    c = XccyCurves(m.spot, m.foreign_curve, m.foreign_curve_adjusted,
                   m.domestic_curve, m.domestic_curve_adjusted)
    for s in (-0.003, 0.0, 0.001):
        ours = FloatFloatXccy(1e6, 5.0, 4, spread_base=s).value(c)
        ref = BasisSwap(1e6, 1e6 * m.spot, s, 5.0, 4, include_initial_exchange=True).value(m)
        assert ours == pytest.approx(ref, abs=1e-6)


def test_one_period_mtm_equals_constant():
    a = FloatFloatXccy(1e6, 0.25, 4, spread_base=0.001, mtm_reset=False).value(C)
    b = FloatFloatXccy(1e6, 0.25, 4, spread_base=0.001, mtm_reset=True).value(C)
    assert a == pytest.approx(b, abs=1e-9)


def test_mtm_at_par_has_no_fx_delta():
    """MtM value is proportional to spot, so at par its FX delta is zero; the
    constant-notional swap at its own par keeps an FX delta."""
    def bumped(spot):
        return XccyCurves(spot, EUR_PROJ, EUR_DISC, USD_PROJ, USD_DISC)

    for mtm, expect_zero in ((True, True), (False, False)):
        s0 = FloatFloatXccy(1e6, 5.0, 4, mtm_reset=mtm, notional_quote=1e6 * C.spot)
        sp = par_base_spread(s0, C)
        s = FloatFloatXccy(1e6, 5.0, 4, spread_base=sp, mtm_reset=mtm,
                           notional_quote=1e6 * C.spot)
        assert s.value(C) == pytest.approx(0.0, abs=1e-7)
        delta = (s.value(bumped(1.095)) - s.value(bumped(1.075))) / 0.02
        if expect_zero:
            assert abs(delta) < 1e-6
        else:
            assert abs(delta) > 1.0


def test_receive_pay_symmetry_and_validation():
    a = FloatFloatXccy(1e6, 3.0, 2, 0.001, 0.0, True, True).value(C)
    b = FloatFloatXccy(1e6, 3.0, 2, 0.001, 0.0, True, False).value(C)
    assert a == pytest.approx(-b, abs=1e-12)
    with pytest.raises(ValueError):
        FloatFloatXccy(1e6, 1.1, 4)
    with pytest.raises(ValueError):
        XccyCurves(-1.0, EUR_PROJ, EUR_DISC, USD_PROJ, USD_DISC)
