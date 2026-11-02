"""Optional outside check: Hull-White against QuantLib's analytic HullWhite model.

Skipped unless QuantLib is installed (it is not a CI dependency):
``pip install QuantLib``. QuantLib 1.44, flat continuously compounded curve
on ``SimpleDayCounter`` (whole years between same-day dates, so QuantLib's
times equal ours exactly), annual fixed leg, single curve. Tolerances are
what the two implementations achieve: zero-bond options agree to 1e-15
absolute (both evaluate Brigo-Mercurio eq. 3.40-3.41 in closed form);
Jamshidian swaptions to 1e-7 relative (about 1e-13 to 4e-8 seen), the gap
being the tolerance of QuantLib's critical-rate root search.
"""

import numpy as np
import pytest
from fi_rates.curve import DiscountCurve
from fi_rates.hull_white import HullWhite

ql = pytest.importorskip("QuantLib")

# flat rate, a, sigma
MODELS = [(0.03, 0.1, 0.01), (0.045, 0.03, 0.015), (-0.005, 0.5, 0.008)]
# expiry, tenor (years), strike offset from the flat rate, payer
SWAPTIONS = [(1, 5, 0.002, True), (5, 5, 0.0, False), (2, 10, -0.01, True)]


def _setup(r: float, a: float, sigma: float):
    d0 = ql.Date(15, 1, 2026)
    ql.Settings.instance().evaluationDate = d0
    yc = ql.YieldTermStructureHandle(
        ql.FlatForward(d0, r, ql.SimpleDayCounter(), ql.Continuous))
    t = np.arange(1, 41) * 0.5
    ours = HullWhite(a, sigma, DiscountCurve(t, np.exp(-r * t)))
    return d0, yc, ql.HullWhite(yc, a, sigma), ours


@pytest.mark.parametrize(("r", "a", "sigma"), MODELS)
def test_zero_bond_option_matches_quantlib(r, a, sigma):
    _, _, m, hw = _setup(r, a, sigma)
    k = float(np.exp(-0.12))
    assert hw.bond_option(1, 5, k, True) == pytest.approx(
        m.discountBondOption(ql.Option.Call, k, 1, 5), rel=0, abs=1e-15)
    assert hw.bond_option(2, 7, 0.99, False) == pytest.approx(
        m.discountBondOption(ql.Option.Put, 0.99, 2, 7), rel=0, abs=1e-15)


@pytest.mark.parametrize(("r", "a", "sigma"), MODELS)
@pytest.mark.parametrize(("ex", "ten", "dk", "payer"), SWAPTIONS)
def test_jamshidian_swaption_matches_quantlib(r, a, sigma, ex, ten, dk, payer):
    d0, yc, m, hw = _setup(r, a, sigma)
    k = r + dk
    dc = ql.SimpleDayCounter()
    sched = ql.Schedule(d0 + ql.Period(ex, ql.Years), d0 + ql.Period(ex + ten, ql.Years),
                        ql.Period(1, ql.Years), ql.NullCalendar(), ql.Unadjusted,
                        ql.Unadjusted, ql.DateGeneration.Forward, False)
    swap = ql.VanillaSwap(ql.Swap.Payer if payer else ql.Swap.Receiver, 1.0, sched, k, dc,
                          sched, ql.Euribor6M(yc), 0.0, dc)
    swpt = ql.Swaption(swap, ql.EuropeanExercise(d0 + ql.Period(ex, ql.Years)))
    swpt.setPricingEngine(ql.JamshidianSwaptionEngine(m, yc))
    assert hw.swaption(ex, ten, k, 1, payer) == pytest.approx(swpt.NPV(), rel=1e-7)
