"""Optional outside check: ISDA CDS legs against QuantLib's IsdaCdsEngine.

Skipped unless QuantLib is installed (it is not a CI dependency):
``pip install QuantLib``. QuantLib 1.44, engine settings NoBias / Taylor /
Piecewise (and HalfDayBias, the default here), weekends-only calendar, flat continuous rate and flat hazard,
both ACT/365F. Tolerances are what the two implementations achieve: the
protection legs agree to 1e-12 relative or better; the risky annuity and
the par spread to 1e-7 relative (about 1e-8 to 8e-8 seen), because the
accrual-on-default terms differ slightly; the trade-date value to 1e-8 (coupon times that annuity gap).
"""

from datetime import date

import numpy as np
import pytest
from eq_credit.cds import HazardCurve
from eq_credit.isda_cds import IsdaCds, LogLinearCurve, legs, par_spread

ql = pytest.importorskip("QuantLib")

CASES = [
    # trade date, maturity, coupon, recovery, flat rate, flat hazard
    (date(2026, 11, 2), date(2031, 12, 20), 0.01, 0.4, 0.03, 0.02),
    (date(2026, 11, 2), date(2028, 12, 20), 0.05, 0.4, 0.01, 0.05),
    (date(2027, 3, 17), date(2034, 6, 20), 0.01, 0.25, 0.045, 0.01),
]


def _quantlib(td: date, mat: date, coupon: float, rec: float, r: float, lam: float, bias: bool = False):
    d = ql.Date(td.day, td.month, td.year)
    ql.Settings.instance().evaluationDate = d
    yc = ql.YieldTermStructureHandle(ql.FlatForward(d, r, ql.Actual365Fixed()))
    hc = ql.DefaultProbabilityTermStructureHandle(
        ql.FlatHazardRate(d, ql.QuoteHandle(ql.SimpleQuote(lam)), ql.Actual365Fixed()))
    sched = ql.Schedule(d, ql.Date(mat.day, mat.month, mat.year), ql.Period(ql.Quarterly),
                        ql.WeekendsOnly(), ql.Following, ql.Unadjusted,
                        ql.DateGeneration.CDS2015, False)
    cds = ql.CreditDefaultSwap(ql.Protection.Buyer, 1.0, coupon, sched, ql.Following,
                               ql.Actual360(), True, True, d + 1, ql.FaceValueClaim(),
                               ql.Actual360(True), True, d, 3)
    cds.setPricingEngine(ql.IsdaCdsEngine(hc, rec, yc, False, ql.IsdaCdsEngine.Taylor,
                                          ql.IsdaCdsEngine.HalfDayBias if bias else ql.IsdaCdsEngine.NoBias,
                                          ql.IsdaCdsEngine.Piecewise))
    return cds, yc


@pytest.mark.parametrize("bias", [False, True], ids=["NoBias", "HalfDayBias"])
@pytest.mark.parametrize(("td", "mat", "coupon", "rec", "r", "lam"), CASES)
def test_isda_cds_matches_quantlib(td, mat, coupon, rec, r, lam, bias):
    cds = IsdaCds(td, mat, coupon, rec, half_day_bias=bias)
    hz = HazardCurve(np.array([50.0]), np.array([lam]))
    disc = LogLinearCurve(np.array([50.0]), np.array([r]))
    ours = legs(cds, hz, disc)
    q, yc = _quantlib(td, mat, coupon, rec, r, lam, bias)

    assert ours["protection"] == pytest.approx(q.defaultLegNPV(), rel=1e-12, abs=1e-15)
    assert ours["rpv01"] == pytest.approx(-q.couponLegBPS() / 1e-4, rel=1e-7)
    assert par_spread(cds, hz, disc) == pytest.approx(q.fairSpread(), rel=1e-7)
    # QuantLib's NPV is at the trade date; ours is the clean cash amount at T+3.
    cs = cds.cash_settle
    df_cash = yc.discount(ql.Date(cs.day, cs.month, cs.year))
    assert ours["clean_upfront"] * df_cash == pytest.approx(q.NPV(), abs=1e-8)
