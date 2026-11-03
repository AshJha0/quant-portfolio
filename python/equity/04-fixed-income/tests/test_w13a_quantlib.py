"""W13a: schedules, overnight / term / fixed legs and swaps against QuantLib 1.44.

QuantLib is an independent implementation (C++, its own calendars and
coupon pricers); agreement to 1e-12 on rates and 1e-8 of notional on swap
NPVs is the acceptance bar of the plan's W13 section.
"""

from __future__ import annotations

import datetime as dt
import math

import pytest

ql = pytest.importorskip("QuantLib")

from fi_rates.fixings import ESTR, EURIBOR_6M, SOFR, SONIA, FixingStore  # noqa: E402
from fi_rates.legs import FixedLeg, FlatCurve, IborLeg, OvernightLeg, Swap  # noqa: E402
from fi_rates.schedule import make_schedule  # noqa: E402

QL_CAL = {
    "USD-GOVT": ql.UnitedStates(ql.UnitedStates.GovernmentBond),
    "USD": ql.UnitedStates(ql.UnitedStates.FederalReserve),
    "EUR": ql.TARGET(),
    "GBP": ql.UnitedKingdom(ql.UnitedKingdom.Exchange),
}
QL_CONV = {"modified_following": ql.ModifiedFollowing, "following": ql.Following,
           "unadjusted": ql.Unadjusted, "preceding": ql.Preceding}


def qd(d: dt.date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)


def pd_(d: ql.Date) -> dt.date:
    return dt.date(d.year(), d.month(), d.dayOfMonth())


def ql_schedule(eff, term, months, cal, conv="modified_following", backward=True, eom=False):
    return ql.Schedule(qd(eff), qd(term), ql.Period(months, ql.Months), QL_CAL[cal],
                       QL_CONV[conv], QL_CONV[conv],
                       ql.DateGeneration.Backward if backward else ql.DateGeneration.Forward, eom)


SCHEDULE_CASES = [
    (dt.date(2025, 1, 15), dt.date(2030, 1, 15), 12, "USD", True, False),
    (dt.date(2025, 1, 15), dt.date(2027, 6, 20), 3, "USD-GOVT", True, False),
    (dt.date(2025, 1, 15), dt.date(2027, 6, 20), 3, "USD-GOVT", False, False),
    (dt.date(2024, 2, 29), dt.date(2029, 2, 28), 6, "EUR", True, True),
    (dt.date(2024, 8, 31), dt.date(2026, 2, 28), 3, "GBP", True, True),
    (dt.date(2025, 3, 31), dt.date(2035, 3, 30), 6, "EUR", True, False),
    (dt.date(2024, 12, 24), dt.date(2025, 12, 26), 1, "GBP", False, False),
]


@pytest.mark.parametrize("eff,term,months,cal,backward,eom", SCHEDULE_CASES)
def test_schedule_dates_match_quantlib(eff, term, months, cal, backward, eom):
    ours = make_schedule(eff, term, months, calendar=cal,
                         stub="short_front" if backward else "short_back", end_of_month=eom)
    theirs = [pd_(d) for d in ql_schedule(eff, term, months, cal, backward=backward, eom=eom)]
    assert ours.dates == theirs


def _ql_ois_index(name, curve_rate, ref):
    h = ql.YieldTermStructureHandle(ql.FlatForward(qd(ref), curve_rate, ql.Actual365Fixed(),
                                                   ql.Continuous))
    return {"SOFR": ql.Sofr, "ESTR": ql.Estr, "SONIA": ql.Sonia}[name](h)


def _history(index, cal_name, start, end, base):
    """Deterministic fake fixings: base + 1bp * (day of year mod 7)."""
    cal = QL_CAL[cal_name]
    store, d = {}, start
    while d < end:
        if cal.isBusinessDay(qd(d)):
            store[d] = base + 1e-4 * (d.timetuple().tm_yday % 7)
        d += dt.timedelta(days=1)
    return store


OVERNIGHT_CASES = [
    # index, calendar, lookback, lockout, shift
    (SOFR, "USD-GOVT", 0, 0, False),
    (SOFR, "USD-GOVT", 2, 0, False),
    (SOFR, "USD-GOVT", 2, 0, True),
    (SOFR, "USD-GOVT", 0, 2, False),
    (SOFR, "USD-GOVT", 5, 0, True),
    (ESTR, "EUR", 0, 0, False),
    (ESTR, "EUR", 2, 0, True),
    (SONIA, "GBP", 5, 0, False),
    (SONIA, "GBP", 5, 0, True),
]


@pytest.mark.parametrize("index,cal,lookback,lockout,shift", OVERNIGHT_CASES)
@pytest.mark.parametrize("val", [dt.date(2025, 2, 3), dt.date(2025, 3, 12), dt.date(2025, 5, 2)])
def test_overnight_coupon_matches_quantlib(index, cal, lookback, lockout, shift, val):
    start, end, pay = dt.date(2025, 2, 3), dt.date(2025, 5, 5), dt.date(2025, 5, 7)
    ql.Settings.instance().evaluationDate = qd(val)
    qidx = _ql_ois_index(index.name, 0.04, val)
    qidx.clearFixings()
    hist = _history(index, cal, dt.date(2025, 1, 1), val, 0.043)
    store = FixingStore()
    store.add_many(index.name, hist)
    for d, r in hist.items():
        qidx.addFixing(qd(d), r, True)
    sched = make_schedule(start, end, 0, calendar=cal)
    start, end = sched[0].start, sched[0].end  # 5 May 2025 is a UK bank holiday
    cpn = ql.OvernightIndexedCoupon(qd(pay), 1e7, qd(start), qd(end), qidx, 1.0, 0.0,
                                    ql.Date(), ql.Date(), qidx.dayCounter(), False,
                                    ql.RateAveraging.Compound, lookback, lockout, shift)
    leg = OvernightLeg(sched, 1e7, index,
                       lookback_days=lookback, lockout_days=lockout, observation_shift=shift)
    c = leg.coupons(FlatCurve(0.04, val), store, val)[0]
    assert c.rate == pytest.approx(cpn.rate(), abs=1e-13)
    assert c.amount == pytest.approx(cpn.amount(), abs=1e-6)
    qidx.clearFixings()


def test_ois_swap_npv_matches_quantlib_to_1e8_of_notional():
    """A seasoned 2y SOFR OIS (pay fixed annual ACT/360, lag 2) valued mid-period."""
    val = dt.date(2025, 6, 18)
    ql.Settings.instance().evaluationDate = qd(val)
    notional, k = 1e8, 0.0395
    eff, term = dt.date(2025, 3, 14), dt.date(2027, 3, 15)
    qidx = _ql_ois_index("SOFR", 0.041, val)
    qidx.clearFixings()
    hist = _history(SOFR, "USD-GOVT", dt.date(2025, 1, 1), val, 0.0430)
    store = FixingStore()
    store.add_many("SOFR", hist)
    for d, r in hist.items():
        qidx.addFixing(qd(d), r, True)
    sched = make_schedule(eff, term, 12, calendar="USD-GOVT", payment_lag=2)
    swap = Swap(FixedLeg(sched, notional, k, "ACT/360"), OvernightLeg(sched, notional, SOFR))
    disc = FlatCurve(0.0405, val)
    ours = swap.npv(FlatCurve(0.041, val), disc, store, val)

    qsched = ql_schedule(eff, term, 12, "USD-GOVT")
    qswap = ql.OvernightIndexedSwap(ql.Swap.Payer, notional, qsched, k, ql.Actual360(), qidx,
                                    0.0, 2, ql.ModifiedFollowing, QL_CAL["USD-GOVT"])
    dh = ql.YieldTermStructureHandle(ql.FlatForward(qd(val), 0.0405, ql.Actual365Fixed(),
                                                    ql.Continuous))
    qswap.setPricingEngine(ql.DiscountingSwapEngine(dh))
    assert abs(ours - qswap.NPV()) <= 1e-8 * notional, (ours, qswap.NPV())
    par = swap.par_rate(FlatCurve(0.041, val), disc, store, val)
    assert par == pytest.approx(qswap.fairRate(), abs=1e-12)
    qidx.clearFixings()


@pytest.mark.parametrize("val", [dt.date(2025, 1, 10), dt.date(2025, 4, 22)])
def test_euribor_leg_matches_quantlib(val):
    ql.Settings.instance().evaluationDate = qd(val)
    h = ql.YieldTermStructureHandle(ql.FlatForward(qd(val), 0.025, ql.Actual365Fixed(),
                                                   ql.Continuous))
    qidx = ql.Euribor6M(h)
    qidx.clearFixings()
    store = FixingStore()
    d = dt.date(2024, 12, 1)
    while d <= val:
        if QL_CAL["EUR"].isBusinessDay(qd(d)):
            r = 0.027 + 1e-4 * (d.day % 5)
            qidx.addFixing(qd(d), r)
            store.add("EURIBOR-6M", d, r)
        d += dt.timedelta(days=1)
    eff, term = dt.date(2025, 1, 15), dt.date(2028, 1, 17)
    sched = make_schedule(eff, term, 6, calendar="EUR")
    leg = IborLeg(sched, 5e7, EURIBOR_6M, spread=0.001)
    ours = leg.coupons(FlatCurve(0.025, val), store, val)
    # Our forecast spans the index's own deposit period (an "indexed"
    # coupon); QuantLib's default par-coupon approximation spans the accrual
    # period instead, so switch it to indexed coupons for the comparison.
    ql.IborCoupon.createIndexedCoupons()
    qleg = ql.IborLeg([5e7], ql_schedule(eff, term, 6, "EUR"), qidx, ql.Actual360(),
                      ql.ModifiedFollowing, [], [], [0.001])
    for c, q in zip(ours, qleg, strict=True):
        q = ql.as_floating_rate_coupon(q)
        assert c.rate == pytest.approx(q.rate(), abs=1e-13)
        assert c.amount == pytest.approx(q.amount(), abs=1e-6)
    ql.IborCoupon.createAtParCoupons()
    qidx.clearFixings()


def test_cross_check_flat_curve_is_quantlibs():
    val = dt.date(2025, 1, 2)
    t = ql.FlatForward(qd(val), 0.04, ql.Actual365Fixed(), ql.Continuous)
    for d in (dt.date(2025, 7, 1), dt.date(2030, 1, 2)):
        assert FlatCurve(0.04, val).df(d) == pytest.approx(t.discount(qd(d)), rel=1e-15)
        assert math.isfinite(FlatCurve(0.04, val).df(d))
