"""W13a: schedules, fixings and legs without QuantLib (properties and hand checks)."""

from __future__ import annotations

import datetime as dt
import math

import pytest
from fi_rates.fixings import (
    ESTR,
    EURIBOR_6M,
    SOFR,
    SONIA,
    FixingStore,
    MissingFixingError,
    OvernightIndex,
)
from fi_rates.legs import (
    FixedLeg,
    FlatCurve,
    IborLeg,
    OvernightLeg,
    Swap,
    compounded_rate,
    futures_convexity_adjustment,
)
from fi_rates.schedule import make_schedule, shift_business_days
from qp_core.holidays import get_calendar

D = dt.date


# ------------------------------------------------------------------ schedule
def test_payment_lag_rolls_over_holiday():
    s = make_schedule(D(2025, 1, 15), D(2026, 1, 15), 12, calendar="USD", payment_lag=2)
    assert len(s) == 1
    assert s[0].payment == D(2026, 1, 20)  # 19 Jan 2026 is Martin Luther King Day


@pytest.mark.parametrize("stub,first,last", [
    ("short_front", (D(2025, 1, 15), D(2025, 3, 20)), (D(2026, 12, 21), D(2027, 3, 22))),
    ("long_front", (D(2025, 1, 15), D(2025, 6, 20)), (D(2026, 12, 21), D(2027, 3, 22))),
    ("short_back", (D(2025, 1, 15), D(2025, 4, 15)), (D(2027, 1, 15), D(2027, 3, 22))),
    ("long_back", (D(2025, 1, 15), D(2025, 4, 15)), (D(2026, 10, 15), D(2027, 3, 22))),
])
def test_stub_types(stub, first, last):
    s = make_schedule(D(2025, 1, 15), D(2027, 3, 20), 3, calendar="USD", stub=stub)
    assert (s[0].start, s[0].end) == first
    assert (s[-1].start, s[-1].end) == last
    stub_period = s[0] if stub.endswith("front") else s[-1]
    assert stub_period.is_stub
    assert sum(p.is_stub for p in s) == 1
    # Contiguous and increasing.
    for a, b in zip(s, s[1:], strict=False):
        assert a.end == b.start and a.start < a.end


def test_no_stub_when_dates_align():
    s = make_schedule(D(2025, 3, 20), D(2027, 3, 20), 3, calendar="USD")
    assert not any(p.is_stub for p in s) and len(s) == 8


def test_end_of_month_rolls():
    s = make_schedule(D(2024, 2, 29), D(2025, 2, 28), 3, calendar="WEEKEND", end_of_month=True,
                      convention="unadjusted")
    assert [p.unadjusted_end for p in s] == [D(2024, 5, 31), D(2024, 8, 31), D(2024, 11, 30),
                                             D(2025, 2, 28)]


def test_shift_business_days_both_ways():
    cal = get_calendar("USD")
    assert shift_business_days(cal, D(2025, 1, 17), 1) == D(2025, 1, 21)  # MLK Monday
    assert shift_business_days(cal, D(2025, 1, 21), -1) == D(2025, 1, 17)
    assert shift_business_days(cal, D(2025, 1, 20), 0) == D(2025, 1, 20)


@pytest.mark.parametrize("kw", [
    dict(effective=D(2025, 1, 1), termination=D(2025, 1, 1)),
    dict(stub="middle"),
    dict(tenor_months=-3),
    dict(payment_lag=-1),
    dict(roll_day=40),
])
def test_schedule_rejects_bad_input(kw):
    args = dict(effective=D(2025, 1, 15), termination=D(2026, 1, 15), tenor_months=3)
    args.update(kw)
    with pytest.raises(ValueError):
        make_schedule(args.pop("effective"), args.pop("termination"), args.pop("tenor_months"),
                      **args)


# ------------------------------------------------------------------ fixings
def test_store_refuses_conflicts_and_nan():
    s = FixingStore()
    s.add("SOFR", D(2024, 1, 2), 0.0531)
    s.add("SOFR", D(2024, 1, 2), 0.0531)  # same value: fine
    with pytest.raises(ValueError):
        s.add("SOFR", D(2024, 1, 2), 0.0530)
    with pytest.raises(ValueError):
        s.add("SOFR", D(2024, 1, 3), float("nan"))
    with pytest.raises(MissingFixingError):
        s.get("SOFR", D(2024, 1, 3))


def test_store_from_csv_percent(tmp_path):
    p = tmp_path / "sofr.csv"
    p.write_text("date,rate\n2024-01-02,5.40\n2024-01-03,5.38\n")
    s = FixingStore.from_csv(p, "SOFR")
    assert s.get("SOFR", D(2024, 1, 3)) == pytest.approx(0.0538, abs=1e-15)


def test_index_conventions():
    assert (SOFR.basis, ESTR.basis, SONIA.basis) == (360.0, 360.0, 365.0)
    assert EURIBOR_6M.fixing_date(D(2025, 1, 15)) == D(2025, 1, 13)
    assert EURIBOR_6M.maturity(D(2025, 1, 15)) == D(2025, 7, 15)
    with pytest.raises(ValueError):
        OvernightIndex("X", "USD", "USD", "30/360")


# ------------------------------------------------------------------ overnight
def _flat_history(index_name, cal, start, end, rate):
    s = FixingStore()
    d = start
    while d < end:
        if cal.is_business_day(d):
            s.add(index_name, d, rate)
        d += dt.timedelta(days=1)
    return s


def test_constant_fixings_compound_by_hand():
    """1-week period at a constant 5%: R = ((1 + .05/360)^4 (1 + .05*3/360) - 1) 360/7.

    Mon 3 Mar 2025 to Mon 10 Mar: four one-day weights and the Friday's
    three-day weight.
    """
    cal = get_calendar("USD-GOVT")
    store = _flat_history("SOFR", cal, D(2025, 2, 1), D(2025, 3, 11), 0.05)
    r, share = compounded_rate(SOFR, D(2025, 3, 3), D(2025, 3, 10), None, store, D(2025, 3, 11))
    want = ((1 + 0.05 / 360) ** 4 * (1 + 0.05 * 3 / 360) - 1) * 360 / 7
    assert r == pytest.approx(want, rel=1e-13) and share == 1.0


def test_forecast_telescopes_to_discount_ratio():
    """No lookback or lockout: the forecast product equals P(S) / P(E)."""
    val = D(2025, 1, 2)
    curve = FlatCurve(0.043, val)
    s, e = D(2025, 2, 3), D(2025, 8, 4)
    r, share = compounded_rate(SOFR, s, e, curve, FixingStore(), val)
    want = (curve.df(s) / curve.df(e) - 1) * 360 / (e - s).days
    assert r == pytest.approx(want, rel=1e-13) and share == 0.0


def test_seasoned_swap_values_from_stored_fixings():
    """Valued mid-period: the published part comes from the store. Bumping a
    stored fixing moves the NPV by its accrual weight, and a missing past
    fixing raises instead of falling back to the curve."""
    val = D(2025, 6, 18)
    cal = get_calendar("USD-GOVT")
    store = _flat_history("SOFR", cal, D(2025, 3, 1), val, 0.043)
    sched = make_schedule(D(2025, 3, 14), D(2026, 3, 14), 12, calendar="USD-GOVT", payment_lag=2)
    swap = Swap(FixedLeg(sched, 1e8, 0.04), OvernightLeg(sched, 1e8, SOFR))
    curve = FlatCurve(0.041, val)
    base = swap.npv(curve, curve, store, val)
    c = swap.floating.coupons(curve, store, val)[0]
    assert 0.25 < c.fixed_part < 0.35
    bumped = FixingStore()
    for d, r in store.series("SOFR").items():
        bumped.add("SOFR", d, r + (0.01 if d == D(2025, 4, 1) else 0.0))
    moved = swap.npv(curve, curve, bumped, val) - base
    # 1 Apr 2025 is a one-day weight: the product gains the factor
    # (1 + (r + 1%)/360) / (1 + r/360), i.e. growth * 0.01/360 / (1 + r/360).
    growth = 1.0 + c.rate * c.accrual
    expect = 1e8 * growth * (0.01 / 360) / (1 + 0.043 / 360) * curve.df(sched[0].payment)
    assert moved == pytest.approx(expect, rel=1e-9)
    holes = FixingStore()
    for d, r in store.series("SOFR").items():
        if d != D(2025, 5, 1):
            holes.add("SOFR", d, r)
    with pytest.raises(MissingFixingError):
        swap.npv(curve, curve, holes, val)


def test_par_swap_has_zero_npv_and_legs_use_conventions():
    val = D(2025, 1, 2)
    curve = FlatCurve(0.03, val)
    for idx, cal, dc in ((SOFR, "USD-GOVT", "ACT/360"), (ESTR, "EUR", "ACT/360"),
                         (SONIA, "GBP", "ACT/365F")):
        sched = make_schedule(D(2025, 1, 6), D(2030, 1, 7), 12, calendar=cal, payment_lag=2)
        flt = OvernightLeg(sched, 1e7, idx)
        k = Swap(FixedLeg(sched, 1e7, 0.0, dc), flt).par_rate(curve, curve, FixingStore(), val)
        swap = Swap(FixedLeg(sched, 1e7, k, dc), flt)
        assert abs(swap.npv(curve, curve, FixingStore(), val)) < 1e-6
        # Continuous 3% on ACT/365F is about 3.0% simple ACT/360 * 365/360...
        assert 0.029 < k < 0.032


def test_lockout_and_lookback_validation():
    with pytest.raises(ValueError):
        compounded_rate(SOFR, D(2025, 3, 3), D(2025, 3, 5), None, FixingStore(), D(2025, 4, 1),
                        lockout_days=5)
    with pytest.raises(ValueError):
        compounded_rate(SOFR, D(2025, 3, 3), D(2025, 3, 5), None, FixingStore(), D(2025, 4, 1),
                        lookback_days=-1)
    with pytest.raises(ValueError, match="projection curve"):
        compounded_rate(SOFR, D(2025, 3, 3), D(2025, 3, 5), None, FixingStore(), D(2025, 1, 1))


def test_ibor_leg_uses_past_fixing_and_forecast():
    val = D(2025, 4, 22)
    store = FixingStore()
    store.add("EURIBOR-6M", D(2025, 1, 13), 0.0271)
    sched = make_schedule(D(2025, 1, 15), D(2026, 1, 15), 6, calendar="EUR")
    leg = IborLeg(sched, 1e7, EURIBOR_6M)
    c = leg.coupons(FlatCurve(0.025, val), store, val)
    assert c[0].rate == 0.0271 and c[0].fixed_part == 1.0
    assert c[1].fixed_part == 0.0 and 0.024 < c[1].rate < 0.027


# ------------------------------------------------------------------ futures
def test_convexity_hull_example_and_limit():
    assert futures_convexity_adjustment(8.0, 8.25, 0.012) == pytest.approx(0.004752, abs=1e-15)
    ho_lee = futures_convexity_adjustment(2.0, 2.25, 0.01)
    assert futures_convexity_adjustment(2.0, 2.25, 0.01, 1e-6) == pytest.approx(ho_lee, rel=1e-5)
    # Mean reversion lowers the adjustment.
    assert futures_convexity_adjustment(2.0, 2.25, 0.01, 0.1) < ho_lee
    assert futures_convexity_adjustment(0.0, 0.25, 0.01, 0.05) == 0.0
    with pytest.raises(ValueError):
        futures_convexity_adjustment(1.0, 0.5, 0.01)
    with pytest.raises(ValueError):
        futures_convexity_adjustment(1.0, 1.25, math.nan)
