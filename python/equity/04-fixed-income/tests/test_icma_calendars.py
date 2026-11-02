"""ACT/ACT-ICMA government-bond accrual and business-day calendars.

Reference values are hand-computed from the ICMA rule: a regular coupon is
coupon / frequency; accrued = coupon / frequency * days accrued / days in
the period.
"""

import datetime as dt

import pytest
from fi_rates import bond as B
from fi_rates.calendars import CALENDARS, get_calendar, joint_calendar
from fi_rates.daycount import year_fraction
from hypothesis import given, settings
from hypothesis import strategies as st

D = dt.date


_EFF, _MAT = D(2024, 2, 15), D(2029, 2, 15)


def _ust(coupon=0.04, eff=_EFF, mat=_MAT, dcc="ACT/ACT-ICMA"):
    return B.FixedRateBond(eff, mat, coupon, 2, dcc)


class TestIcmaCoupons:
    def test_regular_coupons_are_exactly_c_over_f(self):
        # ACT/ACT-ISDA gave 1.9836, 1.9891, 2.0123, ... for this bond
        flows = B.bond_cashflows(_ust(), D(2024, 2, 15))
        coupons = [a for d, a in flows[:-1]]
        assert all(c == 2.0 for c in coupons)
        assert flows[-1][1] == 102.0

    def test_isda_differs_in_every_period(self):
        flows = B.bond_cashflows(_ust(dcc="ACT/ACT-ISDA"), D(2024, 2, 15))
        assert all(abs(a - 2.0) > 1e-3 for _, a in flows[:-1])

    def test_par_at_coupon_date_when_yield_equals_coupon(self):
        # ISDA priced this at 99.9965
        bond = _ust()
        assert B.price_from_ytm(bond, D(2025, 2, 15), 0.04) == pytest.approx(100.0, abs=1e-12)

    def test_icma_accrued_feb15_to_may15(self):
        # 89 days of the 181-day period 15-Feb-2025 .. 15-Aug-2025
        acc = B.accrued_interest(_ust(), D(2025, 5, 15))
        assert acc == pytest.approx(2.0 * 89 / 181, rel=1e-15)
        assert acc == pytest.approx(0.9834, abs=1e-4)

    def test_short_front_stub_uses_reference_period(self):
        bond = B.FixedRateBond(D(2024, 3, 1), D(2026, 8, 15), 0.05, 2, "ACT/ACT-ICMA")
        start, end, ref = bond.periods()[0]
        assert (start, end, ref) == (D(2024, 3, 1), D(2024, 8, 15), D(2024, 2, 15))
        cf = B.bond_cashflows(bond, D(2024, 3, 1))[0][1]
        assert cf == pytest.approx(100 * 0.05 / 2 * (D(2024, 8, 15) - D(2024, 3, 1)).days
                                   / (D(2024, 8, 15) - D(2024, 2, 15)).days, rel=1e-15)

    def test_year_fraction_icma_requires_reference(self):
        with pytest.raises(ValueError, match="ref_start"):
            year_fraction(D(2024, 1, 1), D(2024, 7, 1), "ACT/ACT-ICMA")

    def test_unknown_daycount_rejected(self):
        with pytest.raises(ValueError, match="day count"):
            B.FixedRateBond(D(2024, 1, 1), D(2026, 1, 1), 0.04, 2, "ACT/ACT")

    @settings(max_examples=60, deadline=None)
    @given(
        coupon=st.floats(0.0, 0.12),
        years=st.integers(1, 30),
        month=st.integers(1, 12),
        day=st.integers(1, 31),
        k=st.integers(0, 10),
    )
    def test_par_property(self, coupon, years, month, day, k):
        """At y = c on any coupon date an ICMA bond prices at par."""
        import calendar as cal
        # Feb 29 exists in 2020 but not in most maturity years, so cap at 28.
        day = min(day, 28 if month == 2 else cal.monthrange(2020, month)[1])
        eff = D(2020, month, day)
        mat = D(2020 + years, month, day)
        bond = B.FixedRateBond(eff, mat, coupon, 2, "ACT/ACT-ICMA")
        # Settle on a coupon date. The effective date counts only when it starts
        # a regular period: with eom=True a month-end maturity (28 Feb 2021)
        # rolls the first coupon to 29 Feb 2020, so a 28 Feb 2020 effective
        # date opens a one-day front stub, which is not a coupon date and is
        # not priced at par by ICMA (same schedule as QuantLib backward EOM).
        start, _, ref = bond.periods()[0]
        dates = ([eff] if start == ref else []) + bond.schedule()[:-1]
        settle = dates[min(k, len(dates) - 1)]
        assert B.price_from_ytm(bond, settle, coupon) == pytest.approx(100.0, abs=1e-9)


class TestCalendars:
    def test_weekend_and_fixed_holiday(self):
        usd = get_calendar("USD")
        assert not usd.is_business_day(D(2026, 7, 4))   # Saturday anyway
        assert not usd.is_business_day(D(2025, 7, 4))   # Friday holiday
        assert usd.is_business_day(D(2025, 7, 3))

    def test_modified_following_does_not_cross_month(self):
        assert get_calendar("WEEKEND").adjust(D(2024, 8, 31)) == D(2024, 8, 30)
        assert get_calendar("WEEKEND").adjust(D(2024, 6, 1)) == D(2024, 6, 3)

    def test_following_and_preceding(self):
        c = get_calendar("EUR")
        assert c.adjust(D(2025, 12, 25), "following") == D(2025, 12, 29)
        assert c.adjust(D(2025, 12, 25), "preceding") == D(2025, 12, 24)

    def test_add_business_days_skips_holidays(self):
        assert get_calendar("JPY").add_business_days(D(2025, 12, 30), 1) == D(2026, 1, 5)

    def test_joint_calendar_is_union_of_holidays(self):
        j = joint_calendar("EUR", "USD")
        assert not j.is_business_day(D(2025, 5, 1))
        assert not j.is_business_day(D(2025, 7, 4))
        assert get_calendar("EUR+USD").fixed_holidays == j.fixed_holidays

    def test_unknown_calendar_and_roll_rejected(self):
        with pytest.raises(ValueError, match="unknown calendar"):
            get_calendar("XXX")
        with pytest.raises(ValueError, match="roll convention"):
            CALENDARS["USD"].adjust(D(2025, 1, 4), "nearest")
        with pytest.raises(ValueError):
            CALENDARS["USD"].add_business_days(D(2025, 1, 4), -1)

    def test_business_days_between(self):
        assert get_calendar("WEEKEND").business_days_between(D(2025, 1, 3), D(2025, 1, 10)) == 5
