"""Tests for qp_core.calendars and qp_core.daycount.

Expected values are worked by hand from the convention definitions.
"""

import calendar as pycal
import datetime as dt

import pytest
from hypothesis import given
from hypothesis import strategies as hst

from qp_core.calendars import CALENDARS, ROLL_CONVENTIONS, get_calendar, joint_calendar
from qp_core.daycount import (
    MONEY_MARKET_DAYCOUNT,
    add_months,
    adjust_modified_following,
    generate_schedule,
    is_month_end,
    last_business_day,
    money_market_year_fraction,
    roll_months,
    year_fraction,
)

D = dt.date
dates = hst.dates(min_value=D(1990, 1, 1), max_value=D(2090, 12, 31))


class TestYearFraction:
    def test_act_conventions(self):
        assert year_fraction(D(2024, 1, 1), D(2025, 1, 1), "ACT/365F") == 366 / 365
        assert year_fraction(D(2024, 1, 1), D(2024, 7, 1), "ACT/360") == 182 / 360

    def test_thirty360_rules(self):
        # d1 = 31 -> 30; d2 = 31 -> 30 because d1 is then 30: 6 months exactly.
        assert year_fraction(D(2024, 1, 31), D(2024, 7, 31), "30/360US") == 0.5
        # d2 = 31 stays when d1 < 30: 30 * 6 + 1 days.
        assert year_fraction(D(2024, 1, 15), D(2024, 7, 31), "30/360US") == (180 + 16) / 360
        # No end-of-February rule: Feb 29 is day 29.
        assert year_fraction(D(2024, 2, 29), D(2024, 8, 31), "30/360US") == (180 + 2) / 360

    @given(dates, dates)
    def test_thirty360_alias(self, a, b):
        assert year_fraction(a, b, "30/360") == year_fraction(a, b, "30/360US")

    def test_act_act_isda_split_by_year(self):
        # 2023-07-01 .. 2024-07-01: 184 days of 2023 / 365 + 182 days of 2024 / 366.
        got = year_fraction(D(2023, 7, 1), D(2024, 7, 1), "ACT/ACT-ISDA")
        assert got == pytest.approx(184 / 365 + 182 / 366, rel=0, abs=1e-15)

    def test_act_act_icma_regular_coupon(self):
        got = year_fraction(
            D(2024, 2, 15), D(2024, 8, 15), "ACT/ACT-ICMA",
            ref_start=D(2024, 2, 15), ref_end=D(2024, 8, 15), frequency=2,
        )
        assert got == 0.5

    def test_icma_needs_reference(self):
        with pytest.raises(ValueError, match="ACT/ACT-ICMA needs"):
            year_fraction(D(2024, 1, 1), D(2024, 7, 1), "ACT/ACT-ICMA")

    def test_unknown_convention(self):
        with pytest.raises(ValueError, match="Unknown day count"):
            year_fraction(D(2024, 1, 1), D(2024, 7, 1), "BUS/252")

    @given(dates, dates)
    def test_sign_rule(self, a, b):
        fwd = year_fraction(a, b, "ACT/360")
        assert year_fraction(b, a, "ACT/360") == -fwd
        if b < a:
            with pytest.raises(ValueError, match="precedes"):
                year_fraction(a, b, "ACT/360", allow_negative=False)

    @given(dates, dates, dates)
    def test_isda_additive(self, a, b, c):
        # ACT/ACT ISDA counts each day once at its own year's weight, so it
        # adds over adjacent periods (up to rounding).
        ab = year_fraction(a, b, "ACT/ACT-ISDA")
        bc = year_fraction(b, c, "ACT/ACT-ISDA")
        assert year_fraction(a, c, "ACT/ACT-ISDA") == pytest.approx(ab + bc, abs=1e-12)

    def test_isda_whole_calendar_years(self):
        assert year_fraction(D(2023, 1, 1), D(2027, 1, 1), "ACT/ACT-ISDA") == 4.0

    def test_money_market(self):
        assert MONEY_MARKET_DAYCOUNT["GBP"] == "ACT/365F"
        assert money_market_year_fraction(D(2024, 1, 1), D(2024, 4, 1), "usd") == 91 / 360
        with pytest.raises(ValueError, match="no money-market convention"):
            money_market_year_fraction(D(2024, 1, 1), D(2024, 4, 1), "XAU")
        with pytest.raises(ValueError, match="precedes"):
            money_market_year_fraction(D(2024, 4, 1), D(2024, 1, 1), "EUR")


class TestMonths:
    def test_add_months_clamps(self):
        assert add_months(D(2024, 1, 31), 1) == D(2024, 2, 29)
        assert add_months(D(2023, 1, 31), 1) == D(2023, 2, 28)
        assert add_months(D(2024, 3, 15), -15) == D(2022, 12, 15)

    @given(dates, hst.integers(-240, 240))
    def test_add_months_month_index(self, d, k):
        out = add_months(d, k)
        assert (out.year * 12 + out.month) - (d.year * 12 + d.month) == k
        assert out.day == min(d.day, pycal.monthrange(out.year, out.month)[1])

    def test_roll_months_eom(self):
        assert roll_months(D(2025, 2, 28), 6, eom=True) == D(2025, 8, 31)
        assert roll_months(D(2025, 2, 28), 6, eom=False) == D(2025, 8, 28)
        assert is_month_end(D(2024, 2, 29)) and not is_month_end(D(2024, 2, 28))


class TestCalendars:
    def test_rolls(self):
        usd = CALENDARS["USD"]
        sat = D(2024, 8, 31)  # Saturday, month end
        # Labor Day (2 Sep) is a floating holiday, not in the partial list.
        assert usd.adjust(sat, "following") == D(2024, 9, 2)
        assert usd.adjust(sat, "preceding") == D(2024, 8, 30)
        assert usd.adjust(sat, "modified_following") == D(2024, 8, 30)
        assert usd.adjust(sat, "unadjusted") == sat
        with pytest.raises(ValueError, match="roll convention"):
            usd.adjust(sat, "nearest")
        assert set(ROLL_CONVENTIONS) == {"following", "modified_following", "preceding", "unadjusted"}

    def test_fixed_holidays_and_joint(self):
        assert not CALENDARS["USD"].is_business_day(D(2024, 7, 4))
        joint = get_calendar("EUR+USD")
        assert joint == joint_calendar("EUR", "USD")
        assert not joint.is_business_day(D(2024, 5, 1))
        assert not joint.is_business_day(D(2024, 7, 4))
        with pytest.raises(ValueError, match="unknown calendar"):
            get_calendar("XXX")

    def test_business_day_arithmetic(self):
        wk = CALENDARS["WEEKEND"]
        assert wk.add_business_days(D(2024, 1, 5), 1) == D(2024, 1, 8)
        assert wk.business_days_between(D(2024, 1, 5), D(2024, 1, 12)) == 5
        with pytest.raises(ValueError):
            wk.add_business_days(D(2024, 1, 5), -1)

    @given(dates)
    def test_adjusted_is_business_day(self, d):
        for name in ("WEEKEND", "USD", "EUR", "GBP", "JPY"):
            cal = CALENDARS[name]
            for conv in ("following", "modified_following", "preceding"):
                out = cal.adjust(d, conv)
                assert cal.is_business_day(out)
                assert abs((out - d).days) <= 7
            assert cal.adjust(d, "modified_following").month == d.month

    def test_last_business_day(self):
        assert last_business_day(2024, 8, CALENDARS["USD"]) == D(2024, 8, 30)


class TestSchedule:
    def test_semiannual_with_stub(self):
        got = generate_schedule(D(2024, 3, 1), D(2026, 6, 15), 2)
        assert got == [D(2024, 6, 15), D(2024, 12, 15), D(2025, 6, 15), D(2025, 12, 15), D(2026, 6, 15)]

    def test_eom(self):
        got = generate_schedule(D(2024, 2, 28), D(2025, 2, 28), 2)
        assert got == [D(2024, 2, 29), D(2024, 8, 31), D(2025, 2, 28)]

    def test_adjusted(self):
        # 2026-02-28 is a Saturday: modified following would cross into March, so it rolls back.
        got = generate_schedule(D(2025, 2, 28), D(2026, 2, 28), 1, adjust=True, eom=False)
        assert got == [D(2026, 2, 27)]
        assert adjust_modified_following(D(2026, 2, 28)) == D(2026, 2, 27)

    def test_bad_inputs(self):
        with pytest.raises(ValueError, match="frequency"):
            generate_schedule(D(2024, 1, 1), D(2025, 1, 1), 3)
        with pytest.raises(ValueError, match="must be after"):
            generate_schedule(D(2025, 1, 1), D(2024, 1, 1), 2)

    @given(dates, hst.integers(1, 30), hst.sampled_from([1, 2, 4]))
    def test_schedule_increasing_and_ends_at_maturity(self, eff, years, freq):
        mat = add_months(eff, 12 * years + 1)
        got = generate_schedule(eff, mat, freq)
        assert got[-1] == mat
        assert all(a < b for a, b in zip(got, got[1:]))
        assert got[0] > eff
