"""W1: one rule-based holiday calendar for schedules, rolls and FX spot.

Each named case below was a wrong answer before round 6. The partial
fixed-date tables in ``qp_core.calendars`` knew no Good Friday, no
Thanksgiving, no weekend substitutes, and kept Juneteenth in years before
it existed. The expected dates come from the published lists:
federalreserve.gov "Holidays observed by the Federal Reserve System",
gov.uk/bank-holidays (and the 2002, 2011 and 2012 proclamations), the
ECB's TARGET2 closing days, and the Cabinet Office list of Japanese
national holidays. ``test_matches_quantlib`` checks every year 2000-2030
against QuantLib as an independent oracle.
"""

from __future__ import annotations

import datetime as dt
import importlib
import pathlib
import re
import sys
import warnings

import pytest
from qp_core import holidays as H
from qp_core.daycount import adjust_modified_following, generate_schedule

D = dt.date


class TestNamedHolidays:
    def test_good_friday_rolls_past_easter_monday_in_london(self):
        # Fri 18-Apr-2025 Good Friday, Mon 21-Apr Easter Monday -> Tue 22-Apr
        assert adjust_modified_following(D(2025, 4, 18), "GBP") == D(2025, 4, 22)

    def test_good_friday_is_a_target_holiday(self):
        assert adjust_modified_following(D(2026, 4, 3), "EUR") == D(2026, 4, 7)

    def test_thanksgiving(self):
        assert adjust_modified_following(D(2025, 11, 27), "USD") == D(2025, 11, 28)

    def test_uk_may_bank_holidays(self):
        cal = H.get_calendar("GBP")
        assert not cal.is_business_day(D(2025, 5, 5))   # early May
        assert not cal.is_business_day(D(2025, 5, 26))  # spring
        assert cal.is_business_day(D(2020, 5, 4))       # 2020: moved to Fri 8 May
        assert not cal.is_business_day(D(2020, 5, 8))

    def test_juneteenth_only_from_2022(self):
        usd = H.get_calendar("USD")
        assert usd.is_business_day(D(2020, 6, 19))      # Friday, not yet a holiday
        assert not usd.is_business_day(D(2023, 6, 19))

    def test_sunday_substitutes(self):
        assert not H.get_calendar("USD").is_business_day(D(2021, 7, 5))
        gbp = H.get_calendar("GBP")
        assert not gbp.is_business_day(D(2021, 12, 27))
        assert not gbp.is_business_day(D(2021, 12, 28))

    def test_fed_does_not_observe_saturday_holidays(self):
        assert H.get_calendar("USD").is_business_day(D(2026, 7, 3))

    def test_uk_proclamations(self):
        gbp = H.get_calendar("GBP")
        assert not gbp.is_business_day(D(2011, 4, 29))  # royal wedding
        assert not gbp.is_business_day(D(2012, 6, 4))   # spring holiday moved
        assert not gbp.is_business_day(D(2012, 6, 5))   # diamond jubilee
        assert gbp.is_business_day(D(2012, 5, 28))
        assert not gbp.is_business_day(D(2002, 6, 3))   # spring holiday moved
        assert not gbp.is_business_day(D(2002, 6, 4))   # golden jubilee
        assert gbp.is_business_day(D(2002, 5, 27))

    def test_target_closed_31_december_2001(self):
        assert not H.get_calendar("TARGET").is_business_day(D(2001, 12, 31))
        assert H.get_calendar("TARGET").is_business_day(D(2002, 12, 31))

    def test_japan_respect_for_aged_day_before_2003(self):
        jpy = H.get_calendar("JPY")
        assert not jpy.is_business_day(D(2000, 9, 15))
        assert jpy.is_business_day(D(2000, 9, 18))
        assert jpy.is_business_day(D(2001, 9, 17))

    def test_schedule_rolls_on_rule_calendar(self):
        # 25-Dec-2027 is a Saturday; the London substitutes are Mon 27 and
        # Tue 28, so modified following lands on Wed 29.
        dates = generate_schedule(D(2025, 12, 25), D(2027, 12, 25), 2, adjust=True,
                                  eom=False, calendar="GBP")
        assert dates[-1] == D(2027, 12, 29)


class TestCalendarObject:
    def test_joint_is_union(self):
        j = H.joint_calendar("EUR", "USD")
        assert j == H.get_calendar("EUR+USD") == H.get_calendar("EURUSD")
        assert not j.is_business_day(D(2025, 5, 1))
        assert not j.is_business_day(D(2025, 7, 4))
        assert j.holidays(2025) == H.holidays("EUR", 2025) | H.holidays("USD", 2025)

    def test_weekend_calendar_and_rolls(self):
        wk = H.get_calendar(None)
        assert wk.name == "WEEKEND"
        assert wk.adjust(D(2024, 8, 31)) == D(2024, 8, 30)
        assert wk.adjust(D(2024, 6, 1), "following") == D(2024, 6, 3)
        assert wk.adjust(D(2024, 6, 1), "preceding") == D(2024, 5, 31)
        assert wk.adjust(D(2024, 6, 1), "unadjusted") == D(2024, 6, 1)
        assert wk.add_business_days(D(2025, 1, 3), 1) == D(2025, 1, 6)
        assert wk.business_days_between(D(2025, 1, 3), D(2025, 1, 10)) == 5

    def test_rejects_unknown(self):
        with pytest.raises(ValueError, match="unknown calendar"):
            H.get_calendar("XXX")
        with pytest.raises(ValueError, match="roll convention"):
            H.get_calendar("USD").adjust(D(2025, 1, 4), "nearest")
        with pytest.raises(ValueError):
            H.get_calendar("USD").add_business_days(D(2025, 1, 4), -1)


class TestDeprecatedShim:
    def test_import_warns_and_gives_rule_calendars(self):
        sys.modules.pop("qp_core.calendars", None)
        with pytest.warns(DeprecationWarning, match="qp_core.holidays"):
            old = importlib.import_module("qp_core.calendars")
        # Good Friday was a business day in the old partial table.
        assert not old.get_calendar("GBP").is_business_day(D(2025, 4, 18))
        assert old.BusinessCalendar is H.HolidayCalendar

    def test_nothing_imports_the_old_tables(self):
        root = pathlib.Path(__file__).resolve().parents[3]
        pat = re.compile(r"(from\s+qp_core\.calendars\s+import|import\s+qp_core\.calendars"
                         r"|from\s+qp_core\s+import\s+[^\n]*\bcalendars\b"
                         r"|from\s+\.calendars\s+import)")
        allowed = {pathlib.Path(__file__).resolve()}
        bad = []
        for p in root.rglob("*.py"):
            if any(part in {".venv", "build", "node_modules", ".git"} for part in p.parts):
                continue
            if p.resolve() in allowed:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            if pat.search(text):
                # fi_rates/fx_rates keep a local ``calendars`` module that
                # re-exports qp_core.holidays; importing that is fine.
                own = p.parent / "calendars.py"
                if "from .calendars import" in text and own.exists() and \
                        "qp_core.holidays" in own.read_text(encoding="utf-8"):
                    continue
                bad.append(str(p.relative_to(root)))
        assert bad == []


_QL = {
    "USD": lambda ql: ql.UnitedStates(ql.UnitedStates.FederalReserve),
    "GBP": lambda ql: ql.UnitedKingdom(ql.UnitedKingdom.Settlement),
    "EUR": lambda ql: ql.TARGET(),
    "JPY": lambda ql: ql.Japan(),
    "CHF": lambda ql: ql.Switzerland(),
    "CAD": lambda ql: ql.Canada(ql.Canada.Settlement),
    "AUD": lambda ql: ql.Australia(ql.Australia.Settlement),
}


@pytest.mark.parametrize("ccy", sorted(_QL))
def test_matches_quantlib(ccy):
    """Weekday holiday sets equal QuantLib's for every year 2000-2030.

    USD is the Federal Reserve calendar (FX and money-market settlement).
    QuantLib's ``UnitedStates(GovernmentBond)`` is the SIFMA bond-market
    calendar, a different thing (it closes on Good Friday), so it is not
    the oracle for USD settlement.
    """
    ql = pytest.importorskip("QuantLib")
    cal = _QL[ccy](ql)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for y in range(2000, 2031):
            got = {D(d.year(), d.month(), d.dayOfMonth())
                   for d in ql.Calendar.holidayList(cal, ql.Date(1, 1, y),
                                                    ql.Date(31, 12, y), False)}
            assert H.holidays(ccy, y) == got, (ccy, y)
