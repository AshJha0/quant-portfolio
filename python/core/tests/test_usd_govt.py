"""USD-GOVT: the SIFMA US government bond-market calendar.

The Federal Reserve calendar (``USD``) is the right one for FX and swap
settlement, but Treasuries trade and settle on the bond-market calendar
recommended by SIFMA, which closes on Good Friday, moves Saturday
holidays to the Friday before, and has had one-off closures. Each named
case below is a date on which ``USD`` and ``USD-GOVT`` disagree.
``test_matches_quantlib_government_bond`` checks every year 2000-2030
against QuantLib ``UnitedStates(GovernmentBond)``; the three dates where
this calendar differs from QuantLib 1.44 are listed with their source.
"""

from __future__ import annotations

import datetime as dt
import warnings

import pytest
from qp_core import holidays as H

D = dt.date
GOVT = "USD-GOVT"

#: Closures SIFMA (then The Bond Market Association) recommended that
#: QuantLib 1.44 does not list. 2001: bond market closed 11 and 12 Sep and
#: reopened 13 Sep. 2025: national day of mourning for President Carter.
NOT_IN_QUANTLIB = {D(2001, 9, 11), D(2001, 9, 12), D(2025, 1, 9)}


class TestNamedDays:
    def test_good_friday_closed_for_bonds_open_at_the_fed(self):
        assert not H.is_business_day(D(2025, 4, 18), GOVT)
        assert H.is_business_day(D(2025, 4, 18), "USD")

    def test_good_friday_on_payrolls_day_is_an_early_close_not_a_holiday(self):
        # 3 Apr 2026 is the first Friday of the month (Employment Situation
        # release); SIFMA recommends a noon close, so it is a business day.
        assert H.is_business_day(D(2026, 4, 3), GOVT)
        assert H.is_business_day(D(2023, 4, 7), GOVT)
        assert not H.is_business_day(D(2024, 3, 29), GOVT)

    def test_saturday_holidays_move_to_friday(self):
        # Christmas Sat 25-Dec-2027 and 4 July on Sat 2026: Friday closed.
        assert not H.is_business_day(D(2027, 12, 24), GOVT)
        assert H.is_business_day(D(2027, 12, 24), "USD")
        assert not H.is_business_day(D(2026, 7, 3), GOVT)
        assert not H.is_business_day(D(2027, 6, 18), GOVT)  # Juneteenth Sat

    def test_saturday_new_year_and_veterans_day_not_moved(self):
        assert H.is_business_day(D(2021, 12, 31), GOVT)
        assert H.is_business_day(D(2023, 11, 10), GOVT)

    def test_special_closures(self):
        for d in (D(2004, 6, 11), D(2012, 10, 30), D(2018, 12, 5), *NOT_IN_QUANTLIB):
            assert not H.is_business_day(d, GOVT), d
            assert H.is_business_day(d, "USD") or d == D(2001, 9, 11), d

    def test_lookup_names(self):
        assert H.get_calendar("USD-GOVT").centres == (GOVT,)
        assert H.get_calendar("SIFMA") is H.get_calendar(GOVT)
        joint = H.get_calendar("USD-GOVT+GBP")
        assert joint.centres == (GOVT, "GBP")
        assert not joint.is_business_day(D(2025, 5, 5))     # UK early May
        assert not joint.is_business_day(D(2025, 4, 18))    # Good Friday
        assert GOVT in H.CALENDAR_CODES and GOVT not in H.CURRENCIES

    def test_runtime_closure(self):
        d = D(2031, 5, 7)
        with H.extra_closures({GOVT: [d]}):
            assert not H.is_business_day(d, GOVT)
        assert H.is_business_day(d, GOVT)

    def test_fx_spot_ignores_the_bond_calendar(self):
        # Good Friday 2025: EURUSD is closed anyway (TARGET); USDCAD is not.
        assert H.fx_spot_date(D(2025, 4, 17), "USDCAD") == D(2025, 4, 21)


def test_matches_quantlib_government_bond():
    ql = pytest.importorskip("QuantLib")
    cal = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for y in range(2000, 2031):
            got = {D(d.year(), d.month(), d.dayOfMonth())
                   for d in ql.Calendar.holidayList(cal, ql.Date(1, 1, y),
                                                    ql.Date(31, 12, y), False)}
            want = got | {d for d in NOT_IN_QUANTLIB if d.year == y}
            assert H.holidays(GOVT, y) == want, y
