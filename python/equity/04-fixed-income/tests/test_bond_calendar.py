"""Treasury coupon payment dates roll on the SIFMA bond calendar (USD-GOVT).

Accrual runs on the unadjusted schedule; a coupon due on a non-business
day is paid the next business day with no extra interest (Treasury
convention: the uniform offering circular, 31 CFR Part 356). Good Friday is open at the Fed but
closed in the bond market, so the two calendars give different dates.
"""

from __future__ import annotations

import datetime as dt

from fi_rates.bond import FixedRateBond
from fi_rates.calendars import SWAP_CALENDAR, TREASURY_CALENDAR

D = dt.date


def _bond(**kw):
    return FixedRateBond(D(2025, 10, 14), D(2028, 4, 14), 0.04, **kw)


def test_default_calendar_is_the_bond_market():
    assert TREASURY_CALENDAR == "USD-GOVT" and SWAP_CALENDAR == "USD"
    assert _bond().calendar == "USD-GOVT"


def test_good_friday_coupon_paid_on_monday():
    b = _bond()
    assert b.schedule()[-1] == D(2028, 4, 14)          # accrual date unchanged
    assert b.payment_dates()[-1] == D(2028, 4, 17)     # Good Friday 14-Apr-2028


def test_fed_calendar_would_pay_on_good_friday():
    assert _bond(calendar="USD").payment_dates()[-1] == D(2028, 4, 14)


def test_weekend_only_and_unchanged_dates():
    b = _bond(calendar=None)
    assert b.payment_dates() == b.schedule()  # all weekdays here
    assert _bond().payment_dates()[:-1] == b.schedule()[:-1]
