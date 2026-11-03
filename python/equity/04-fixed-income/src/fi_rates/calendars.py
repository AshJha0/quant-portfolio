"""Business-day calendars and roll conventions, from :mod:`qp_core.holidays`.

This module re-exports the rule-based calendars so ``fi_rates.calendars``
keeps working. Until round 6 it re-exported partial fixed-date tables;
see :mod:`qp_core.holidays` for the rules and their sources.
``BusinessCalendar`` is kept as a name for ``HolidayCalendar``.

Which calendar: Treasury and agency bonds pay on the SIFMA bond-market
calendar (``TREASURY_CALENDAR = "USD-GOVT"``); USD swaps and deposits keep
the Federal Reserve calendar (``SWAP_CALENDAR = "USD"``). The two differ on
Good Friday and when a fixed-date holiday falls on a Saturday.
"""

from __future__ import annotations

from qp_core.holidays import (
    CALENDARS,
    ROLL_CONVENTIONS,
    HolidayCalendar,
    get_calendar,
    joint_calendar,
)

BusinessCalendar = HolidayCalendar

#: Calendar for US Treasury and agency bond schedules: the SIFMA bond-market
#: calendar, which closes on Good Friday (see :mod:`qp_core.holidays`).
TREASURY_CALENDAR = "USD-GOVT"
#: Calendar for USD swaps and money markets: Federal Reserve holidays.
SWAP_CALENDAR = "USD"

__all__ = [
    "CALENDARS",
    "ROLL_CONVENTIONS",
    "SWAP_CALENDAR",
    "TREASURY_CALENDAR",
    "BusinessCalendar",
    "HolidayCalendar",
    "get_calendar",
    "joint_calendar",
]
