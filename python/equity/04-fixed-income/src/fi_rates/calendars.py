"""Business-day calendars and roll conventions, from :mod:`qp_core.holidays`.

This module re-exports the rule-based calendars so ``fi_rates.calendars``
keeps working. Until round 6 it re-exported partial fixed-date tables;
see :mod:`qp_core.holidays` for the rules and their sources.
``BusinessCalendar`` is kept as a name for ``HolidayCalendar``.
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

__all__ = [
    "CALENDARS",
    "ROLL_CONVENTIONS",
    "BusinessCalendar",
    "HolidayCalendar",
    "get_calendar",
    "joint_calendar",
]
