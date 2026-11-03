"""Deprecated: use :mod:`qp_core.holidays`.

Until round 6 this module held partial calendars: weekends plus a short
list of holidays on the same date every year, with no Good Friday, no
Thanksgiving, no UK bank holidays and no weekend substitutes. Schedules
rolled on it could land on a closed day. The rule-based calendars in
:mod:`qp_core.holidays` replace it everywhere.

The names below now point at the rule-based calendars, so old code gets
correct dates. ``BusinessCalendar`` is :class:`qp_core.holidays.HolidayCalendar`;
it has no ``fixed_holidays`` attribute any more (use ``.holidays(year)``).
Importing this module warns. It ships in v3.0.0 with the warning and is
removed in v4.0.0 (docs/VERSIONING.md).
"""

from __future__ import annotations

import warnings

from .holidays import CALENDARS, ROLL_CONVENTIONS, get_calendar, joint_calendar
from .holidays import HolidayCalendar as BusinessCalendar

warnings.warn(
    "qp_core.calendars is deprecated and will be removed in v4.0.0; "
    "import from qp_core.holidays (HolidayCalendar, get_calendar, joint_calendar)",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CALENDARS",
    "ROLL_CONVENTIONS",
    "BusinessCalendar",
    "get_calendar",
    "joint_calendar",
]
