"""Business-day calendars and roll conventions. The code lives in :mod:`qp_core.calendars`.

This module re-exports it so ``fx_rates.calendars`` keeps working. The
calendars are partial: weekends plus fixed-date holidays only.
"""

from __future__ import annotations

from qp_core.calendars import (
    CALENDARS,
    ROLL_CONVENTIONS,
    BusinessCalendar,
    get_calendar,
    joint_calendar,
)

__all__ = [
    "BusinessCalendar",
    "CALENDARS",
    "ROLL_CONVENTIONS",
    "get_calendar",
    "joint_calendar",
]
