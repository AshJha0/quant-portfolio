"""Business-day calendars and date-roll conventions.

The calendars here are **partial**. Each one knows weekends and a short list
of holidays that fall on the same calendar date every year. It does not know
moveable feasts (Easter, Golden Week bridges), floating holidays (US
Thanksgiving, Labor Day, UK bank holidays) or weekend-observance rules. A
production system loads holiday lists from a vendor feed; this module gives
the abstraction those feeds plug into, and the rolls are correct for every
date the partial lists cover.

=========  ======================================================
Name       Fixed-date holidays (month, day)
=========  ======================================================
WEEKEND    none (Saturday and Sunday only)
USD        1 Jan, 19 Jun, 4 Jul, 11 Nov, 25 Dec
EUR        1 Jan, 1 May, 25 Dec, 26 Dec (TARGET2 fixed dates)
GBP        1 Jan, 25 Dec, 26 Dec
JPY        1-3 Jan, 11 Feb, 23 Feb, 29 Apr, 3-5 May, 3 Nov, 23 Nov, 31 Dec
=========  ======================================================

Roll conventions: ``following``, ``modified_following``, ``preceding``,
``unadjusted``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

__all__ = [
    "CALENDARS",
    "ROLL_CONVENTIONS",
    "BusinessCalendar",
    "get_calendar",
    "joint_calendar",
]

ROLL_CONVENTIONS: tuple[str, ...] = (
    "following",
    "modified_following",
    "preceding",
    "unadjusted",
)


@dataclass(frozen=True)
class BusinessCalendar:
    """Weekend plus fixed-date holiday calendar (partial, see module docs).

    Parameters
    ----------
    name : str
        Label, e.g. ``"USD"`` or ``"EUR+USD"`` for a joint calendar.
    fixed_holidays : frozenset of (month, day)
        Holidays that fall on the same date every year.
    """

    name: str
    fixed_holidays: frozenset[tuple[int, int]] = frozenset()

    def is_business_day(self, date: dt.date) -> bool:
        """True when ``date`` is a weekday and not a listed holiday."""
        if date.weekday() >= 5:
            return False
        return (date.month, date.day) not in self.fixed_holidays

    def adjust(self, date: dt.date, convention: str = "modified_following") -> dt.date:
        """Roll ``date`` to a business day under ``convention``."""
        if convention not in ROLL_CONVENTIONS:
            raise ValueError(
                f"unknown roll convention {convention!r}; expected one of {ROLL_CONVENTIONS}"
            )
        if convention == "unadjusted" or self.is_business_day(date):
            return date
        if convention == "preceding":
            return self._step(date, -1)
        rolled = self._step(date, +1)
        if convention == "modified_following" and rolled.month != date.month:
            return self._step(date, -1)
        return rolled

    def _step(self, date: dt.date, direction: int) -> dt.date:
        d = date
        while not self.is_business_day(d):
            d += dt.timedelta(days=direction)
        return d

    def add_business_days(self, date: dt.date, n: int) -> dt.date:
        """Move ``n`` business days forward (``n >= 0``).

        ``n = 0`` returns ``date`` rolled forward to a business day.
        """
        if int(n) != n or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n}")
        d = self._step(date, +1)
        for _ in range(int(n)):
            d = self._step(d + dt.timedelta(days=1), +1)
        return d

    def business_days_between(self, start: dt.date, end: dt.date) -> int:
        """Number of business days in ``(start, end]``."""
        if end < start:
            raise ValueError(f"end {end} precedes start {start}")
        count, d = 0, start
        while d < end:
            d += dt.timedelta(days=1)
            count += self.is_business_day(d)
        return count


CALENDARS: dict[str, BusinessCalendar] = {
    "WEEKEND": BusinessCalendar("WEEKEND"),
    "USD": BusinessCalendar(
        "USD", frozenset({(1, 1), (6, 19), (7, 4), (11, 11), (12, 25)})
    ),
    "EUR": BusinessCalendar("EUR", frozenset({(1, 1), (5, 1), (12, 25), (12, 26)})),
    "GBP": BusinessCalendar("GBP", frozenset({(1, 1), (12, 25), (12, 26)})),
    "JPY": BusinessCalendar(
        "JPY",
        frozenset(
            {
                (1, 1), (1, 2), (1, 3), (2, 11), (2, 23), (4, 29),
                (5, 3), (5, 4), (5, 5), (11, 3), (11, 23), (12, 31),
            }
        ),
    ),
}


def get_calendar(name: str | BusinessCalendar | None) -> BusinessCalendar:
    """Look up a calendar by name. ``None`` gives the weekend-only calendar."""
    if name is None:
        return CALENDARS["WEEKEND"]
    if isinstance(name, BusinessCalendar):
        return name
    key = str(name).upper()
    if "+" in key:
        return joint_calendar(*key.split("+"))
    if key not in CALENDARS:
        raise ValueError(f"unknown calendar {name!r}; expected one of {tuple(CALENDARS)}")
    return CALENDARS[key]


def joint_calendar(*names: str | BusinessCalendar) -> BusinessCalendar:
    """Calendar whose business days are business days in every input."""
    if not names:
        return CALENDARS["WEEKEND"]
    cals = [get_calendar(n) for n in names]
    hol: frozenset[tuple[int, int]] = frozenset().union(*(c.fixed_holidays for c in cals))
    return BusinessCalendar("+".join(c.name for c in cals), hol)
