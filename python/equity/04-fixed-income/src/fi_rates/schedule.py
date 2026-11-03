"""Dated accrual schedules for swaps and floating-rate legs (W13a).

Plain idea
----------
A swap leg pays on a grid of dates. The grid starts from the contract's
effective and termination dates, steps by a fixed number of months, rolls
each date to a business day, and pays a few business days after each
period ends. When the term is not a whole number of steps, one period
(the *stub*) is shorter or longer than the rest.

Formal statement
----------------
Unadjusted dates are generated from the termination date backwards (front
stubs) or from the effective date forwards (back stubs) in steps of
``tenor_months``. Each generated date takes the roll day ``roll_day``
(clamped to the month's length), or the month end when ``end_of_month`` is
set and the anchor date is a month end. A short stub keeps the odd
remainder as its own period; a long stub merges it with the neighbouring
regular period. Accrual dates are the unadjusted dates rolled with
``convention`` on ``calendar`` (ISDA 2021 Definitions, section 4.12:
"Adjusted" period end dates), except that the effective date is rolled with
the same convention and the termination date with
``termination_convention``. The payment date is the adjusted accrual end
moved ``payment_lag`` business days forward (USD SOFR OIS: 2 days; ARRC
2020 conventions).

This matches ``QuantLib.Schedule`` with ``DateGeneration.Backward`` /
``Forward`` for the cases in ``tests/test_schedule.py`` (cross-checked
against QuantLib 1.44 when it is installed).

Worked example
--------------
A 1-year USD OIS from Wed 15 Jan 2025, annual, payment lag 2::

    >>> import datetime as dt
    >>> s = make_schedule(dt.date(2025, 1, 15), dt.date(2026, 1, 15), 12,
    ...                   calendar="USD", payment_lag=2)
    >>> [(p.start, p.end, p.payment) for p in s]   # doctest: +NORMALIZE_WHITESPACE
    [(datetime.date(2025, 1, 15), datetime.date(2026, 1, 15), datetime.date(2026, 1, 20))]

15 Jan 2026 is a Thursday; two USD business days later is Monday 19 Jan,
which is Martin Luther King Day, so the payment rolls to Tuesday 20 Jan.
"""

from __future__ import annotations

import calendar as _cal
import datetime as dt
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from qp_core.holidays import HolidayCalendar, get_calendar

__all__ = [
    "STUB_TYPES",
    "Period",
    "Schedule",
    "make_schedule",
    "shift_business_days",
]

STUB_TYPES: tuple[str, ...] = ("short_front", "long_front", "short_back", "long_back")
_ONE_DAY = dt.timedelta(days=1)


def shift_business_days(
    calendar: HolidayCalendar, date: dt.date, n: int
) -> dt.date:
    """Move ``n`` business days (``n`` may be negative or zero).

    ``n = 0`` returns ``date`` unchanged even on a holiday; otherwise each
    step lands on a business day of ``calendar``.
    """
    if int(n) != n:
        raise ValueError(f"n must be an integer, got {n}")
    step = _ONE_DAY if n > 0 else -_ONE_DAY
    d = date
    for _ in range(abs(int(n))):
        d += step
        while not calendar.is_business_day(d):
            d += step
    return d


def _month_end(year: int, month: int) -> int:
    return _cal.monthrange(year, month)[1]


def _add_months(anchor: dt.date, months: int, roll_day: int, eom: bool) -> dt.date:
    y, m = divmod(anchor.month - 1 + months, 12)
    year, month = anchor.year + y, m + 1
    last = _month_end(year, month)
    day = last if eom else min(roll_day, last)
    return dt.date(year, month, day)


@dataclass(frozen=True)
class Period:
    """One accrual period.

    ``start``/``end`` are adjusted accrual dates, ``unadjusted_start`` and
    ``unadjusted_end`` the generated dates before rolling, and ``payment``
    the payment date. ``is_stub`` marks an irregular period.
    """

    start: dt.date
    end: dt.date
    payment: dt.date
    unadjusted_start: dt.date
    unadjusted_end: dt.date
    is_stub: bool = False


@dataclass(frozen=True)
class Schedule(Sequence[Period]):
    """An ordered tuple of :class:`Period` plus the conventions that made it."""

    periods: tuple[Period, ...]
    calendar: HolidayCalendar
    convention: str
    payment_lag: int

    def __len__(self) -> int:
        return len(self.periods)

    def __getitem__(self, i):  # type: ignore[no-untyped-def, override]
        return self.periods[i]

    def __iter__(self) -> Iterator[Period]:
        return iter(self.periods)

    @property
    def dates(self) -> list[dt.date]:
        """Adjusted accrual boundaries: ``start`` of every period, then the last ``end``."""
        return [p.start for p in self.periods] + [self.periods[-1].end]


def _unadjusted(
    effective: dt.date, termination: dt.date, months: int, stub: str,
    roll_day: int | None, end_of_month: bool,
) -> tuple[list[dt.date], bool]:
    """Unadjusted boundaries (ascending) and whether the stub is real."""
    backward = stub in ("short_front", "long_front")
    anchor = termination if backward else effective
    eom = end_of_month and anchor.day == _month_end(anchor.year, anchor.month)
    rd = anchor.day if roll_day is None else int(roll_day)
    if not 1 <= rd <= 31:
        raise ValueError(f"roll_day must be in 1..31, got {roll_day}")
    dates = [anchor]
    k = 1
    if backward:
        while True:
            d = _add_months(anchor, -k * months, rd, eom)
            if d <= effective:
                break
            dates.append(d)
            k += 1
        dates.append(effective)
        dates.reverse()
    else:
        while True:
            d = _add_months(anchor, k * months, rd, eom)
            if d >= termination:
                break
            dates.append(d)
            k += 1
        dates.append(termination)
    # Is the odd period a true stub? Compare it with a regular step.
    if backward:
        regular = _add_months(dates[1], -months, rd, eom) if len(dates) > 2 else (
            _add_months(termination, -months, rd, eom))
        has_stub = regular != dates[0]
    else:
        regular = _add_months(dates[-2], months, rd, eom) if len(dates) > 2 else (
            _add_months(effective, months, rd, eom))
        has_stub = regular != dates[-1]
    if has_stub and stub.startswith("long") and len(dates) > 2:
        # Merge the stub into its neighbour.
        del dates[1 if backward else -2]
    return dates, has_stub


def make_schedule(
    effective: dt.date,
    termination: dt.date,
    tenor_months: int,
    *,
    calendar: str | HolidayCalendar | None = "USD",
    convention: str = "modified_following",
    termination_convention: str | None = None,
    stub: str = "short_front",
    roll_day: int | None = None,
    end_of_month: bool = False,
    payment_lag: int = 0,
    payment_convention: str = "following",
) -> Schedule:
    """Build a dated accrual schedule (see the module docstring).

    Parameters
    ----------
    effective, termination : datetime.date
        Unadjusted start and end of the leg; ``effective < termination``.
    tenor_months : int
        Months per regular period (12 annual, 6, 3, 1). ``0`` gives a single
        period (a zero-coupon / term leg).
    calendar : str or HolidayCalendar
        Business-day calendar (``"USD"``, ``"USD-GOVT"``, ``"EUR"``/``"TARGET"``,
        ``"GBP"``, joint ``"EUR+GBP"``...).
    convention, termination_convention : str
        Roll conventions from :data:`qp_core.holidays.ROLL_CONVENTIONS`.
    stub : str
        One of :data:`STUB_TYPES`. Front stubs generate backwards from the
        termination date; back stubs forwards from the effective date.
    roll_day : int, optional
        Day of month for generated dates; defaults to the anchor's day.
    end_of_month : bool
        Keep generated dates on month ends when the anchor is a month end.
    payment_lag : int
        Business days from the adjusted accrual end to payment (``>= 0``).
    payment_convention : str
        Roll for the payment date when ``payment_lag == 0``.
    """
    if not effective < termination:
        raise ValueError(f"effective {effective} must precede termination {termination}")
    if stub not in STUB_TYPES:
        raise ValueError(f"stub must be one of {STUB_TYPES}, got {stub!r}")
    if int(tenor_months) != tenor_months or tenor_months < 0:
        raise ValueError(f"tenor_months must be a non-negative integer, got {tenor_months}")
    if int(payment_lag) != payment_lag or payment_lag < 0:
        raise ValueError(f"payment_lag must be a non-negative integer, got {payment_lag}")
    cal = get_calendar(calendar)
    term_conv = convention if termination_convention is None else termination_convention
    if tenor_months == 0:
        raw, has_stub = [effective, termination], False
    else:
        raw, has_stub = _unadjusted(effective, termination, int(tenor_months), stub,
                                    roll_day, end_of_month)
    adj = [cal.adjust(d, convention) for d in raw[:-1]] + [cal.adjust(raw[-1], term_conv)]
    periods = []
    n = len(raw) - 1
    stub_index = 0 if stub.endswith("front") else n - 1
    for i in range(n):
        end = adj[i + 1]
        pay = (shift_business_days(cal, end, payment_lag) if payment_lag
               else cal.adjust(end, payment_convention))
        periods.append(Period(adj[i], end, pay, raw[i], raw[i + 1],
                              is_stub=has_stub and i == stub_index))
    for p in periods:
        if not p.start < p.end:
            raise ValueError(f"adjusted period {p.start}..{p.end} is empty; check the dates")
    return Schedule(tuple(periods), cal, convention, int(payment_lag))
