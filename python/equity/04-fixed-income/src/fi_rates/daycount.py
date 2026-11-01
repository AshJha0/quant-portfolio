"""Day count conventions, accrual fractions and coupon schedule generation.

Conventions implemented
-----------------------
``ACT/365F``     actual days / 365 (fixed).  Used throughout this package to
                 map calendar dates onto the curve's time axis.
``ACT/360``      actual days / 360.  Money-market convention (deposits, FRAs).
``30/360US``     US (NASD) 30/360 bond basis, without the end-of-February
                 special rule (documented simplification).
``ACT/ACT-ISDA`` actual days split by calendar year, leap years / 366,
                 non-leap years / 365. Swap and some euro-area
                 conventions. Not the government-bond convention.
``ACT/ACT-ICMA`` actual days in the accrual period over (frequency x actual
                 days in the reference coupon period). Regular coupons are
                 exactly ``coupon / frequency``. US Treasuries, gilts and
                 most government bonds use it. It needs the reference
                 period, so call ``year_fraction`` with ``ref_start``,
                 ``ref_end`` and ``frequency``.

Schedules
---------
``generate_schedule`` rolls coupon dates backwards from maturity in steps of
``12 / frequency`` months (annual, semiannual, quarterly), producing a short
front stub if the effective date does not align. With ``eom=True`` (the
default) a month-end maturity keeps every coupon date on a month end, so a
28-Feb-2025 maturity pays on 31 Aug. Payment dates can be rolled with any
calendar from :mod:`fi_rates.calendars` (``adjust=True``); accrual uses
unadjusted dates. 30/360US omits the end-of-February rule.
"""

from __future__ import annotations

import calendar
import datetime as dt

__all__ = [
    "SUPPORTED_CONVENTIONS",
    "SUPPORTED_FREQUENCIES",
    "year_fraction",
    "add_months",
    "adjust_modified_following",
    "is_month_end",
    "roll_months",
    "generate_schedule",
]

SUPPORTED_CONVENTIONS: tuple[str, ...] = (
    "ACT/365F",
    "ACT/360",
    "30/360US",
    "ACT/ACT-ISDA",
    "ACT/ACT-ICMA",
)

#: Coupon frequencies supported everywhere in the package: annual, semiannual,
#: quarterly (payments per year).
SUPPORTED_FREQUENCIES: tuple[int, ...] = (1, 2, 4)


def _days(start: dt.date, end: dt.date) -> int:
    return (end - start).days


def _thirty360_us(start: dt.date, end: dt.date) -> float:
    """30/360 US (NASD) day count fraction, without the Feb-EOM rule."""
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    days = 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)
    return days / 360.0


def _act_act_isda(start: dt.date, end: dt.date) -> float:
    """ACT/ACT ISDA: split the accrual period at each calendar year end."""
    if start.year == end.year:
        denom = 366.0 if calendar.isleap(start.year) else 365.0
        return _days(start, end) / denom
    frac = 0.0
    # Stub in the starting year.
    end_of_start_year = dt.date(start.year + 1, 1, 1)
    denom = 366.0 if calendar.isleap(start.year) else 365.0
    frac += _days(start, end_of_start_year) / denom
    # Whole years in between.
    frac += float(end.year - start.year - 1)
    # Stub in the ending year.
    start_of_end_year = dt.date(end.year, 1, 1)
    denom = 366.0 if calendar.isleap(end.year) else 365.0
    frac += _days(start_of_end_year, end) / denom
    return frac


def year_fraction(
    start: dt.date,
    end: dt.date,
    convention: str = "ACT/365F",
    *,
    ref_start: dt.date | None = None,
    ref_end: dt.date | None = None,
    frequency: int | None = None,
) -> float:
    """Day count fraction between two dates.

    Parameters
    ----------
    start, end : datetime.date
        Accrual period start / end.  ``end`` may precede ``start``; the
        fraction is then negative (useful for horizon arithmetic).
    convention : str
        One of :data:`SUPPORTED_CONVENTIONS`.
    ref_start, ref_end, frequency
        Reference coupon period and coupons per year. Required for
        ``ACT/ACT-ICMA`` and ignored otherwise.

    Returns
    -------
    float
        Year fraction under the requested convention.
    """
    if convention not in SUPPORTED_CONVENTIONS:
        raise ValueError(
            f"Unknown day count convention {convention!r}; "
            f"supported: {SUPPORTED_CONVENTIONS}"
        )
    if convention == "ACT/365F":
        return _days(start, end) / 365.0
    if convention == "ACT/360":
        return _days(start, end) / 360.0
    if convention == "30/360US":
        return _thirty360_us(start, end)
    if convention == "ACT/ACT-ICMA":
        if ref_start is None or ref_end is None or frequency is None:
            raise ValueError(
                "ACT/ACT-ICMA needs ref_start, ref_end and frequency "
                "(the reference coupon period)"
            )
        if frequency not in SUPPORTED_FREQUENCIES:
            raise ValueError(f"frequency must be one of {SUPPORTED_FREQUENCIES}")
        ref_days = _days(ref_start, ref_end)
        if ref_days <= 0:
            raise ValueError(f"reference period {ref_start}..{ref_end} is empty")
        return _days(start, end) / (frequency * ref_days)
    return _act_act_isda(start, end)


def add_months(date: dt.date, months: int) -> dt.date:
    """Add calendar months, clamping the day-of-month to the target month end.

    E.g. 2024-01-31 + 1 month = 2024-02-29 (leap year).
    """
    month_index = date.month - 1 + months
    year = date.year + month_index // 12
    month = month_index % 12 + 1
    day = min(date.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)


def is_month_end(date: dt.date) -> bool:
    """True when ``date`` is the last calendar day of its month."""
    return date.day == calendar.monthrange(date.year, date.month)[1]


def roll_months(date: dt.date, months: int, eom: bool = False) -> dt.date:
    """Add months; with ``eom`` a month-end date maps to a month end."""
    out = add_months(date, months)
    if eom and is_month_end(date):
        out = dt.date(out.year, out.month, calendar.monthrange(out.year, out.month)[1])
    return out


def adjust_modified_following(date: dt.date, calendar=None) -> dt.date:
    """Modified-following roll on ``calendar`` (default: weekends only).

    Roll forward to the next business day unless that crosses a month end,
    in which case roll backward. ``calendar`` is a name (``"USD"``,
    ``"EUR+USD"``) or a :class:`fi_rates.calendars.BusinessCalendar`.
    """
    from .calendars import get_calendar

    return get_calendar(calendar).adjust(date, "modified_following")


def generate_schedule(
    effective: dt.date,
    maturity: dt.date,
    frequency: int,
    adjust: bool = False,
    eom: bool = True,
    calendar=None,
) -> list[dt.date]:
    """Generate coupon payment dates by rolling back from maturity.

    Parameters
    ----------
    effective : datetime.date
        Interest start date (issue / dated date).
    maturity : datetime.date
        Final redemption date (always a payment date).
    frequency : int
        Payments per year: 1 (annual), 2 (semiannual) or 4 (quarterly).
    adjust : bool
        If True, apply modified-following on ``calendar`` to payment dates.
        Accrual elsewhere in the package always uses unadjusted dates.
    eom : bool
        End-of-month rule: when ``maturity`` is a month end, every coupon
        date is a month end.
    calendar : str or BusinessCalendar, optional
        Calendar for ``adjust`` (default weekends only).

    Returns
    -------
    list[datetime.date]
        Strictly increasing payment dates in ``(effective, maturity]``.  A
        short front stub arises naturally when ``effective`` is not a whole
        number of periods before ``maturity``.
    """
    if frequency not in SUPPORTED_FREQUENCIES:
        raise ValueError(
            f"frequency must be one of {SUPPORTED_FREQUENCIES} "
            f"(annual/semiannual/quarterly), got {frequency}"
        )
    if maturity <= effective:
        raise ValueError(f"maturity {maturity} must be after effective {effective}")
    step = 12 // frequency
    dates: list[dt.date] = []
    k = 0
    current = maturity
    while current > effective:
        dates.append(current)
        k += 1
        current = roll_months(maturity, -step * k, eom=eom)
    dates.reverse()
    if adjust:
        dates = [adjust_modified_following(d, calendar) for d in dates]
    return dates
