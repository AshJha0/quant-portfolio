"""Day count conventions, accrual fractions and coupon schedule generation.

Conventions implemented
-----------------------
``ACT/365F``     actual days / 365 (fixed).  Used throughout this package to
                 map calendar dates onto the curve's time axis.
``ACT/360``      actual days / 360.  Money-market convention (deposits, FRAs).
``30/360US``     US (NASD) 30/360 bond basis, without the end-of-February
                 special rule (documented simplification).
``ACT/ACT-ISDA`` actual days split by calendar year, leap years / 366,
                 non-leap years / 365.

Schedules
---------
``generate_schedule`` rolls coupon dates *backwards* from maturity in steps of
``12 / frequency`` months (annual / semiannual / quarterly), producing a short
front stub if the effective date does not align.  Business-day adjustment is a
deliberately simple *modified following* over a weekend-only calendar
(no holiday calendars) and is off by default; accrual is computed on
unadjusted dates.  Both simplifications are documented in
``docs/METHODOLOGY.md`` (assumptions register).
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
    "generate_schedule",
]

SUPPORTED_CONVENTIONS: tuple[str, ...] = (
    "ACT/365F",
    "ACT/360",
    "30/360US",
    "ACT/ACT-ISDA",
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


def year_fraction(start: dt.date, end: dt.date, convention: str = "ACT/365F") -> float:
    """Day count fraction between two dates.

    Parameters
    ----------
    start, end : datetime.date
        Accrual period start / end.  ``end`` may precede ``start``; the
        fraction is then negative (useful for horizon arithmetic).
    convention : str
        One of :data:`SUPPORTED_CONVENTIONS`.

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


def adjust_modified_following(date: dt.date) -> dt.date:
    """Modified-following adjustment over a weekend-only calendar.

    Roll a Saturday/Sunday forward to Monday unless that crosses a month end,
    in which case roll backward to Friday.  No holiday calendars — documented
    simplification.
    """
    adjusted = date
    while adjusted.weekday() >= 5:  # Sat=5, Sun=6
        adjusted += dt.timedelta(days=1)
    if adjusted.month != date.month:
        adjusted = date
        while adjusted.weekday() >= 5:
            adjusted -= dt.timedelta(days=1)
    return adjusted


def generate_schedule(
    effective: dt.date,
    maturity: dt.date,
    frequency: int,
    adjust: bool = False,
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
        If True, apply modified-following (weekend-only) to *payment* dates.
        Accrual elsewhere in the package always uses unadjusted dates.

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
        current = add_months(maturity, -step * k)
    dates.reverse()
    if adjust:
        dates = [adjust_modified_following(d) for d in dates]
    return dates
