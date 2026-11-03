"""Day count conventions, accrual fractions, month arithmetic and schedules.

Shared by ``fi_rates`` and ``fx_rates``. Each of those packages keeps its own
public ``year_fraction`` with its own list of accepted conventions and its own
sign rule, and delegates the arithmetic here.

Conventions
-----------
``ACT/365F``     actual days / 365 (fixed).
``ACT/360``      actual days / 360. Money-market convention (deposits, FRAs).
``30/360US``     US (NASD) 30/360 bond basis: a day-31 start becomes 30, and a
                 day-31 end becomes 30 when the start is (after that change)
                 30. No end-of-February rule. ``30/360`` is an alias.
``ACT/ACT-ISDA`` actual days split by calendar year, leap years / 366,
                 non-leap years / 365.
``ACT/ACT-ICMA`` actual days in the accrual period over (frequency x actual
                 days in the reference coupon period). Needs ``ref_start``,
                 ``ref_end`` and ``frequency``.

Sign rule
---------
With ``allow_negative=True`` (the default) an ``end`` before ``start`` gives
a negative fraction, which is useful for horizon arithmetic. With
``allow_negative=False`` it raises ``ValueError``.
"""

from __future__ import annotations

import calendar
import datetime as dt

from .holidays import HolidayCalendar as BusinessCalendar
from .holidays import get_calendar

__all__ = [
    "MONEY_MARKET_DAYCOUNT",
    "SUPPORTED_CONVENTIONS",
    "SUPPORTED_FREQUENCIES",
    "add_months",
    "adjust_modified_following",
    "generate_schedule",
    "is_month_end",
    "last_business_day",
    "money_market_year_fraction",
    "roll_months",
    "year_fraction",
]

SUPPORTED_CONVENTIONS: tuple[str, ...] = (
    "ACT/365F",
    "ACT/360",
    "30/360US",
    "30/360",
    "ACT/ACT-ISDA",
    "ACT/ACT-ICMA",
)

#: Coupon frequencies (payments per year): annual, semiannual, quarterly.
SUPPORTED_FREQUENCIES: tuple[int, ...] = (1, 2, 4)

#: Money-market (deposit, FX-swap implied rate) day count per currency.
MONEY_MARKET_DAYCOUNT: dict[str, str] = {
    "USD": "ACT/360", "EUR": "ACT/360", "JPY": "ACT/360", "CHF": "ACT/360",
    "GBP": "ACT/365F", "AUD": "ACT/365F", "CAD": "ACT/365F", "NZD": "ACT/365F",
}


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
    end_of_start_year = dt.date(start.year + 1, 1, 1)
    denom = 366.0 if calendar.isleap(start.year) else 365.0
    frac += _days(start, end_of_start_year) / denom
    frac += float(end.year - start.year - 1)
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
    allow_negative: bool = True,
) -> float:
    """Day count fraction between two dates, in years.

    Parameters
    ----------
    start, end : datetime.date
        Accrual period start and end.
    convention : str
        One of :data:`SUPPORTED_CONVENTIONS`.
    ref_start, ref_end, frequency
        Reference coupon period and coupons per year. Required for
        ``ACT/ACT-ICMA`` and ignored otherwise.
    allow_negative : bool
        If False, ``end < start`` raises ``ValueError``.

    Raises
    ------
    ValueError
        Unknown convention, missing ICMA reference data, or ``end < start``
        with ``allow_negative=False``.
    """
    if convention not in SUPPORTED_CONVENTIONS:
        raise ValueError(
            f"Unknown day count convention {convention!r}; "
            f"supported: {SUPPORTED_CONVENTIONS}"
        )
    if not allow_negative and end < start:
        raise ValueError(f"end date {end} precedes start date {start}")
    if convention == "ACT/365F":
        return _days(start, end) / 365.0
    if convention == "ACT/360":
        return _days(start, end) / 360.0
    if convention in ("30/360US", "30/360"):
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


def money_market_year_fraction(start: dt.date, end: dt.date, currency: str) -> float:
    """Deposit accrual fraction in the money-market convention of ``currency``.

    ``end`` must not precede ``start``.
    """
    ccy = currency.upper()
    if ccy not in MONEY_MARKET_DAYCOUNT:
        raise ValueError(
            f"no money-market convention for {currency!r}; known: {tuple(MONEY_MARKET_DAYCOUNT)}"
        )
    return year_fraction(start, end, MONEY_MARKET_DAYCOUNT[ccy], allow_negative=False)


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


def last_business_day(year: int, month: int, cal: BusinessCalendar) -> dt.date:
    """Last business day of ``year``-``month`` on ``cal``."""
    d = dt.date(year, month, calendar.monthrange(year, month)[1])
    return cal.adjust(d, "preceding")


def adjust_modified_following(
    date: dt.date, calendar: str | BusinessCalendar | None = None
) -> dt.date:
    """Modified-following roll on ``calendar`` (default: weekends only).

    ``calendar`` is a name (``"USD"``, ``"EUR+USD"``) or a
    :class:`qp_core.holidays.HolidayCalendar`.
    """
    return get_calendar(calendar).adjust(date, "modified_following")


def generate_schedule(
    effective: dt.date,
    maturity: dt.date,
    frequency: int,
    adjust: bool = False,
    eom: bool = True,
    calendar: str | BusinessCalendar | None = None,
) -> list[dt.date]:
    """Coupon payment dates, rolled backwards from maturity.

    Steps of ``12 / frequency`` months; a short front stub arises when
    ``effective`` is not a whole number of periods before ``maturity``. With
    ``eom`` a month-end maturity keeps every coupon date on a month end. With
    ``adjust`` payment dates are rolled modified-following on ``calendar``;
    accrual should use the unadjusted dates.

    Returns strictly increasing dates in ``(effective, maturity]``.
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
