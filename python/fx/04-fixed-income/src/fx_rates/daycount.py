"""Day count conventions and (simplified) FX settlement date logic.

Conventions implemented
-----------------------
- ``ACT/360``  : actual days / 360 (money-market convention, USD/EUR deposits).
- ``ACT/365F`` : actual days / 365 fixed (common for GBP money market, zero rates).
- ``30/360``   : US (bond basis) 30/360 — 31sts rolled back to 30 per the
  standard rule (d1=31 -> 30; d2=31 -> 30 only if d1 >= 30).

Settlement logic
----------------
FX spot settles T+2 for most pairs.  Production systems roll the settlement
date over currency holidays in *both* currencies plus USD; here we use a
**calendar-day** approximation (``spot_date`` simply adds ``spot_lag_days``
calendar days).  This simplification is documented in
``docs/METHODOLOGY.md`` (assumption A6) and its impact is discussed in
``docs/VALIDATION.md``.

All year fractions are returned as plain floats (units: years).
"""

from __future__ import annotations

import datetime as dt

__all__ = [
    "VALID_CONVENTIONS",
    "year_fraction",
    "add_calendar_days",
    "spot_date",
    "forward_settlement_date",
    "tenor_to_years",
]

VALID_CONVENTIONS: tuple[str, ...] = ("ACT/360", "ACT/365F", "30/360")


def year_fraction(start: dt.date, end: dt.date, convention: str = "ACT/365F") -> float:
    """Year fraction between two dates under a day-count convention.

    Parameters
    ----------
    start, end : datetime.date
        Accrual start and end dates.  ``end`` must not precede ``start``.
    convention : str
        One of ``ACT/360``, ``ACT/365F``, ``30/360``.

    Returns
    -------
    float
        Accrual fraction in years (0.0 when ``start == end``).

    Raises
    ------
    ValueError
        If the convention is unknown or ``end < start``.
    """
    if convention not in VALID_CONVENTIONS:
        raise ValueError(
            f"Unknown day-count convention {convention!r}; expected one of {VALID_CONVENTIONS}"
        )
    if end < start:
        raise ValueError(f"end date {end} precedes start date {start}")
    if convention == "ACT/360":
        return (end - start).days / 360.0
    if convention == "ACT/365F":
        return (end - start).days / 365.0
    # 30/360 US (bond basis)
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 >= 30:
        d2 = 30
    return (
        360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)
    ) / 360.0


def add_calendar_days(date: dt.date, days: int) -> dt.date:
    """Add ``days`` calendar days to ``date`` (may land on a weekend/holiday)."""
    return date + dt.timedelta(days=days)


def spot_date(trade_date: dt.date, spot_lag_days: int = 2) -> dt.date:
    """FX spot settlement date, **calendar-day** T+2 approximation.

    Production systems use T+2 *good business days* rolled over the holiday
    calendars of both currencies (T+1 for USDCAD).  We deliberately use
    calendar days — the error is at most a few days of carry (documented
    simplification, see METHODOLOGY.md assumption A6).

    Raises
    ------
    ValueError
        If ``spot_lag_days`` is negative.
    """
    if spot_lag_days < 0:
        raise ValueError(f"spot_lag_days must be >= 0, got {spot_lag_days}")
    return add_calendar_days(trade_date, spot_lag_days)


def forward_settlement_date(
    trade_date: dt.date, tenor: str, spot_lag_days: int = 2
) -> dt.date:
    """Settlement date of an FX forward: spot date + tenor (calendar approximation).

    Tenors are quoted spot-to-settlement (market convention): a "3M" forward
    settles three months after the *spot* date, not the trade date.
    Months are approximated as ``round(365.25 * years / 12)`` calendar days.
    """
    spot = spot_date(trade_date, spot_lag_days)
    years = tenor_to_years(tenor)
    return add_calendar_days(spot, round(365.25 * years))


def tenor_to_years(tenor: str) -> float:
    """Parse a market tenor string into years.

    Supports ``D`` (ACT/365 days), ``W`` (weeks), ``M`` (months / 12),
    ``Y`` (years).  Examples: ``"ON"`` -> 1/365, ``"3M"`` -> 0.25,
    ``"5Y"`` -> 5.0.

    Raises
    ------
    ValueError
        If the tenor cannot be parsed.
    """
    t = tenor.strip().upper()
    if t in ("ON", "TN", "SN"):
        return 1.0 / 365.0
    if len(t) < 2:
        raise ValueError(f"Cannot parse tenor {tenor!r}")
    unit, num = t[-1], t[:-1]
    try:
        n = float(num)
    except ValueError as exc:
        raise ValueError(f"Cannot parse tenor {tenor!r}") from exc
    if n <= 0:
        raise ValueError(f"Tenor must be positive: {tenor!r}")
    if unit == "D":
        return n / 365.0
    if unit == "W":
        return 7.0 * n / 365.0
    if unit == "M":
        return n / 12.0
    if unit == "Y":
        return n
    raise ValueError(f"Cannot parse tenor {tenor!r}: unknown unit {unit!r}")
