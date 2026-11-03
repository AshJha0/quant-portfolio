"""Day count conventions and FX settlement date logic.

Conventions implemented
-----------------------
- ``ACT/360``  : actual days / 360 (money-market convention, USD/EUR deposits).
- ``ACT/365F`` : actual days / 365 fixed (common for GBP money market, zero rates).
- ``30/360``   : US (bond basis) 30/360, 31sts rolled back to 30 per the
  standard rule (d1=31 -> 30; d2=31 -> 30 only if d1 >= 30).

Money-market accrual follows :data:`MONEY_MARKET_DAYCOUNT` (ACT/360 for
USD, EUR, JPY, CHF; ACT/365F for GBP, AUD, CAD, NZD).

Settlement logic
----------------
``spot_date`` counts ``spot_lag_days`` business days on the non-USD
currencies' calendars, then rolls forward to a day that is good in every
currency of the pair including USD. ``tenor_dates`` gives the start and end
of each market tenor: ON runs trade date to T+1, TN runs T+1 to spot, SN
runs spot to spot+1, and W/M/Y tenors run from spot with the end-end rule
and modified following. Holiday calendars are the rule-based ones in
:mod:`qp_core.holidays` (sources listed there); a currency with no shipped
calendar counts weekends only.

All year fractions are returned as plain floats (units: years).
"""

from __future__ import annotations

import datetime as dt

from qp_core import daycount as _core
from qp_core.daycount import MONEY_MARKET_DAYCOUNT, money_market_year_fraction
from qp_core.daycount import add_months as _roll_months
from qp_core.daycount import last_business_day as _last_business_day
from qp_core.holidays import fx_spot_date as _fx_spot_date
from qp_core.holidays import spot_lag as _spot_lag

from .calendars import CALENDARS, BusinessCalendar, joint_calendar

__all__ = [
    "MONEY_MARKET_DAYCOUNT",
    "VALID_CONVENTIONS",
    "add_calendar_days",
    "forward_settlement_date",
    "money_market_year_fraction",
    "pair_calendar",
    "spot_date",
    "tenor_dates",
    "tenor_to_years",
    "year_fraction",
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
    return _core.year_fraction(start, end, convention, allow_negative=False)


def _ccy_calendar(ccy: str) -> BusinessCalendar:
    return CALENDARS.get(ccy.upper(), CALENDARS["WEEKEND"])


def pair_calendar(pair: tuple[str, str] | None) -> BusinessCalendar:
    """Joint calendar of both currencies (weekends only when ``pair`` is None)."""
    if pair is None:
        return CALENDARS["WEEKEND"]
    return joint_calendar(*(_ccy_calendar(c) for c in pair))


def add_calendar_days(date: dt.date, days: int) -> dt.date:
    """Add ``days`` calendar days to ``date`` (may land on a weekend/holiday)."""
    return date + dt.timedelta(days=days)


def spot_date(
    trade_date: dt.date,
    spot_lag_days: int | None = None,
    pair: tuple[str, str] | None = None,
) -> dt.date:
    """FX spot date: T + ``spot_lag_days`` business days.

    Business days are counted on the non-USD currencies of ``pair``; the
    result is then rolled forward to a day that is good in every currency
    of the pair and in USD (a cross never settles on a New York holiday).
    The rule lives in :func:`qp_core.holidays.fx_spot_date`. With
    ``pair=None`` only weekends count.

    ``spot_lag_days=None`` (the default) uses the pair's market lag: T+1
    for USDCAD, USDTRY, USDRUB and USDPHP, T+2 otherwise. Before round 6
    the default was 2 for every pair.

    The old ``calendar_days=True`` (calendar-day arithmetic) was removed in
    v3.0.0.

    Raises
    ------
    ValueError
        If ``spot_lag_days`` is negative.
    """
    if spot_lag_days is None:
        spot_lag_days = 2 if pair is None else _spot_lag(pair)
    if int(spot_lag_days) != spot_lag_days or spot_lag_days < 0:
        raise ValueError(f"spot_lag_days must be >= 0, got {spot_lag_days}")
    if pair is None:
        wk, d = CALENDARS["WEEKEND"], trade_date
        for _ in range(int(spot_lag_days)):
            d = wk.add_business_days(d + dt.timedelta(days=1), 0)
        return wk.adjust(d, "following")
    return _fx_spot_date(trade_date, pair, int(spot_lag_days))


def tenor_dates(
    trade_date: dt.date,
    tenor: str,
    pair: tuple[str, str] | None = None,
    spot_lag_days: int | None = None,
) -> tuple[dt.date, dt.date]:
    """Start and end (value) dates of a market tenor.

    * ``ON``: trade date to the next business day.
    * ``TN``: next business day to spot.
    * ``SN``: spot to the business day after spot.
    * ``nD``/``nW``: spot plus calendar days, modified following.
    * ``nM``/``nY``: spot plus months, modified following, with the end-end
      rule (spot on the last business day of its month maps to the last
      business day of the target month).
    """
    cal = pair_calendar(pair)
    t = tenor.strip().upper()
    today = cal.adjust(trade_date, "following")
    spot = spot_date(trade_date, spot_lag_days, pair)
    if t == "ON":
        return today, cal.add_business_days(today + dt.timedelta(days=1), 0)
    if t == "TN":
        return cal.add_business_days(today + dt.timedelta(days=1), 0), spot
    if t == "SN":
        return spot, cal.add_business_days(spot + dt.timedelta(days=1), 0)
    if len(t) < 2:
        raise ValueError(f"Cannot parse tenor {tenor!r}")
    unit, num = t[-1], t[:-1]
    try:
        n = int(num)
    except ValueError as exc:
        raise ValueError(f"Cannot parse tenor {tenor!r}") from exc
    if n <= 0:
        raise ValueError(f"Tenor must be positive: {tenor!r}")
    if unit in ("D", "W"):
        days = n if unit == "D" else 7 * n
        return spot, cal.adjust(spot + dt.timedelta(days=days), "modified_following")
    if unit in ("M", "Y"):
        months = n if unit == "M" else 12 * n
        if spot == _last_business_day(spot.year, spot.month, cal):
            raw = _roll_months(spot, months)
            return spot, _last_business_day(raw.year, raw.month, cal)
        return spot, cal.adjust(_roll_months(spot, months), "modified_following")
    raise ValueError(f"Cannot parse tenor {tenor!r}: unknown unit {unit!r}")


def forward_settlement_date(
    trade_date: dt.date,
    tenor: str,
    spot_lag_days: int | None = None,
    pair: tuple[str, str] | None = None,
) -> dt.date:
    """Value date of an FX forward of market ``tenor`` (see :func:`tenor_dates`).

    Tenors run from spot (a "3M" forward settles three calendar months
    after the spot date), except ON and TN, which settle before spot.
    """
    return tenor_dates(trade_date, tenor, pair, spot_lag_days)[1]


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
        raise ValueError(
            f"tenor_to_years({t!r}): ON and TN settle before spot, so a year "
            "fraction from spot is not defined; use tenor_dates for value dates "
            "(the 1-day length was removed in v3.0.0)")
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
