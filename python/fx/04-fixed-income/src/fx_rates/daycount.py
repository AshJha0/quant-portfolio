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
and modified following. Holiday calendars are partial (weekends and
fixed-date holidays only, see :mod:`fx_rates.calendars`).

All year fractions are returned as plain floats (units: years).
"""

from __future__ import annotations

import calendar as _cal
import datetime as dt
import warnings

from .calendars import CALENDARS, BusinessCalendar, joint_calendar

__all__ = [
    "VALID_CONVENTIONS",
    "MONEY_MARKET_DAYCOUNT",
    "year_fraction",
    "money_market_year_fraction",
    "add_calendar_days",
    "pair_calendar",
    "spot_date",
    "tenor_dates",
    "forward_settlement_date",
    "tenor_to_years",
]

VALID_CONVENTIONS: tuple[str, ...] = ("ACT/360", "ACT/365F", "30/360")

#: Money-market (deposit, FX-swap implied rate) day count per currency.
MONEY_MARKET_DAYCOUNT: dict[str, str] = {
    "USD": "ACT/360", "EUR": "ACT/360", "JPY": "ACT/360", "CHF": "ACT/360",
    "GBP": "ACT/365F", "AUD": "ACT/365F", "CAD": "ACT/365F", "NZD": "ACT/365F",
}


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


def money_market_year_fraction(start: dt.date, end: dt.date, currency: str) -> float:
    """Deposit accrual fraction in ``currency``'s money-market convention."""
    ccy = currency.upper()
    if ccy not in MONEY_MARKET_DAYCOUNT:
        raise ValueError(
            f"no money-market convention for {currency!r}; known: {tuple(MONEY_MARKET_DAYCOUNT)}"
        )
    return year_fraction(start, end, MONEY_MARKET_DAYCOUNT[ccy])


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
    spot_lag_days: int = 2,
    pair: tuple[str, str] | None = None,
    calendar_days: bool = False,
) -> dt.date:
    """FX spot date: T + ``spot_lag_days`` business days.

    Business days are counted on the non-USD currencies of ``pair``; the
    result is then rolled forward to a day that is good in every currency
    of the pair, USD included. With ``pair=None`` only weekends count.
    Use ``spot_lag_days=1`` for USDCAD.

    ``calendar_days=True`` restores the old calendar-day arithmetic and is
    deprecated.

    Raises
    ------
    ValueError
        If ``spot_lag_days`` is negative.
    """
    if int(spot_lag_days) != spot_lag_days or spot_lag_days < 0:
        raise ValueError(f"spot_lag_days must be >= 0, got {spot_lag_days}")
    if calendar_days:
        warnings.warn(
            "calendar-day spot dates are deprecated; spot_date now counts business days",
            DeprecationWarning,
            stacklevel=2,
        )
        return add_calendar_days(trade_date, spot_lag_days)
    if pair is None:
        count_cal = CALENDARS["WEEKEND"]
    else:
        non_usd = [c for c in pair if c.upper() != "USD"]
        if non_usd:
            count_cal = joint_calendar(*(_ccy_calendar(c) for c in non_usd))
        else:
            count_cal = _ccy_calendar("USD")
    d = trade_date
    for _ in range(int(spot_lag_days)):
        d = count_cal.add_business_days(d + dt.timedelta(days=1), 0)
    return pair_calendar(pair).adjust(d, "following")


def _roll_months(date: dt.date, months: int) -> dt.date:
    idx = date.month - 1 + months
    y, m = date.year + idx // 12, idx % 12 + 1
    return dt.date(y, m, min(date.day, _cal.monthrange(y, m)[1]))


def _last_business_day(year: int, month: int, cal: BusinessCalendar) -> dt.date:
    d = dt.date(year, month, _cal.monthrange(year, month)[1])
    return cal.adjust(d, "preceding")


def tenor_dates(
    trade_date: dt.date,
    tenor: str,
    pair: tuple[str, str] | None = None,
    spot_lag_days: int = 2,
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
    spot_lag_days: int = 2,
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
        warnings.warn(
            f"tenor_to_years({t!r}) returns a 1-day length only; ON and TN "
            "settle before spot. Use tenor_dates for value dates.",
            DeprecationWarning,
            stacklevel=2,
        )
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
