"""FX option dates and cuts: spot date, expiry and delivery from a tenor, time to the cut.

Plain idea: an FX option expires at a fixed local time on its expiry day,
the "cut" (New York 10:00, Tokyo 15:00, and so on), and settles on the spot
date of that expiry day. Tenors are counted from the trade's spot date for
months and years and from the trade date for days and weeks. Time to expiry
is the calendar time from now to the cut, which moves with daylight saving.

Formal statement (Clark 2011, "Foreign Exchange Option Pricing", Wiley,
ch. 1; Wystup 2017, "FX Options and Structured Products", 2nd ed., section
1.5). Business days skip Saturdays, Sundays and holidays. ``holidays`` is
``"auto"`` (the default) or a collection of dates supplied by the caller.

* Spot date. With ``"auto"`` this is :func:`qp_core.holidays.fx_spot_date`,
  the one FX spot rule in the repository: ``spot_lag`` business days (1 for
  USDCAD, USDTRY, USDRUB and USDPHP, else 2) counted on the non-USD
  centres, then rolled forward until both centres and New York are open.
  A USD holiday on T+1 does not stop the count for a USD pair, and a cross
  never settles on a USD holiday. With a caller-supplied set the lag is
  counted on that set alone. Until round 7 the default was an empty set
  (no holidays at all) and ``"auto"`` counted the lag on the joint pair
  calendar, so EURUSD traded on the day before Thanksgiving 2026 got Mon
  30 Nov instead of Fri 27 Nov.
* Day and week tenors: expiry = trade date plus the calendar days, rolled
  forward to a business day; delivery = spot date of the expiry.
* Month and year tenors: delivery = spot date plus the months, modified
  following, end-of-month rule (spot on the month's last business day
  maps to the last business day); expiry = the latest business day whose
  spot date is on or before delivery.
* Time to expiry: ACT/365F calendar time from the valuation instant to the
  cut instant, both in UTC (``zoneinfo``).

Worked example: EURUSD traded Monday 2026-11-02: spot Wednesday 11-04; 1M
delivery Friday 12-04, expiry Wednesday 12-02 at 10:00 New York, which is
15:00 UTC (EST).
"""

from __future__ import annotations

import calendar
import datetime as dt
from collections.abc import Collection
from zoneinfo import ZoneInfo

from qp_core.holidays import fx_spot_date
from qp_core.holidays import spot_lag as _spot_lag

from .holidays import pair_holidays

__all__ = ["CUTS", "add_business_days", "cut_datetime_utc", "expiry_and_delivery",
           "spot_date", "spot_lag", "time_to_expiry"]

CUTS = {
    "NY": ("America/New_York", 10, 0),
    "TOK": ("Asia/Tokyo", 15, 0),
    "ECB": ("Europe/Berlin", 14, 15),
}

def spot_lag(pair: str) -> int:
    """Spot lag in business days for a six-letter pair such as ``"EURUSD"``
    (:func:`qp_core.holidays.spot_lag`)."""
    return _spot_lag(pair)


Holidays = Collection[dt.date] | str


def _resolve(holidays: Holidays, pair: str, first_year: int, last_year: int) -> Collection[dt.date]:
    if isinstance(holidays, str):
        if holidays != "auto":
            raise ValueError(f'holidays must be a collection of dates or "auto", got {holidays!r}')
        return pair_holidays(pair, first_year, last_year)
    return holidays


def _is_business(d: dt.date, holidays: Collection[dt.date]) -> bool:
    return d.weekday() < 5 and d not in holidays


def _roll_following(d: dt.date, holidays: Collection[dt.date]) -> dt.date:
    while not _is_business(d, holidays):
        d += dt.timedelta(days=1)
    return d


def add_business_days(d: dt.date, n: int, holidays: Collection[dt.date] = ()) -> dt.date:
    """``d`` plus ``n >= 0`` business days."""
    if n < 0:
        raise ValueError("n must be >= 0")
    while n > 0:
        d += dt.timedelta(days=1)
        if _is_business(d, holidays):
            n -= 1
    return d


def spot_date(trade_date: dt.date, pair: str, holidays: Holidays = "auto") -> dt.date:
    """Spot date of a trade.

    ``"auto"`` delegates to :func:`qp_core.holidays.fx_spot_date`; a
    collection of dates counts ``spot_lag`` business days on that set.
    """
    if isinstance(holidays, str):
        if holidays != "auto":
            raise ValueError(f'holidays must be a collection of dates or "auto", got {holidays!r}')
        return fx_spot_date(trade_date, pair)
    return add_business_days(trade_date, spot_lag(pair), holidays)


def _last_business_of_month(year: int, month: int, holidays: Collection[dt.date]) -> dt.date:
    d = dt.date(year, month, calendar.monthrange(year, month)[1])
    while not _is_business(d, holidays):
        d -= dt.timedelta(days=1)
    return d


def _add_months(d: dt.date, months: int, holidays: Collection[dt.date]) -> dt.date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    if d == _last_business_of_month(d.year, d.month, holidays):
        return _last_business_of_month(year, month, holidays)
    target = dt.date(year, month, min(d.day, calendar.monthrange(year, month)[1]))
    rolled = _roll_following(target, holidays)
    if rolled.month != month:  # modified following
        rolled = _last_business_of_month(year, month, holidays)
    return rolled


def expiry_and_delivery(trade_date: dt.date, tenor: str, pair: str,
                        holidays: Holidays = "auto") -> tuple[dt.date, dt.date]:
    """Expiry and delivery dates for a tenor such as ``"1W"``, ``"3M"``, ``"1Y"``."""
    t = tenor.strip().upper()
    if len(t) < 2 or not t[:-1].isdigit() or t[-1] not in "DWMY" or int(t[:-1]) <= 0:
        raise ValueError(f"tenor must look like 1D, 2W, 3M or 1Y, got {tenor!r}")
    n, unit = int(t[:-1]), t[-1]
    span = n // 365 + 1 if unit == "D" else n // 52 + 1 if unit == "W" else n // 12 + 1         if unit == "M" else n + 1
    spot_cal: Holidays = holidays
    holidays = _resolve(holidays, pair, trade_date.year, trade_date.year + span)

    def spot(d: dt.date) -> dt.date:
        return spot_date(d, pair, spot_cal)

    if unit in "DW":
        expiry = _roll_following(trade_date + dt.timedelta(days=n * (7 if unit == "W" else 1)),
                                 holidays)
        return expiry, spot(expiry)
    delivery = _add_months(spot(trade_date), n * (12 if unit == "Y" else 1), holidays)
    expiry = delivery - dt.timedelta(days=1)
    while not _is_business(expiry, holidays) or spot(expiry) > delivery:
        expiry -= dt.timedelta(days=1)
    return expiry, delivery


def cut_datetime_utc(expiry: dt.date, cut: str = "NY") -> dt.datetime:
    """The cut instant on ``expiry``, as an aware UTC datetime."""
    if cut not in CUTS:
        raise ValueError(f"cut must be one of {tuple(CUTS)}, got {cut!r}")
    zone, hh, mm = CUTS[cut]
    local = dt.datetime(expiry.year, expiry.month, expiry.day, hh, mm, tzinfo=ZoneInfo(zone))
    return local.astimezone(dt.UTC)


def time_to_expiry(valuation: dt.datetime, expiry: dt.date, cut: str = "NY") -> float:
    """ACT/365F years from an aware ``valuation`` instant to the cut."""
    if valuation.tzinfo is None:
        raise ValueError("valuation must be timezone-aware")
    seconds = (cut_datetime_utc(expiry, cut) - valuation).total_seconds()
    if seconds < 0.0:
        raise ValueError("valuation is after the cut")
    return seconds / (365.0 * 86400.0)
