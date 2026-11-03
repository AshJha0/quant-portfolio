"""Settlement-date rules for US cash equities (T+1) and FX spot (T+2).

Plain idea: a trade agreed today is paid for and delivered a fixed number of
*good business days* later. "Good" depends on which calendars must be open.

Formal rules implemented here:

- **US equities, T+1.** SEC Rule 15c6-1 as amended (compliance date
  2024-05-28). The trade date must be an NYSE trading day. Settlement is the
  first day after it that is neither a weekend, an NYSE holiday, nor a
  Federal Reserve holiday (DTC follows the Fed: Columbus Day and Veterans Day
  are trading days but not settlement days).
- **FX spot, T+2.** Market convention as
  described in the ISDA 1998 FX and Currency Option Definitions and the
  ACI Model Code: count the spot lag in days that are business days in each
  *non-USD* currency of the pair (a USD holiday on the intermediate day does
  not stop the count); then roll the result forward until it is a business
  day in both currencies *and* in USD.

Calendars. The Federal Reserve, TARGET2 and UK calendars, and the FX spot
rule itself, are the repository's single rules-based copy in
``qp_core.holidays`` (``USD``, ``EUR``, ``GBP``, ... and
``qp_core.holidays.fx_spot_date``), so post-trade and the pricing packages
cannot disagree about a value date. NYSE trading days are not a settlement
centre and are not in ``qp_core.holidays``; they are generated here by the
NYSE Rule 7.2 holiday rules (fixed dates and "nth weekday" rules, Good
Friday from the Easter computus, a Saturday holiday observed on the Friday
before except New Year's Day, a Sunday holiday on the Monday after) plus the
one-off closures listed in ``_NYSE_SPECIAL``. Both cover 2000 to 2099;
outside that range a date raises, so a missing table cannot give a silent
wrong answer. A currency without a shipped calendar also raises (the core
function would otherwise count weekends only for it).

Worked example: a US equity trade on Friday 2026-10-09 settles Tuesday
2026-10-13, because Monday 2026-10-12 (Columbus Day) is a Fed holiday.

>>> from datetime import date
>>> us_equity_settle(date(2026, 10, 9))
datetime.date(2026, 10, 13)
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import cache

from qp_core import holidays as _hol

_DAY = timedelta(days=1)

#: One-off NYSE full-day closures (NYSE press releases): 11-14 Sep 2001,
#: 11 Jun 2004 (Reagan), 2 Jan 2007 (Ford), 29-30 Oct 2012 (Sandy),
#: 5 Dec 2018 (G. H. W. Bush), 9 Jan 2025 (Carter).
_NYSE_SPECIAL = frozenset({
    date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13), date(2001, 9, 14),
    date(2004, 6, 11), date(2007, 1, 2), date(2012, 10, 29), date(2012, 10, 30),
    date(2018, 12, 5), date(2025, 1, 9),
})


def _check(d: date) -> None:
    if not _hol.MIN_YEAR <= d.year <= _hol.MAX_YEAR:
        raise ValueError(f"no holiday table for {d.year}; covered: {_hol.MIN_YEAR}-{_hol.MAX_YEAR}")


def _nth(year: int, month: int, weekday: int, n: int) -> date:
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))
        return d
    d = date(year + month // 12, month % 12 + 1, 1) - _DAY
    return d - timedelta(days=(d.weekday() - weekday) % 7)


@cache
def nyse_holidays(year: int) -> frozenset[date]:
    """Weekday full-day NYSE closures in ``year`` (Rule 7.2 plus one-offs).

    >>> sorted(nyse_holidays(2026))[:3]
    [datetime.date(2026, 1, 1), datetime.date(2026, 1, 19), datetime.date(2026, 2, 16)]
    """
    _check(date(year, 1, 1))
    out = set()
    new_year = date(year, 1, 1)
    if new_year.weekday() == 6:
        out.add(new_year + _DAY)
    elif new_year.weekday() < 5:
        out.add(new_year)  # a Saturday New Year's Day is not moved to 31 Dec
    fixed = [date(year, 7, 4), date(year, 12, 25)] + ([date(year, 6, 19)] if year >= 2022 else [])
    for d in fixed:
        out.add(d - _DAY if d.weekday() == 5 else d + _DAY if d.weekday() == 6 else d)
    out |= {_nth(year, 1, 0, 3), _nth(year, 2, 0, 3), _nth(year, 5, 0, -1),
            _nth(year, 9, 0, 1), _nth(year, 11, 3, 4)}
    out.add(_hol.easter_sunday(year) - 2 * _DAY)
    out |= {d for d in _NYSE_SPECIAL if d.year == year}
    return frozenset(d for d in out if d.weekday() < 5 and d.year == year)


def _nyse_open(d: date) -> bool:
    _check(d)
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def _fed_open(d: date) -> bool:
    _check(d)
    return _hol.is_business_day(d, "USD")


def us_equity_settle(trade_date: date) -> date:
    """T+1 settlement date for a US cash-equity trade."""
    if not _nyse_open(trade_date):
        raise ValueError(f"{trade_date} is not an NYSE trading day")
    d = trade_date + _DAY
    while not (_nyse_open(d) and _fed_open(d)):
        d += _DAY
    return d


def fx_spot_date(pair: str, trade_date: date) -> date:
    """Spot value date for a currency pair such as ``"EURUSD"``.

    Delegates to :func:`qp_core.holidays.fx_spot_date` (T+2, T+1 for
    USDCAD and the other ``T_PLUS_1_PAIRS``) after checking that both
    currencies have a shipped calendar.
    """
    p = pair.upper().replace("/", "")
    for c in (p[:3], p[3:]):
        if c not in _hol.CURRENCIES:
            raise ValueError(f"no holiday table for currency {c}; shipped: {_hol.CURRENCIES}")
    _check(trade_date)
    spot = _hol.fx_spot_date(trade_date, p)
    _check(spot)
    return spot
