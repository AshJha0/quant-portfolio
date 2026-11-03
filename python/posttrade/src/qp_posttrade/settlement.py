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

The holiday tables cover 2026 only, from the published NYSE, Federal
Reserve, ECB (TARGET2) and UK bank-holiday calendars. Any date outside 2026
raises, so a stale table cannot give a silent wrong answer.

Worked example: a US equity trade on Friday 2026-10-09 settles Tuesday
2026-10-13, because Monday 2026-10-12 (Columbus Day) is a Fed holiday.

>>> from datetime import date
>>> us_equity_settle(date(2026, 10, 9))
datetime.date(2026, 10, 13)
"""

from __future__ import annotations

from datetime import date, timedelta

_COVERED_YEARS = frozenset({2026})

NYSE_2026 = frozenset(
    date(2026, m, d)
    for m, d in [(1, 1), (1, 19), (2, 16), (4, 3), (5, 25), (6, 19), (7, 3), (9, 7), (11, 26), (12, 25)]
)
# Federal Reserve: a holiday on Saturday is NOT moved to Friday, so July 4th
# 2026 (a Saturday) gives no Fed holiday, unlike NYSE's 3 July.
FED_2026 = frozenset(
    date(2026, m, d)
    for m, d in [(1, 1), (1, 19), (2, 16), (5, 25), (6, 19), (9, 7), (10, 12), (11, 11), (11, 26), (12, 25)]
)
TARGET_2026 = frozenset(date(2026, m, d) for m, d in [(1, 1), (4, 3), (4, 6), (5, 1), (12, 25), (12, 26)])
UK_2026 = frozenset(
    date(2026, m, d) for m, d in [(1, 1), (4, 3), (4, 6), (5, 4), (5, 25), (8, 31), (12, 25), (12, 28)]
)

CCY_HOLIDAYS: dict[str, frozenset[date]] = {"USD": FED_2026, "EUR": TARGET_2026, "GBP": UK_2026}


def _check(d: date) -> None:
    if d.year not in _COVERED_YEARS:
        raise ValueError(f"no holiday table for {d.year}; covered: {sorted(_COVERED_YEARS)}")


def _good(d: date, *holidays: frozenset[date]) -> bool:
    _check(d)
    return d.weekday() < 5 and not any(d in h for h in holidays)


def us_equity_settle(trade_date: date) -> date:
    """T+1 settlement date for a US cash-equity trade."""
    if not _good(trade_date, NYSE_2026):
        raise ValueError(f"{trade_date} is not an NYSE trading day")
    d = trade_date + timedelta(days=1)
    while not _good(d, NYSE_2026, FED_2026):
        d += timedelta(days=1)
    return d


def fx_spot_date(pair: str, trade_date: date) -> date:
    """Spot value date for a currency pair such as ``"EURUSD"``."""
    pair = pair.upper()
    base, quote = pair[:3], pair[3:]
    for c in (base, quote):
        if c not in CCY_HOLIDAYS:
            raise ValueError(f"no holiday table for currency {c}")
    _check(trade_date)
    lag = 2  # T+1 pairs (USDCAD) need a CAD table, which is not shipped
    non_usd = [CCY_HOLIDAYS[c] for c in (base, quote) if c != "USD"]
    d = trade_date
    counted = 0
    while counted < lag:
        d += timedelta(days=1)
        if _good(d, *non_usd):
            counted += 1
    while not _good(d, CCY_HOLIDAYS[base], CCY_HOLIDAYS[quote], FED_2026):
        d += timedelta(days=1)
    return d
