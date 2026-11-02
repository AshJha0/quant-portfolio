"""Rules-based settlement holiday calendars for the supported currencies.

Plain idea: an FX trade settles only on a day that is a business day in
both currencies' settlement centres. The dates follow fixed rules (fixed
dates, "third Monday", Easter offsets, weekend substitutes), so a small
generator reproduces them without a network call or a data file. It is
deterministic and covers the years 2010 to 2099 inclusive (``MIN_YEAR``, ``MAX_YEAR``);
outside that range it raises rather than guess.

Formal statement. Each currency maps to the settlement holidays of its
centre, generated from these published rules:

* USD, New York: Federal Reserve Bank holidays (federalreserve.gov,
  "Holidays observed by the Federal Reserve System"). New Year, MLK (3rd
  Mon Jan), Presidents (3rd Mon Feb), Memorial (last Mon May), Juneteenth
  (Jun 19, from 2022), Independence (Jul 4), Labor (1st Mon Sep), Columbus
  (2nd Mon Oct), Veterans (Nov 11), Thanksgiving (4th Thu Nov), Christmas.
  A Sunday holiday moves to Monday; a Saturday holiday is not observed.
* EUR, TARGET2 closing days (ECB): Jan 1, Good Friday, Easter Monday,
  May 1, Dec 25, Dec 26. No substitutes.
* GBP, London: England and Wales bank holidays (Banking and Financial
  Dealings Act 1971; gov.uk/bank-holidays). New Year (weekend to Monday),
  Good Friday, Easter Monday, early May (1st Mon), spring (last Mon May),
  summer (last Mon Aug), Christmas and Boxing Day with substitutes. Known
  proclamations: 2020-05-08 (early May moved), 2022-06-02/03 (jubilee,
  spring moved), 2022-09-19 (state funeral), 2023-05-08 (coronation).
* JPY, Tokyo: the Act on National Holidays (1948, as amended) plus the
  bank closing days Jan 2, Jan 3 and Dec 31. Substitute holiday: a Sunday
  holiday moves to the next non-holiday day. Citizens' holiday: a day
  between two holidays. Equinoxes by the National Astronomical
  Observatory approximation valid 1980-2099. Olympic moves in 2020 and
  2021 are included.
* CHF, Zurich (SIX): Jan 1, Jan 2, Good Friday, Easter Monday, May 1,
  Ascension, Whit Monday, Aug 1, Dec 25, Dec 26.
* CAD, Toronto (Bank of Canada): New Year, Family Day (3rd Mon Feb, from
  2008), Good Friday, Victoria Day (Monday before May 25), Canada Day,
  Civic (1st Mon Aug), Labour (1st Mon Sep), Truth and Reconciliation
  (Sep 30, from 2021), Thanksgiving (2nd Mon Oct), Remembrance (Nov 11),
  Christmas and Boxing Day. Weekend holidays move to the next weekday.
* AUD, Sydney (NSW bank holidays): New Year, Australia Day (Jan 26), Good
  Friday, Easter Monday, Anzac Day (Apr 25, no substitute), King's/Queen's
  Birthday (2nd Mon Jun), Bank Holiday (1st Mon Aug), Labour (1st Mon
  Oct), Christmas and Boxing Day, weekend days moved to the next weekday;
  2022-09-22 national day of mourning.

Limits: one-off closures announced after this code was written are not
known; pass extra dates yourself. Only these seven currencies are shipped.

Worked example: ``holidays("USD", 2026)`` has 10 days; Jul 4 is a Saturday
so the Fed is open on Friday Jul 3.
"""

from __future__ import annotations

import datetime as dt
from functools import cache

__all__ = ["CURRENCIES", "MAX_YEAR", "MIN_YEAR", "easter_sunday", "holidays", "pair_holidays"]

MIN_YEAR, MAX_YEAR = 2010, 2099
D = dt.date
_DAY = dt.timedelta(days=1)


def easter_sunday(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm, Meeus 1991, ch. 8)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return D(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """n-th (1-based) ``weekday`` (Mon=0) of a month; ``n = -1`` is the last."""
    if n > 0:
        d = D(year, month, 1)
        d += dt.timedelta(days=(weekday - d.weekday()) % 7)
        return d + dt.timedelta(weeks=n - 1)
    d = D(year + month // 12, month % 12 + 1, 1) - _DAY
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _next_weekday(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d += _DAY
    return d


def _xmas_boxing(year: int) -> set[dt.date]:
    """Christmas and Boxing Day with UK/Commonwealth weekend substitutes."""
    c, b = D(year, 12, 25), D(year, 12, 26)
    if c.weekday() == 5:  # Sat: Mon 27 and Tue 28
        return {D(year, 12, 27), D(year, 12, 28)}
    if c.weekday() == 6:  # Sun: Boxing Mon 26, Christmas Tue 27
        return {b, D(year, 12, 27)}
    if b.weekday() == 5:  # Fri Christmas, Sat Boxing -> Mon 28
        return {c, D(year, 12, 28)}
    return {c, b}


def _usd(y: int) -> set[dt.date]:
    fixed = [D(y, 1, 1), D(y, 7, 4), D(y, 11, 11), D(y, 12, 25)]
    if y >= 2022:
        fixed.append(D(y, 6, 19))
    out = {d + _DAY if d.weekday() == 6 else d for d in fixed if d.weekday() != 5}
    out |= {_nth_weekday(y, 1, 0, 3), _nth_weekday(y, 2, 0, 3), _nth_weekday(y, 5, 0, -1),
            _nth_weekday(y, 9, 0, 1), _nth_weekday(y, 10, 0, 2), _nth_weekday(y, 11, 3, 4)}
    return out


def _eur(y: int) -> set[dt.date]:
    e = easter_sunday(y)
    return {D(y, 1, 1), e - 2 * _DAY, e + _DAY, D(y, 5, 1), D(y, 12, 25), D(y, 12, 26)}


def _gbp(y: int) -> set[dt.date]:
    e = easter_sunday(y)
    out = {_next_weekday(D(y, 1, 1)), e - 2 * _DAY, e + _DAY, _nth_weekday(y, 8, 0, -1)}
    out |= _xmas_boxing(y)
    out.add(D(2020, 5, 8) if y == 2020 else _nth_weekday(y, 5, 0, 1))
    if y == 2022:
        out |= {D(2022, 6, 2), D(2022, 6, 3), D(2022, 9, 19)}
    else:
        out.add(_nth_weekday(y, 5, 0, -1))
    if y == 2023:
        out.add(D(2023, 5, 8))
    return out


def _equinox(y: int, base: float) -> int:
    return int(base + 0.242194 * (y - 1980)) - (y - 1980) // 4


def _jpy(y: int) -> set[dt.date]:
    nat = {D(y, 1, 1), D(y, 2, 11), D(y, 4, 29), D(y, 5, 3), D(y, 5, 4), D(y, 5, 5),
           D(y, 11, 3), D(y, 11, 23), _nth_weekday(y, 1, 0, 2), _nth_weekday(y, 9, 0, 3),
           D(y, 3, _equinox(y, 20.8431)), D(y, 9, _equinox(y, 23.2488))}
    if y >= 2020:
        nat.add(D(y, 2, 23))
    elif 1989 <= y <= 2018:
        nat.add(D(y, 12, 23))
    if y == 2019:
        nat |= {D(2019, 4, 30), D(2019, 5, 1), D(2019, 5, 2), D(2019, 10, 22)}
    if y == 2020:
        nat |= {D(2020, 7, 23), D(2020, 7, 24), D(2020, 8, 10)}
    elif y == 2021:
        nat |= {D(2021, 7, 22), D(2021, 7, 23), D(2021, 8, 8)}
    else:
        marine = _nth_weekday(y, 7, 0, 3) if y >= 2003 else D(y, 7, 20)
        sports = _nth_weekday(y, 10, 0, 2)
        nat |= {marine, sports}
        if y >= 2016:
            nat.add(D(y, 8, 11))
    # Citizens' holiday: a weekday sandwiched between two national holidays.
    for d in sorted(nat):
        mid = d + _DAY
        if mid not in nat and mid + _DAY in nat and mid.weekday() != 6:
            nat.add(mid)
    # Substitute holiday: a Sunday holiday moves to the next non-holiday day.
    out = set(nat)
    for d in sorted(nat):
        if d.weekday() == 6:
            s = d + _DAY
            while s in out:
                s += _DAY
            out.add(s)
    return out | {D(y, 1, 2), D(y, 1, 3), D(y, 12, 31)}


def _chf(y: int) -> set[dt.date]:
    e = easter_sunday(y)
    return {D(y, 1, 1), D(y, 1, 2), e - 2 * _DAY, e + _DAY, D(y, 5, 1), e + 39 * _DAY,
            e + 50 * _DAY, D(y, 8, 1), D(y, 12, 25), D(y, 12, 26)}


def _cad(y: int) -> set[dt.date]:
    e = easter_sunday(y)
    may24 = D(y, 5, 24)
    out = {_next_weekday(D(y, 1, 1)), e - 2 * _DAY,
           may24 - dt.timedelta(days=may24.weekday()), _next_weekday(D(y, 7, 1)),
           _nth_weekday(y, 8, 0, 1), _nth_weekday(y, 9, 0, 1), _nth_weekday(y, 10, 0, 2),
           _next_weekday(D(y, 11, 11))}
    if y >= 2008:
        out.add(_nth_weekday(y, 2, 0, 3))
    if y >= 2021:
        out.add(_next_weekday(D(y, 9, 30)))
    return out | _xmas_boxing(y)


def _aud(y: int) -> set[dt.date]:
    e = easter_sunday(y)
    out = {_next_weekday(D(y, 1, 1)), _next_weekday(D(y, 1, 26)), e - 2 * _DAY, e + _DAY,
           D(y, 4, 25), _nth_weekday(y, 6, 0, 2), _nth_weekday(y, 8, 0, 1),
           _nth_weekday(y, 10, 0, 1)}
    if y == 2022:
        out.add(D(2022, 9, 22))
    return out | _xmas_boxing(y)


_RULES = {"USD": _usd, "EUR": _eur, "GBP": _gbp, "JPY": _jpy, "CHF": _chf, "CAD": _cad,
          "AUD": _aud}
CURRENCIES = tuple(_RULES)


@cache
def holidays(currency: str, year: int) -> frozenset[dt.date]:
    """Weekday settlement holidays of ``currency`` in ``year``.

    Weekend dates are dropped: the business-day logic already skips them.
    """
    ccy = currency.upper()
    if ccy not in _RULES:
        raise ValueError(f"no calendar for {currency!r}; shipped: {CURRENCIES}")
    if not (isinstance(year, int) and MIN_YEAR <= year <= MAX_YEAR):
        raise ValueError(f"year must be an int in [{MIN_YEAR}, {MAX_YEAR}], got {year!r}")
    return frozenset(d for d in _RULES[ccy](year) if d.weekday() < 5 and d.year == year)


def pair_holidays(pair: str, start_year: int, end_year: int) -> frozenset[dt.date]:
    """Joint holidays of both currencies of ``pair`` over ``[start_year, end_year]``.

    Settlement needs both centres open, so the union is the right set.
    """
    p = pair.upper().replace("/", "")
    if len(p) != 6 or not p.isalpha():
        raise ValueError(f"pair must be six letters, got {pair!r}")
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    out: set[dt.date] = set()
    for y in range(start_year, end_year + 1):
        out |= holidays(p[:3], y) | holidays(p[3:], y)
    return frozenset(out)
