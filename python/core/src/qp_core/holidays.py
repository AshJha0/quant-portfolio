"""Rules-based settlement holiday calendars for the supported currencies.

Plain idea: an FX trade settles only on a day that is a business day in
both currencies' settlement centres. The dates follow fixed rules (fixed
dates, "third Monday", Easter offsets, weekend substitutes), so a small
generator reproduces them without a network call or a data file. It is
deterministic and covers the years 2000 to 2099 inclusive (``MIN_YEAR``, ``MAX_YEAR``);
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
  May 1, Dec 25, Dec 26. No substitutes. TARGET also closed on Dec 31
  in 2000 and 2001. The name ``TARGET`` is an alias of ``EUR``; use it for
  ECB reference-rate fixing dates.
* GBP, London: England and Wales bank holidays (Banking and Financial
  Dealings Act 1971; gov.uk/bank-holidays). New Year (weekend to Monday),
  Good Friday, Easter Monday, early May (1st Mon), spring (last Mon May),
  summer (last Mon Aug), Christmas and Boxing Day with substitutes. Known
  proclamations: 2002-06-03/04 (golden jubilee, spring moved),
  2011-04-29 (royal wedding), 2012-06-04/05 (diamond jubilee, spring
  moved), 2020-05-08 (early May moved), 2022-06-02/03 (platinum jubilee,
  spring moved), 2022-09-19 (state funeral), 2023-05-08 (coronation).
* JPY, Tokyo: the Act on National Holidays (1948, as amended) plus the
  bank closing days Jan 2, Jan 3 and Dec 31. Substitute holiday: a Sunday
  holiday moves to the next non-holiday day. Citizens' holiday: a day
  between two holidays. Respect for the Aged Day is Sep 15 before 2003
  and the 3rd Monday of September from 2003. Equinoxes by the National Astronomical
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

* USD-GOVT, US government bond market: the holidays SIFMA recommends
  (sifma.org, "Holiday Schedule"; QuantLib ``UnitedStates(GovernmentBond)``).
  The Fed list plus Good Friday, with Jul 4, Juneteenth and Christmas on a
  Saturday observed on the Friday before (New Year's Day and Veterans Day
  on a Saturday are not moved). A Good Friday that is the first Friday of
  the month, the Employment Situation release day, gets a noon early close
  instead of a full close (2007, 2010, 2012, 2015, 2021, 2023, 2026); for
  future years this is the forecast, and SIFMA confirms each year. One-off
  closures: 2001-09-11/12, 2004-06-11, 2012-10-30, 2018-12-05, 2025-01-09.
  Early closes (usually 14:00 New York before a holiday) are not modelled:
  an early-close day is a business day here, which is right for settlement.
  Use it for Treasury and agency bond schedules; swaps and FX keep ``USD``.
  Aliases: ``SIFMA``. It is not a currency, so it is in ``CALENDAR_CODES``
  but not in ``CURRENCIES``, and FX spot dates never use it.

Limits: one-off closures announced after this code was written are not
known to the rules. Only these seven currencies are shipped.

Adding a one-off closure (a state funeral, a jubilee, an emergency market
closure). There are two routes:

1. Permanent: once the closure is officially announced (gazette, central
   bank or TARGET2 notice), add it to the currency's rule function below
   as a dated ``if y == YYYY`` line, the way 2022-09-19 (GBP) and
   2022-09-22 (AUD) already are, cite the notice in this docstring, and
   add a test that pins the date. This is the only route that changes the
   shipped calendars.
2. Runtime, until the code is updated: ``add_closure("GBP", date)`` merges
   the date into every later ``holidays`` call for that currency in this
   process. In tests use the context manager
   ``extra_closures({"GBP": [...]})``, which restores the previous state on
   exit so tests stay independent. ``clear_closures()`` drops all runtime
   additions.

The rules-only sets are cached; runtime closures are merged on top at call
time, so adding one never needs a cache flush.

History: this generator lived in ``fx_options.holidays`` until round 4. It
moved here so the equity credit package (ISDA CDS dates) and the FX
packages share one copy with no cross-package import;
``fx_options.holidays`` re-exports it.

Worked example: ``holidays("USD", 2026)`` has 10 days; Jul 4 is a Saturday
so the Fed is open on Friday Jul 3.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache

__all__ = [
    "CALENDARS",
    "CALENDAR_CODES",
    "CURRENCIES",
    "MAX_YEAR",
    "MIN_YEAR",
    "ROLL_CONVENTIONS",
    "T_PLUS_1_PAIRS",
    "HolidayCalendar",
    "add_business_days",
    "add_closure",
    "adjust_following",
    "clear_closures",
    "easter_sunday",
    "extra_closures",
    "fx_spot_date",
    "get_calendar",
    "holidays",
    "is_business_day",
    "joint_calendar",
    "pair_holidays",
    "spot_lag",
]

Calendar = str | Iterable[str] | None

MIN_YEAR, MAX_YEAR = 2000, 2099
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


def _sat_fri_sun_mon(d: dt.date) -> dt.date:
    if d.weekday() == 5:
        return d - _DAY
    if d.weekday() == 6:
        return d + _DAY
    return d


#: One-off SIFMA full closures. 2001-09-11/12 (markets closed after the
#: attacks; TBMA reopened the bond market on 13 Sep), 2004-06-11 (President
#: Reagan's funeral), 2012-10-30 (Hurricane Sandy), 2018-12-05 (President
#: G. H. W. Bush's funeral), 2025-01-09 (President Carter's funeral).
_GOVT_SPECIAL = (D(2001, 9, 11), D(2001, 9, 12), D(2004, 6, 11), D(2012, 10, 30),
                 D(2018, 12, 5), D(2025, 1, 9))


def _usd_govt(y: int) -> set[dt.date]:
    out = {_nth_weekday(y, 1, 0, 3), _nth_weekday(y, 2, 0, 3), _nth_weekday(y, 5, 0, -1),
           _nth_weekday(y, 9, 0, 1), _nth_weekday(y, 10, 0, 2), _nth_weekday(y, 11, 3, 4)}
    for d in (D(y, 1, 1), D(y, 11, 11)):  # Sunday -> Monday; Saturday not moved
        if d.weekday() != 5:
            out.add(d + _DAY if d.weekday() == 6 else d)
    fixed = [D(y, 7, 4), D(y, 12, 25)] + ([D(y, 6, 19)] if y >= 2022 else [])
    out |= {_sat_fri_sun_mon(d) for d in fixed}
    good_friday = easter_sunday(y) - 2 * _DAY
    if good_friday.day > 7:  # first-Friday Good Friday: payrolls day, early close
        out.add(good_friday)
    out |= {d for d in _GOVT_SPECIAL if d.year == y}
    return out


def _eur(y: int) -> set[dt.date]:
    e = easter_sunday(y)
    out = {D(y, 1, 1), e - 2 * _DAY, e + _DAY, D(y, 5, 1), D(y, 12, 25), D(y, 12, 26)}
    if y in (2000, 2001):  # TARGET closed on 31 Dec in its first years (to 2001)
        out.add(D(y, 12, 31))
    return out


def _gbp(y: int) -> set[dt.date]:
    e = easter_sunday(y)
    out = {_next_weekday(D(y, 1, 1)), e - 2 * _DAY, e + _DAY, _nth_weekday(y, 8, 0, -1)}
    out |= _xmas_boxing(y)
    out.add(D(2020, 5, 8) if y == 2020 else _nth_weekday(y, 5, 0, 1))
    # Spring holiday moved for a jubilee in 2002, 2012 and 2022.
    if y == 2002:
        out |= {D(2002, 6, 3), D(2002, 6, 4)}
    elif y == 2012:
        out |= {D(2012, 6, 4), D(2012, 6, 5)}
    elif y == 2022:
        out |= {D(2022, 6, 2), D(2022, 6, 3), D(2022, 9, 19)}
    else:
        out.add(_nth_weekday(y, 5, 0, -1))
    if y == 2011:
        out.add(D(2011, 4, 29))
    if y == 2023:
        out.add(D(2023, 5, 8))
    return out


def _equinox(y: int, base: float) -> int:
    return int(base + 0.242194 * (y - 1980)) - (y - 1980) // 4


def _jpy(y: int) -> set[dt.date]:
    nat = {D(y, 1, 1), D(y, 2, 11), D(y, 4, 29), D(y, 5, 3), D(y, 5, 4), D(y, 5, 5),
           D(y, 11, 3), D(y, 11, 23), _nth_weekday(y, 1, 0, 2),
           _nth_weekday(y, 9, 0, 3) if y >= 2003 else D(y, 9, 15),
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
#: Currency settlement centres (FX spot, swaps, money markets).
CURRENCIES = tuple(_RULES)
_RULES["USD-GOVT"] = _usd_govt
#: Every calendar code: the currencies plus ``USD-GOVT`` (SIFMA bond market).
CALENDAR_CODES = tuple(_RULES)


_EXTRA: dict[str, set[dt.date]] = {c: set() for c in CALENDAR_CODES}


def _check_ccy(currency: str) -> str:
    ccy = currency.upper()
    if ccy not in _RULES:
        raise ValueError(f"no calendar for {currency!r}; shipped: {CALENDAR_CODES}")
    return ccy


@cache
def _rule_holidays(ccy: str, year: int) -> frozenset[dt.date]:
    return frozenset(d for d in _RULES[ccy](year) if d.weekday() < 5 and d.year == year)


def holidays(currency: str, year: int) -> frozenset[dt.date]:
    """Weekday settlement holidays of ``currency`` in ``year``.

    Rules-generated dates plus any runtime closures (``add_closure``).
    Weekend dates are dropped: the business-day logic already skips them.
    """
    ccy = _check_ccy(currency)
    if isinstance(year, bool) or not (isinstance(year, int) and MIN_YEAR <= year <= MAX_YEAR):
        raise ValueError(f"year must be an int in [{MIN_YEAR}, {MAX_YEAR}], got {year!r}")
    base = _rule_holidays(ccy, year)
    extra = frozenset(d for d in _EXTRA[ccy] if d.year == year)
    return base | extra if extra else base


def add_closure(currency: str, day: dt.date) -> None:
    """Add a one-off weekday closure for ``currency`` for this process.

    Raises on a weekend (already closed), on a year outside
    ``[MIN_YEAR, MAX_YEAR]``, on a datetime and on an unknown currency.
    Adding a date that is already a holiday is a no-op.
    """
    ccy = _check_ccy(currency)
    if not isinstance(day, dt.date) or isinstance(day, dt.datetime):
        raise TypeError(f"day must be a datetime.date, got {type(day).__name__}")
    if not MIN_YEAR <= day.year <= MAX_YEAR:
        raise ValueError(f"closure year must be in [{MIN_YEAR}, {MAX_YEAR}], got {day}")
    if day.weekday() >= 5:
        raise ValueError(f"{day} is a weekend day; it is already closed")
    _EXTRA[ccy].add(day)


def clear_closures(currency: str | None = None) -> None:
    """Drop runtime closures for one currency, or for all when ``None``."""
    for c in CALENDAR_CODES if currency is None else (_check_ccy(currency),):
        _EXTRA[c].clear()


@contextmanager
def extra_closures(closures: Mapping[str, Iterable[dt.date]]) -> Iterator[None]:
    """Test hook: add closures inside a ``with`` block, restore on exit.

    >>> import datetime as dt
    >>> d = dt.date(2031, 5, 6)
    >>> with extra_closures({"GBP": [d]}):
    ...     d in holidays("GBP", 2031)
    True
    >>> d in holidays("GBP", 2031)
    False
    """
    saved = {c: set(v) for c, v in _EXTRA.items()}
    try:
        for ccy, days in closures.items():
            for d in days:
                add_closure(ccy, d)
        yield
    finally:
        for c, v in saved.items():
            _EXTRA[c] = v


def _centres(calendar: Calendar) -> tuple[str, ...]:
    if calendar is None:
        return ()
    if isinstance(calendar, str):
        key = calendar.upper().strip()
        if key in _RULES:
            return (key,)
        if "+" in key or "/" in key:
            return _centres([p for p in re.split(r"[+/]", key) if p])
        code = key
        if not code or len(code) % 3:
            raise ValueError(f"calendar code must be 3, 6, 9... letters, got {calendar!r}")
        calendar = [code[i:i + 3] for i in range(0, len(code), 3)]
    out = tuple(_check_ccy(c) for c in calendar)
    if not out:
        raise ValueError("calendar must name at least one currency")
    return out


def is_business_day(day: dt.date, calendar: Calendar = None) -> bool:
    """True when ``day`` is a weekday open in every named centre.

    ``calendar`` is a currency code (``"USD"``), a joint code
    (``"EURUSD"``, ``"EUR+USD"``), an iterable of codes, or ``None`` for
    weekends only. A joint calendar is the union of holidays (all centres
    must be open), the usual convention for multi-centre schedules.

    >>> import datetime as dt
    >>> is_business_day(dt.date(2026, 11, 11), "USD")  # Veterans Day
    False
    >>> is_business_day(dt.date(2026, 7, 3), "USD")  # Jul 4 is a Saturday; the Fed opens
    True
    """
    if day.weekday() >= 5:
        return False
    return all(day not in holidays(c, day.year) for c in _centres(calendar))


def adjust_following(day: dt.date, calendar: Calendar = None) -> dt.date:
    """Following roll (ISDA 2006 Definitions s.4.12): first business day on or after ``day``."""
    centres = _centres(calendar)
    while not is_business_day(day, centres or None):
        day += _DAY
    return day


def add_business_days(day: dt.date, n: int, calendar: Calendar = None) -> dt.date:
    """Move ``n >= 0`` business days forward (``n = 0`` returns ``day`` unchanged)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    centres = _centres(calendar)
    while n > 0:
        day += _DAY
        if is_business_day(day, centres or None):
            n -= 1
    return day


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


# ---------------------------------------------------------------------- #
# Calendar objects: the one calendar for schedules, rolls and FX spot
# ---------------------------------------------------------------------- #
ROLL_CONVENTIONS: tuple[str, ...] = (
    "following",
    "modified_following",
    "preceding",
    "unadjusted",
)

_ALIASES = {"TARGET": "EUR", "TARGET2": "EUR", "SIFMA": "USD-GOVT"}


@dataclass(frozen=True)
class HolidayCalendar:
    """Business-day calendar over one or more settlement centres.

    Plain idea: a day is a business day when it is a weekday and no named
    centre is closed. ``centres=()`` is the weekend-only calendar. A joint
    calendar (``EUR+USD``) is the union of the centres' holidays, the
    convention for multi-centre schedules (ISDA 2006 Definitions s.4.10).

    Roll conventions follow ISDA 2006 s.4.12: ``following``,
    ``modified_following`` (following unless that leaves the month, then
    preceding), ``preceding`` and ``unadjusted``.

    This class replaces ``qp_core.calendars.BusinessCalendar``, which knew
    fixed-date holidays only. It has the same methods.
    """

    name: str
    centres: tuple[str, ...] = ()

    def holidays(self, year: int) -> frozenset[dt.date]:
        """Weekday holidays of every centre in ``year`` (the union)."""
        out: frozenset[dt.date] = frozenset()
        for c in self.centres:
            out = out | holidays(c, year)
        return out

    def is_business_day(self, date: dt.date) -> bool:
        """True when ``date`` is a weekday open in every centre."""
        return is_business_day(date, self.centres or None)

    def adjust(self, date: dt.date, convention: str = "modified_following") -> dt.date:
        """Roll ``date`` to a business day under ``convention``."""
        if convention not in ROLL_CONVENTIONS:
            raise ValueError(
                f"unknown roll convention {convention!r}; expected one of {ROLL_CONVENTIONS}"
            )
        if convention == "unadjusted" or self.is_business_day(date):
            return date
        if convention == "preceding":
            return self._step(date, -1)
        rolled = self._step(date, +1)
        if convention == "modified_following" and rolled.month != date.month:
            return self._step(date, -1)
        return rolled

    def _step(self, date: dt.date, direction: int) -> dt.date:
        d = date
        while not self.is_business_day(d):
            d += dt.timedelta(days=direction)
        return d

    def add_business_days(self, date: dt.date, n: int) -> dt.date:
        """Move ``n >= 0`` business days forward.

        ``n = 0`` returns ``date`` rolled forward to a business day (unlike
        the module function :func:`add_business_days`, which returns it
        unchanged).
        """
        if int(n) != n or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n}")
        d = self._step(date, +1)
        for _ in range(int(n)):
            d = self._step(d + _DAY, +1)
        return d

    def business_days_between(self, start: dt.date, end: dt.date) -> int:
        """Number of business days in ``(start, end]``."""
        if end < start:
            raise ValueError(f"end {end} precedes start {start}")
        count, d = 0, start
        while d < end:
            d += _DAY
            count += self.is_business_day(d)
        return count


CALENDARS: dict[str, HolidayCalendar] = {
    "WEEKEND": HolidayCalendar("WEEKEND"),
    **{c: HolidayCalendar(c, (c,)) for c in CALENDAR_CODES},
}


def get_calendar(name: str | HolidayCalendar | Iterable[str] | None) -> HolidayCalendar:
    """Look up a calendar.

    ``None`` or ``"WEEKEND"`` gives weekends only. A currency code
    (``"USD"``), ``"TARGET"``, a joint code (``"EUR+USD"``, ``"EURUSD"``,
    ``"EUR/USD"``) or an iterable of codes gives the rule-based calendar.
    """
    if name is None:
        return CALENDARS["WEEKEND"]
    if isinstance(name, HolidayCalendar):
        return name
    if not isinstance(name, str):
        return joint_calendar(*name)
    key = name.upper().strip()
    key = _ALIASES.get(key, key)
    if key in CALENDARS:
        return CALENDARS[key]
    parts = [p for p in re.split(r"[+/ ]", key) if p]
    if len(parts) == 1 and len(key) % 3 == 0 and len(key) > 3:
        parts = [key[i:i + 3] for i in range(0, len(key), 3)]
    if len(parts) > 1:
        return joint_calendar(*parts)
    raise ValueError(f"unknown calendar {name!r}; expected one of {tuple(CALENDARS)}")


def joint_calendar(*names: str | HolidayCalendar) -> HolidayCalendar:
    """Calendar whose business days are business days in every input."""
    centres: list[str] = []
    for n in names:
        for c in get_calendar(n).centres:
            if c not in centres:
                centres.append(c)
    if not centres:
        return CALENDARS["WEEKEND"]
    if len(centres) == 1:
        return CALENDARS[centres[0]]
    return HolidayCalendar("+".join(centres), tuple(centres))


# ---------------------------------------------------------------------- #
# FX spot dates
# ---------------------------------------------------------------------- #
#: Pairs that settle T+1 against USD (Clark 2011, ch. 1; Wystup, ch. 1).
T_PLUS_1_PAIRS = frozenset({"USDCAD", "USDTRY", "USDRUB", "USDPHP"})


def _pair_codes(pair: str | tuple[str, str]) -> tuple[str, str]:
    if isinstance(pair, str):
        p = pair.upper().replace("/", "").strip()
        if len(p) != 6 or not p.isalpha():
            raise ValueError(f"pair must be six letters, got {pair!r}")
        return p[:3], p[3:]
    base, quote = (str(c).upper() for c in pair)
    if len(base) != 3 or len(quote) != 3:
        raise ValueError(f"pair must be two 3-letter codes, got {pair!r}")
    return base, quote


def spot_lag(pair: str | tuple[str, str]) -> int:
    """Business days from trade date to spot: 1 for USDCAD, USDTRY, USDRUB,
    USDPHP (either order), 2 otherwise."""
    b, q = _pair_codes(pair)
    return 1 if (b + q in T_PLUS_1_PAIRS or q + b in T_PLUS_1_PAIRS) else 2


def _shipped(codes: Iterable[str]) -> tuple[str, ...]:
    return tuple(c for c in codes if c in _RULES)


def fx_spot_date(trade_date: dt.date, pair: str | tuple[str, str],
                 lag: int | None = None) -> dt.date:
    """Spot value date of an FX pair traded on ``trade_date``.

    Plain idea: count ``lag`` good business days in the non-USD centres,
    then roll forward until every centre, and New York, is open.

    Formal rule (Clark 2011, ch. 1; Wystup 2017, ch. 1):

    1. ``lag`` is :func:`spot_lag` unless given (T+2, or T+1 for USDCAD,
       USDTRY, USDRUB, USDPHP).
    2. The lag days are counted on the joint calendar of the non-USD
       currencies of the pair. A USD holiday on T+1 does not stop the count
       for a USD pair: EURUSD traded the day before 4 July still settles on
       the next day that both centres are open.
    3. The candidate is rolled forward (following) until it is a business
       day in both currencies and in USD. For a cross such as EURJPY the
       USD leg of the two settlements still has to clear, so a USD holiday
       cannot be a spot date.

    Currencies with no shipped calendar (``CURRENCIES``) count weekends
    only; check them against the local market before relying on the date.

    Worked example: EURUSD traded Wed 2 Jul 2025. Two TARGET days give
    Fri 4 Jul, a USD holiday, so spot is Mon 7 Jul.
    """
    b, q = _pair_codes(pair)
    if b == q:
        raise ValueError(f"pair needs two different currencies, got {b}{q}")
    n = spot_lag((b, q)) if lag is None else lag
    if isinstance(n, bool) or int(n) != n or n < 0:
        raise ValueError(f"lag must be a non-negative integer, got {lag!r}")
    # Count on the shipped non-USD centres; none shipped means weekends only.
    count_cal = joint_calendar(*_shipped(c for c in (b, q) if c != "USD"))
    d = trade_date
    for _ in range(int(n)):
        d = count_cal.add_business_days(d + _DAY, 0)
    settle = joint_calendar(*_shipped((b, q, "USD")))
    return settle.adjust(d, "following")
