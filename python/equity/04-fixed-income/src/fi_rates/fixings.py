"""Rate indices and a fixings store (W13a).

Plain idea
----------
A floating coupon needs index rates. Rates already published are facts
and come from a store; rates not yet published come from a curve. The
index definition says which calendar, day count and publication rule
apply, so a coupon knows which dates to look up.

Formal statement
----------------
:class:`OvernightIndex` is an overnight rate (SOFR, ESTR, SONIA) fixed for
each business day ``d`` of its calendar and accruing over ``[d, next
business day)``. :class:`IborIndex` is a term rate (EURIBOR 3M/6M, USD Term
SOFR 3M) fixed ``fixing_days`` business days before its value date, for a
period of ``tenor_months``. :class:`FixingStore` maps ``(index name,
fixing date)`` to a decimal rate; a lookup of a missing past fixing raises
:class:`MissingFixingError` instead of silently falling back to a curve.

Conventions, with sources:

========  ========  ===========  =========  ==========================================
Index     Currency  Calendar     Day count  Source
========  ========  ===========  =========  ==========================================
SOFR      USD       USD-GOVT     ACT/360    NY Fed SOFR methodology; ARRC (2020)
ESTR      EUR       TARGET       ACT/360    ECB euro short-term rate methodology
SONIA     GBP       GBP          ACT/365F   BoE SONIA key features and policies
EURIBOR   EUR       TARGET       ACT/360    EMMI EURIBOR benchmark statement (T-2)
TERMSOFR  USD       USD-GOVT     ACT/360    CME Term SOFR reference rates (T-2)
========  ========  ===========  =========  ==========================================

SOFR is published on U.S. Government Securities business days, which is
the SIFMA calendar (``USD-GOVT`` in :mod:`qp_core.holidays`).

Worked example
--------------
>>> import datetime as dt
>>> store = FixingStore()
>>> store.add("SOFR", dt.date(2024, 1, 2), 0.0540)
>>> store.get("SOFR", dt.date(2024, 1, 2))
0.054
>>> store.has("SOFR", dt.date(2024, 1, 3))
False
"""

from __future__ import annotations

import csv
import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from qp_core.holidays import HolidayCalendar, get_calendar

__all__ = [
    "ESTR",
    "EURIBOR_3M",
    "EURIBOR_6M",
    "SOFR",
    "SONIA",
    "TERM_SOFR_3M",
    "FixingStore",
    "IborIndex",
    "MissingFixingError",
    "OvernightIndex",
]

_DAY_BASIS = {"ACT/360": 360.0, "ACT/365F": 365.0}


class MissingFixingError(KeyError):
    """A fixing that should already be published is not in the store."""


@dataclass(frozen=True)
class OvernightIndex:
    """Overnight index: one fixing per business day of ``calendar``."""

    name: str
    currency: str
    calendar_name: str
    day_count: str

    def __post_init__(self) -> None:
        if self.day_count not in _DAY_BASIS:
            raise ValueError(f"day_count must be ACT/360 or ACT/365F, got {self.day_count!r}")

    @property
    def calendar(self) -> HolidayCalendar:
        return get_calendar(self.calendar_name)

    @property
    def basis(self) -> float:
        """Days in the year for simple accrual (360 or 365)."""
        return _DAY_BASIS[self.day_count]


@dataclass(frozen=True)
class IborIndex:
    """Term index fixed ``fixing_days`` business days before its value date."""

    name: str
    currency: str
    calendar_name: str
    day_count: str
    tenor_months: int
    fixing_days: int = 2
    convention: str = "modified_following"
    end_of_month: bool = True

    def __post_init__(self) -> None:
        if self.day_count not in _DAY_BASIS:
            raise ValueError(f"day_count must be ACT/360 or ACT/365F, got {self.day_count!r}")

    @property
    def calendar(self) -> HolidayCalendar:
        return get_calendar(self.calendar_name)

    @property
    def basis(self) -> float:
        return _DAY_BASIS[self.day_count]

    def fixing_date(self, value_date: dt.date) -> dt.date:
        """Fixing date for a period starting on ``value_date``."""
        from .schedule import shift_business_days

        return shift_business_days(self.calendar, value_date, -self.fixing_days)

    def maturity(self, value_date: dt.date) -> dt.date:
        """End of the index's own deposit period starting on ``value_date``."""
        from .schedule import _add_months, _month_end

        eom = self.end_of_month and value_date.day == _month_end(value_date.year,
                                                                  value_date.month)
        raw = _add_months(value_date, self.tenor_months, value_date.day, eom)
        return self.calendar.adjust(raw, self.convention)


SOFR = OvernightIndex("SOFR", "USD", "USD-GOVT", "ACT/360")
ESTR = OvernightIndex("ESTR", "EUR", "EUR", "ACT/360")
SONIA = OvernightIndex("SONIA", "GBP", "GBP", "ACT/365F")
EURIBOR_3M = IborIndex("EURIBOR-3M", "EUR", "EUR", "ACT/360", 3)
EURIBOR_6M = IborIndex("EURIBOR-6M", "EUR", "EUR", "ACT/360", 6)
TERM_SOFR_3M = IborIndex("TERMSOFR-3M", "USD", "USD-GOVT", "ACT/360", 3)


@dataclass
class FixingStore:
    """Published fixings, ``{index name: {fixing date: decimal rate}}``."""

    _data: dict[str, dict[dt.date, float]] = field(default_factory=dict)

    def add(self, index: str, date: dt.date, rate: float) -> None:
        """Record one fixing; a different value for an existing date raises."""
        r = float(rate)
        if r != r or r in (float("inf"), float("-inf")):
            raise ValueError(f"fixing for {index} on {date} must be finite, got {rate}")
        series = self._data.setdefault(index, {})
        old = series.get(date)
        if old is not None and old != r:
            raise ValueError(f"{index} {date}: already stored {old}, refusing {r}")
        series[date] = r

    def add_many(self, index: str, fixings: Mapping[dt.date, float] | Iterable[tuple[dt.date, float]]) -> None:
        """Record several fixings (a mapping or ``(date, rate)`` pairs)."""
        items = fixings.items() if isinstance(fixings, Mapping) else fixings
        for d, r in items:
            self.add(index, d, r)

    def has(self, index: str, date: dt.date) -> bool:
        return date in self._data.get(index, {})

    def get(self, index: str, date: dt.date) -> float:
        """The fixing, or :class:`MissingFixingError`."""
        try:
            return self._data[index][date]
        except KeyError:
            raise MissingFixingError(f"no {index} fixing for {date}") from None

    def series(self, index: str) -> dict[dt.date, float]:
        """A copy of one index's fixings."""
        return dict(self._data.get(index, {}))

    @classmethod
    def from_csv(cls, path: str | Path, index: str, *, percent: bool = True,
                 date_col: str = "date", rate_col: str = "rate") -> FixingStore:
        """Load ``date,rate`` rows (ISO dates). ``percent=True`` divides by 100,
        the format of the NY Fed, ECB and BoE downloads."""
        store = cls()
        with Path(path).open(newline="") as fh:
            for row in csv.DictReader(fh):
                r = float(row[rate_col])
                store.add(index, dt.date.fromisoformat(row[date_col]), r / 100.0 if percent else r)
        return store
