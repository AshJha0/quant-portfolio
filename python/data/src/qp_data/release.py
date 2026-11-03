"""Publication lags: the first date on which an observation could be known.

Plain idea
----------
A value dated ``t`` is usually not public on day ``t``. The ten-year
Treasury yield for Monday appears in the Fed's H.15 on Tuesday; March CPI
appears in mid-April. A backtest that uses a value before its release date
looks ahead. This module turns every observation date into a *release date*
and offers one check, :func:`check_release`, that research code (W6) and the
store share.

Formal statement
----------------
For series ``s`` with rule ``R_s``, the release date of the observation dated
``t`` is ``r = R_s(t)``. A value may be used for a decision taken on date
``d`` only if ``r <= d``. Rules are deterministic functions of ``t`` built
from the issuer's published schedule; where the schedule varies from month
to month, the rule is a conservative *upper bound* (never earlier than the
true date in normal months) plus an explicit override table for known
exceptions. Without the FRED API key there are no ALFRED vintages (owner
decision D6), so the first-release value itself is not available; only the
timing is enforced. Revisions are tracked from the store's own snapshots.

Release times inside the day are not modelled: an observation released at
16:15 New York time on ``r`` counts as known on ``r``. A strategy that trades
at the close of ``r`` must therefore fill at ``r + 1`` (the repository's t+1
fill rule, ``qp_core.execution``).

Worked example
--------------
Monday 1 July 2024 for DGS10 (H.15, next US business day); the Friday 5 July
EURUSD H.10 noon buying rate (weekly, the following Monday); March 2024 CPI
(bounded at the 20th of the next month)::

    >>> import datetime as dt
    >>> release_date("fred:DGS10", dt.date(2024, 7, 1))
    datetime.date(2024, 7, 2)
    >>> release_date("fred:DGS10", dt.date(2024, 7, 3))  # 4 July is a US holiday
    datetime.date(2024, 7, 5)
    >>> release_date("fred:DEXUSEU", dt.date(2024, 7, 5))
    datetime.date(2024, 7, 8)
    >>> release_date("fred:CPIAUCSL", dt.date(2024, 3, 1))
    datetime.date(2024, 4, 22)

(20 April 2024 was a Saturday, so the bound rolls to Monday the 22nd. BLS
actually published March 2024 CPI on 10 April; the bound is later, which is
the safe direction.)

Sources (evidence class: issuer's published schedule)
-----------------------------------------------------
- H.15 Selected Interest Rates, daily update "for the prior business day",
  federalreserve.gov/releases/h15.
- H.10 Foreign Exchange Rates, released weekly on Monday,
  federalreserve.gov/releases/h10.
- SOFR: published by the NY Fed at about 08:00 ET on the next business day,
  newyorkfed.org/markets/reference-rates/sofr.
- euro short-term rate: published at 08:00 CET on the next TARGET2 day,
  ecb.europa.eu (euro short-term rate, publication).
- SONIA: published by the Bank of England at 09:00 on the next London
  business day, bankofengland.co.uk/markets/sonia-benchmark.
- ECB euro reference rates: 14:15 CET concertation, published about 16:00 CET
  the same day (the ECB series title says "2.15 pm (C.E.T.)").
- US Treasury daily par yield curve: published the same business day after
  the close, home.treasury.gov (interest rate statistics).
- CBOE VIX history: close of the same day.
- BLS CPI release schedule, bls.gov/schedule/news_release/cpi.htm. Known
  exceptions in the override table: September 2013 CPI (released 30 October
  2013) and September 2025 CPI (released 24 October 2025), both delayed by
  federal government shutdowns.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from qp_core.holidays import add_business_days, adjust_following

from .errors import UnknownReleaseRule

__all__ = [
    "BusinessDayLag",
    "CalendarDayLag",
    "MonthEndLag",
    "MonthlyBound",
    "ReleaseRule",
    "WeeklyRelease",
    "check_release",
    "register_rule",
    "release_date",
    "rule_for",
]


class ReleaseRule:
    """Maps an observation date to its release date."""

    description: str = ""

    def __call__(self, obs: dt.date) -> dt.date:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class BusinessDayLag(ReleaseRule):
    """Released ``n`` business days after the observation on ``calendar``."""

    n: int
    calendar: str
    description: str = ""

    def __call__(self, obs: dt.date) -> dt.date:
        if self.n == 0:
            return adjust_following(obs, self.calendar)
        return add_business_days(obs, self.n, self.calendar)


@dataclass(frozen=True)
class CalendarDayLag(ReleaseRule):
    """Released ``n`` calendar days after the observation (24/7 markets)."""

    n: int
    description: str = ""

    def __call__(self, obs: dt.date) -> dt.date:
        return obs + dt.timedelta(days=self.n)


@dataclass(frozen=True)
class WeeklyRelease(ReleaseRule):
    """All observations of a Monday-to-Sunday week are released on the next
    ``weekday`` (0 = Monday) after that week, rolled forward over holidays."""

    weekday: int
    calendar: str
    description: str = ""

    def __call__(self, obs: dt.date) -> dt.date:
        week_end = obs + dt.timedelta(days=6 - obs.weekday())  # Sunday of the obs week
        day = week_end + dt.timedelta(days=1 + self.weekday)
        return adjust_following(day, self.calendar)


@dataclass(frozen=True)
class MonthlyBound(ReleaseRule):
    """A monthly observation (dated the first of its month, as FRED dates it)
    is released no later than ``day`` of the month ``months`` after it,
    rolled forward over holidays, unless ``overrides`` names the true date."""

    months: int
    day: int
    calendar: str
    overrides: Mapping[dt.date, dt.date] = field(default_factory=dict)
    description: str = ""

    def __call__(self, obs: dt.date) -> dt.date:
        first = obs.replace(day=1)
        if first in self.overrides:
            return self.overrides[first]
        m = first.month - 1 + self.months
        target = dt.date(first.year + m // 12, m % 12 + 1, self.day)
        return adjust_following(target, self.calendar)


@dataclass(frozen=True)
class MonthEndLag(ReleaseRule):
    """A monthly average dated the first of its month is known ``n`` business
    days after the month's last calendar day."""

    n: int
    calendar: str
    description: str = ""

    def __call__(self, obs: dt.date) -> dt.date:
        nxt = (obs.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        return add_business_days(nxt - dt.timedelta(days=1), self.n, self.calendar)


_H15 = BusinessDayLag(1, "USD", "H.15: prior business day, published the next US business day")
_OECD_MEI = MonthlyBound(2, 1, "USD", description="OECD MEI monthly: bounded at the 1st of month m+2")
_RULES: dict[str, ReleaseRule] = {}
_PREFIX_RULES: list[tuple[str, ReleaseRule]] = [
    ("fred:DGS", _H15),
    ("fred:DTB", _H15),
    ("fred:DEX", WeeklyRelease(0, "USD", "H.10: weekly, the Monday after the observation week")),
    ("coinbase:", CalendarDayLag(1, "UTC daily candle, complete at 00:00 UTC the next day")),
    ("binance:", CalendarDayLag(1, "UTC daily candle, complete at 00:00 UTC the next day")),
    # The library posts a month's factors some weeks after month end and has
    # no fixed schedule; 45 US business days (about two months) is a
    # conservative upper bound, never earlier than the real posting.
    # OECD MEI monthly series (interbank rates, CPI): national releases
    # land within about six weeks; bounded at the 1st of month m+2 (the
    # convention in research/lib/excess_returns.as_known).
    ("fred:IR3TIB01", _OECD_MEI),
    ("french:", MonthEndLag(45, "USD", "French Data Library: monthly update, bounded at 45 business days")),
]


def register_rule(series: str, rule: ReleaseRule) -> None:
    """Add or replace the rule for one exact series key."""
    _RULES[series] = rule


for _key, _rule in {
    "fred:DFF": _H15,
    "fred:SOFR": BusinessDayLag(1, "USD", "SOFR: 08:00 ET next US business day"),
    "fred:FEDFUNDS": MonthEndLag(
        1, "USD", "monthly average of daily H.15 values; known the business day after month end"
    ),
    "fred:CPIAUCSL": MonthlyBound(
        1, 20, "USD",
        overrides={dt.date(2013, 9, 1): dt.date(2013, 10, 30), dt.date(2025, 9, 1): dt.date(2025, 10, 24)},
        description="BLS CPI: bound at the 20th of the next month; shutdown delays overridden",
    ),
    "nyfed:SOFR": BusinessDayLag(1, "USD", "SOFR: 08:00 ET next US business day"),
    "ecb:ESTR": BusinessDayLag(1, "EUR", "euro short-term rate: 08:00 CET next TARGET2 day"),
    "boe:IUDSOIA": BusinessDayLag(1, "GBP", "SONIA: 09:00 next London business day"),
    "ust:par": BusinessDayLag(0, "USD", "Treasury par yields: same business day, after the close"),
    "cboe:VIX": BusinessDayLag(0, "USD", "VIX close: same day"),
}.items():
    register_rule(_key, _rule)


def rule_for(series: str) -> ReleaseRule:
    """The rule for ``series``: an exact key first, then the prefix table.

    ECB reference rates (``ecb:EXR:*``) and Frankfurter (which republishes
    them) are released the same TARGET2 day.
    """
    if series in _RULES:
        return _RULES[series]
    if series.startswith(("ecb:EXR:", "frankfurter:")):
        return BusinessDayLag(0, "EUR", "ECB reference rate: 14:15 CET fixing, same day")
    for prefix, rule in _PREFIX_RULES:
        if series.startswith(prefix):
            return rule
    if series.startswith("fred:") and series.endswith("CPIALLMINMEI"):
        return _OECD_MEI
    raise UnknownReleaseRule(
        f"no publication-lag rule for {series!r}; add one with register_rule() "
        "from the issuer's release calendar before using the series"
    )


def release_date(series: str, obs: dt.date) -> dt.date:
    """Release date of the observation of ``series`` dated ``obs``."""
    if isinstance(obs, dt.datetime):
        obs = obs.date()
    return rule_for(series)(obs)


def check_release(series: str, obs_dates: Iterable[dt.date], used_on: Iterable[dt.date]) -> None:
    """Raise ``ValueError`` if any observation is used before its release date.

    ``obs_dates[i]`` is the date of an observation and ``used_on[i]`` the date
    a decision used it. This is the check W6 runs on every study input.
    """
    bad = []
    for obs, used in zip(obs_dates, used_on, strict=True):
        r = release_date(series, obs)
        u = used.date() if isinstance(used, dt.datetime) else used
        if u < r:
            bad.append((obs, u, r))
    if bad:
        head = ", ".join(f"obs {o} used {u} released {r}" for o, u, r in bad[:3])
        raise ValueError(f"{series}: {len(bad)} value(s) used before release ({head})")
