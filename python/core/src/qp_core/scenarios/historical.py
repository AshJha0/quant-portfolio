"""Named historical stress scenarios from free, public series (W14c).

The plain idea
--------------
A stress test asks "what would this book lose if a known bad episode
happened again?". The answer is only as good as the shock vector, so each
scenario here is built from the published daily series themselves, not from
round numbers: the 2008 shock to the 10-year Treasury zero rate is the
Fed's own fitted-curve value on 10 October 2008 minus its value on 12
September 2008. A test checks every move against the frozen source
observations to the last bit.

The formal statement
--------------------
A scenario is a window ``[t0, t1]`` and a set of series. For each series
``x`` the shock is::

    absolute move   dx = x(t1) - x(t0)           (rates, in decimal; VIX, in points)
    log move        dx = ln(x(t1) / x(t0))       (FX rates)

The shock is applied to a quote ``q`` that the series stands for as
``q + dx`` (absolute) or ``q exp(dx)`` (log), so a snapshot equal to the
source on ``t0`` is moved exactly onto the source on ``t1``. Daily paths
(``path_shocks``) give the cumulative move to each date in the window for
path-dependent stress and for reverse-stress searches. BCBS (2018) *Stress
testing principles* (principle 5: scenarios should include historical
episodes) and the EBA stress-test methodology (2023, section 2) motivate the
set; the episodes are the ones listed in the plan (W14c).

Sources (all free, no account; retrieved 2026-11-03)
----------------------------------------------------
* ``USD_ZERO_2Y``, ``USD_ZERO_10Y``: Gurkaynak, Sack and Wright (2007)
  fitted Treasury curve, series SVENY02 and SVENY10 (continuously compounded
  zero yields, percent, divided by 100 here),
  https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv
* ``VIX``: CBOE VIX daily close,
  https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
* ``EURUSD``, ``EURCHF``, ``EURGBP``, ``EURJPY``: ECB euro foreign exchange
  reference rates (quote currency per 1 EUR), dataset EXR, D.*.EUR.SP00.A,
  https://data-api.ecb.europa.eu/service/data/EXR/
* ``GBP_ZERO_10Y``, ``GBP_ZERO_20Y``: Bank of England nominal spot curve,
  series IUDMNZC and IUDLNZC (percent, divided by 100 here), IADB,
  https://www.bankofengland.co.uk/boeapps/database/

Calendars differ between sources (the ECB publishes on TARGET days, the Fed
curve on US bond-market days), so two series in one scenario may have
different numbers of observations; both window ends are trading days for
every series used, which the frozen data and a test confirm.

Worked example
--------------
The SNB removed the EURCHF floor on 15 January 2015. The ECB fixing went
from 1.2010 on the 14th to 1.0280 on the 15th, a log move of
``ln(1.0280/1.2010) = -0.1555``. ``scenario("chf_2015").shocks["EURCHF"]``
is that number, and applying it to a snapshot quoting 1.2010 gives 1.0280.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from ._windows import SOURCE_SHA256, WINDOWS

__all__ = [
    "DESCRIPTIONS",
    "SERIES",
    "SOURCE_SHA256",
    "HistoricalScenario",
    "SeriesSpec",
    "apply_to_quotes",
    "apply_to_snapshot",
    "reverse_stress_scale",
    "scenario",
    "scenario_names",
]

MoveKind = Literal["absolute", "log"]


@dataclass(frozen=True)
class SeriesSpec:
    """How a source series is measured and shocked."""

    name: str
    kind: MoveKind
    unit: str
    source: str


SERIES: dict[str, SeriesSpec] = {
    "USD_ZERO_2Y": SeriesSpec("USD_ZERO_2Y", "absolute", "decimal rate", "Fed GSW SVENY02"),
    "USD_ZERO_10Y": SeriesSpec("USD_ZERO_10Y", "absolute", "decimal rate", "Fed GSW SVENY10"),
    "VIX": SeriesSpec("VIX", "absolute", "index points", "CBOE VIX close"),
    "EURUSD": SeriesSpec("EURUSD", "log", "USD per EUR", "ECB EXR D.USD.EUR.SP00.A"),
    "EURCHF": SeriesSpec("EURCHF", "log", "CHF per EUR", "ECB EXR D.CHF.EUR.SP00.A"),
    "EURGBP": SeriesSpec("EURGBP", "log", "GBP per EUR", "ECB EXR D.GBP.EUR.SP00.A"),
    "EURJPY": SeriesSpec("EURJPY", "log", "JPY per EUR", "ECB EXR D.JPY.EUR.SP00.A"),
    "GBP_ZERO_10Y": SeriesSpec("GBP_ZERO_10Y", "absolute", "decimal rate", "BoE IUDMNZC"),
    "GBP_ZERO_20Y": SeriesSpec("GBP_ZERO_20Y", "absolute", "decimal rate", "BoE IUDLNZC"),
}

DESCRIPTIONS: dict[str, str] = {
    "gfc_2008": "Lehman to the G7 meeting: 12 Sep to 10 Oct 2008",
    "eur_2011": "Euro-area sovereign crisis and US downgrade: 1 Jul to 30 Sep 2011",
    "chf_2015": "SNB removes the EURCHF floor: 14 to 15 Jan 2015",
    "gbp_2016": "UK EU-membership referendum: 23 to 24 Jun 2016",
    "covid_2020": "COVID-19 sell-off: 21 Feb to 23 Mar 2020",
    "gilts_2022": "UK mini-budget and LDI gilt sell-off: 22 to 27 Sep 2022",
}


def _move(kind: MoveKind, x0: float, x1: float) -> float:
    return x1 - x0 if kind == "absolute" else math.log(x1 / x0)


@dataclass(frozen=True)
class HistoricalScenario:
    """One named episode: window, source observations and shocks."""

    name: str
    description: str
    start: str
    end: str
    observations: Mapping[str, tuple[tuple[str, float], ...]]

    @property
    def shocks(self) -> dict[str, float]:
        """End-to-end move per series (see the module docstring)."""
        return {
            s: _move(SERIES[s].kind, obs[0][1], obs[-1][1])
            for s, obs in self.observations.items()
        }

    def path_shocks(self) -> dict[str, tuple[tuple[str, float], ...]]:
        """Cumulative move from ``start`` to every observation date."""
        return {
            s: tuple((d, _move(SERIES[s].kind, obs[0][1], v)) for d, v in obs)
            for s, obs in self.observations.items()
        }

    def scaled(self, factor: float) -> dict[str, float]:
        """Shocks multiplied by ``factor`` (for reverse stress searches)."""
        if not math.isfinite(factor):
            raise ValueError("scale factor must be finite")
        return {k: factor * v for k, v in self.shocks.items()}


def scenario_names() -> tuple[str, ...]:
    return tuple(WINDOWS)


def scenario(name: str) -> HistoricalScenario:
    """The named scenario; ``KeyError`` lists the valid names."""
    try:
        obs = WINDOWS[name]
    except KeyError:
        raise KeyError(f"unknown scenario {name!r}; choose from {scenario_names()}") from None
    first = next(iter(obs.values()))
    return HistoricalScenario(name, DESCRIPTIONS[name], first[0][0], first[-1][0], obs)


def apply_to_quotes(
    quotes: Mapping[str, float],
    shocks: Mapping[str, float],
    mapping: Mapping[str, str] | None = None,
) -> dict[str, float]:
    """Apply series shocks to a quote dictionary.

    ``mapping`` sends a series name to the quote key it stands for (default:
    the same name). Quotes without a shock are unchanged; a shock whose
    quote is missing raises ``KeyError`` so a scenario never silently does
    nothing.
    """
    out = dict(quotes)
    for series, dx in shocks.items():
        key = (mapping or {}).get(series, series)
        if key not in out:
            raise KeyError(f"scenario shocks {series!r} but the quotes have no {key!r}")
        q = out[key]
        out[key] = q + dx if SERIES[series].kind == "absolute" else q * math.exp(dx)
    return out


def apply_to_snapshot(
    snapshot: Any,
    shocks: Mapping[str, float],
    mapping: Mapping[str, str] | None = None,
) -> Any:
    """Apply shocks to a market snapshot with a ``quotes`` mapping.

    Integration seam for the W10 ``MarketSnapshot`` (immutable dataclass
    with a ``quotes`` field). It is duck-typed: any frozen dataclass with a
    ``quotes`` mapping works, and the result is ``dataclasses.replace`` of
    the input, so curves and surfaces built from quotes are rebuilt by the
    snapshot's own logic, not here.
    """
    if (
        isinstance(snapshot, type)
        or not dataclasses.is_dataclass(snapshot)
        or not hasattr(snapshot, "quotes")
    ):
        raise TypeError("snapshot must be a dataclass with a 'quotes' mapping")
    new_quotes = apply_to_quotes(snapshot.quotes, shocks, mapping)
    return dataclasses.replace(snapshot, quotes=new_quotes)  # type: ignore[type-var]


def reverse_stress_scale(
    sc: HistoricalScenario,
    loss: Callable[[dict[str, float]], float],
    threshold: float,
    *,
    max_scale: float = 10.0,
    tol: float = 1e-10,
) -> float:
    """Smallest ``k`` in ``(0, max_scale]`` with ``loss(k * shocks) >= threshold``.

    The historical shock is the seed direction; the search finds how much
    of it the book can take before the loss reaches ``threshold`` (a
    reverse stress test along a plausible direction). ``loss`` returns a
    positive number for a loss. Bisection on the bracket ``[0, max_scale]``
    assumes the loss rises along the ray; ``ValueError`` if the threshold is
    not reached at ``max_scale``.
    """
    if threshold <= loss(sc.scaled(0.0)):
        return 0.0
    if loss(sc.scaled(max_scale)) < threshold:
        raise ValueError(f"loss stays below {threshold} up to {max_scale} x the {sc.name} shock")
    lo, hi = 0.0, max_scale
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if loss(sc.scaled(mid)) >= threshold:
            hi = mid
        else:
            lo = mid
    return hi
