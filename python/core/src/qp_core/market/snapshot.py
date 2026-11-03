"""Immutable market snapshot: quotes, curves, surfaces and fixings as of a date.

Plain idea
----------
Everything a pricer may read about the market on one day sits in one frozen
object. Nothing changes it in place: a scenario or a risk bump builds a new
snapshot with :meth:`MarketSnapshot.replace` or :meth:`MarketSnapshot.shock`,
the pattern ``fx_rates.MarketState.replace`` already uses. Because the
snapshot is a value, the same snapshot and the same trades always give the
same PV, which is what makes restating a past day possible (``qp_core.store``).

Formal statement
----------------
A snapshot is the tuple ``(a, Q, C, V, F)``: as-of date ``a``; quotes
``Q: key -> float``; discount curves ``C: currency -> P(t)`` with ``t`` in
years ACT/365F from ``a``; volatility surfaces ``V: key -> sigma(t, K)``;
and fixings ``F: index -> (date -> float)``. Keys follow one scheme:

======================  ===============================================
``FX/EURUSD``           FX spot, quote per base, for the spot date
``EQ/SPX``              equity or index spot
``FXSPOTTIME/EURUSD``   optional override of the spot-date year fraction
``DIV/SPX``             continuous dividend yield (decimal)
======================  ===============================================

FX spot is a price for delivery on the spot date ``t_s`` (T+2, T+1 for
USDCAD and a few others; :func:`qp_core.holidays.fx_spot_date`), so the
value today of one unit of base currency is ``S P_d(t_s) / P_f(t_s)``
(Clark 2011, ch. 1); :meth:`MarketSnapshot.fx_today` returns it. Curves are
the collateral-consistent discount curves of each currency (for EURUSD the
EUR curve already carries the cross-currency basis), so a cashflow in
currency ``c`` at ``t`` is worth ``x P_c(t)``.

Named shocks (:class:`Shock`) are data, not code, so a scenario can be
stored next to the snapshot and replayed.

Worked example
--------------
>>> import datetime as dt
>>> s = MarketSnapshot(dt.date(2025, 6, 2), quotes={"FX/EURUSD": 1.10},
...     curves={"USD": DiscountCurve.flat(0.04), "EUR": DiscountCurve.flat(0.02)})
>>> s.spot_time("EURUSD")          # Mon 2 Jun -> Wed 4 Jun
0.005479452054794521
>>> round(s.fx_today("EUR", "USD"), 10)
1.0998794587
>>> up = s.shock(Shock("quote", "FX/EURUSD", 0.01, relative=True))
>>> round(up.quote("FX/EURUSD"), 12), s.quote("FX/EURUSD")
(1.111, 1.1)
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ..errors import InputError
from ..holidays import fx_spot_date
from .curves import DiscountCurve, VolSurface

__all__ = ["MarketSnapshot", "Shock", "year_fraction"]

SCHEMA_VERSION = 1


def year_fraction(start: dt.date, end: dt.date) -> float:
    """ACT/365F year fraction, the time axis of every snapshot curve."""
    return (end - start).days / 365.0


def _freeze(m: Mapping[Any, Any] | None) -> Mapping[Any, Any]:
    return MappingProxyType(dict(m or {}))


@dataclass(frozen=True)
class Shock:
    """A named, serialisable market move.

    kind:
        ``"quote"`` adds ``size`` to quote ``target`` (or multiplies by
        ``1 + size`` when ``relative``); ``"curve"`` shifts every zero rate
        of curve ``target`` by ``size`` (decimal; ``pillar`` limits it to one
        pillar); ``"vol"`` adds ``size`` to every vol of surface ``target``.
        ``target="*"`` applies to every key of that kind.
    """

    kind: str
    target: str
    size: float
    relative: bool = False
    pillar: int | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("quote", "curve", "vol"):
            raise InputError(f"unknown shock kind {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "size": self.size,
                "relative": self.relative, "pillar": self.pillar, "name": self.name}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Shock:
        return cls(**dict(d))


@dataclass(frozen=True, eq=False)
class MarketSnapshot:
    """Frozen market state as of ``as_of``; see the module docstring."""

    as_of: dt.date
    quotes: Mapping[str, float] = field(default_factory=dict)
    curves: Mapping[str, DiscountCurve] = field(default_factory=dict)
    surfaces: Mapping[str, VolSurface] = field(default_factory=dict)
    fixings: Mapping[str, Mapping[dt.date, float]] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.as_of, dt.datetime) or not isinstance(self.as_of, dt.date):
            raise InputError("as_of must be a datetime.date")
        object.__setattr__(self, "quotes", _freeze({k: float(v) for k, v in self.quotes.items()}))
        object.__setattr__(self, "curves", _freeze(self.curves))
        object.__setattr__(self, "surfaces", _freeze(self.surfaces))
        object.__setattr__(self, "fixings", _freeze({k: _freeze(v) for k, v in self.fixings.items()}))

    # -- lookups ----------------------------------------------------------
    def quote(self, key: str) -> float:
        try:
            return self.quotes[key]
        except KeyError:
            raise InputError(f"snapshot {self.as_of} has no quote {key!r}") from None

    def curve(self, ccy: str) -> DiscountCurve:
        try:
            return self.curves[ccy]
        except KeyError:
            raise InputError(f"snapshot {self.as_of} has no curve {ccy!r}") from None

    def surface(self, key: str) -> VolSurface:
        try:
            return self.surfaces[key]
        except KeyError:
            raise InputError(f"snapshot {self.as_of} has no surface {key!r}") from None

    def fixing(self, index: str, day: dt.date) -> float:
        try:
            return self.fixings[index][day]
        except KeyError:
            raise InputError(f"snapshot {self.as_of} has no fixing {index} {day}") from None

    def time_to(self, when: dt.date | float) -> float:
        """Years from ``as_of`` (ACT/365F); a float is taken as already in years."""
        if isinstance(when, dt.date):
            return year_fraction(self.as_of, when)
        return float(when)

    def df(self, ccy: str, when: dt.date | float) -> float:
        return self.curve(ccy).df(self.time_to(when))

    def spot_time(self, pair: str) -> float:
        """Year fraction to the FX spot date of ``pair`` (e.g. ``"EURUSD"``)."""
        override = self.quotes.get(f"FXSPOTTIME/{pair}")
        if override is not None:
            return override
        return year_fraction(self.as_of, fx_spot_date(self.as_of, (pair[:3], pair[3:])))

    def fx_today(self, base: str, quote: str) -> float:
        """Value today, in ``quote``, of one unit of ``base`` paid today.

        Uses the direct pair, its inverse, or a cross through USD.
        """
        if base == quote:
            return 1.0
        direct = f"FX/{base}{quote}"
        if direct in self.quotes:
            ts = self.spot_time(base + quote)
            s = self.quotes[direct]
            if not ts:
                return s
            return float(s * self.curve(quote).df(ts) / self.curve(base).df(ts))
        if f"FX/{quote}{base}" in self.quotes:
            return 1.0 / self.fx_today(quote, base)
        if "USD" not in (base, quote):
            return self.fx_today(base, "USD") * self.fx_today("USD", quote)
        raise InputError(f"snapshot {self.as_of} cannot convert {base} to {quote}")

    # -- new snapshots ----------------------------------------------------
    def replace(self, **changes: Any) -> MarketSnapshot:
        """Copy with fields replaced (``quotes=``, ``curves=``, ...)."""
        d: dict[str, Any] = {"as_of": self.as_of, "quotes": self.quotes, "curves": self.curves,
                             "surfaces": self.surfaces, "fixings": self.fixings, "label": self.label}
        d.update(changes)
        return MarketSnapshot(**d)

    def with_quote(self, key: str, value: float) -> MarketSnapshot:
        return self.replace(quotes={**self.quotes, key: value})

    def with_curve(self, ccy: str, curve: DiscountCurve) -> MarketSnapshot:
        return self.replace(curves={**self.curves, ccy: curve})

    def shock(self, *shocks: Shock) -> MarketSnapshot:
        """Apply named shocks in order and return the shocked snapshot."""
        q = dict(self.quotes)
        c = dict(self.curves)
        v = dict(self.surfaces)
        for s in shocks:
            if s.kind == "quote":
                keys = list(q) if s.target == "*" else [s.target]
                for k in keys:
                    if k not in q:
                        raise InputError(f"shock on missing quote {k!r}")
                    q[k] = q[k] * (1.0 + s.size) if s.relative else q[k] + s.size
            elif s.kind == "curve":
                keys = list(c) if s.target == "*" else [s.target]
                for k in keys:
                    if k not in c:
                        raise InputError(f"shock on missing curve {k!r}")
                    c[k] = c[k].shifted(s.size) if s.pillar is None else c[k].bumped_pillar(s.pillar, s.size)
            else:
                keys = list(v) if s.target == "*" else [s.target]
                for k in keys:
                    if k not in v:
                        raise InputError(f"shock on missing surface {k!r}")
                    v[k] = v[k].shifted(s.size)
        label = self.label + "".join(f"|{s.name or s.kind + ':' + s.target}" for s in shocks)
        return self.replace(quotes=q, curves=c, surfaces=v, label=label)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready dict; floats round-trip exactly through ``json``."""
        return {
            "schema": SCHEMA_VERSION,
            "as_of": self.as_of.isoformat(),
            "label": self.label,
            "quotes": dict(sorted(self.quotes.items())),
            "curves": {k: self.curves[k].to_dict() for k in sorted(self.curves)},
            "surfaces": {k: self.surfaces[k].to_dict() for k in sorted(self.surfaces)},
            "fixings": {k: {d.isoformat(): x for d, x in sorted(self.fixings[k].items())}
                        for k in sorted(self.fixings)},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MarketSnapshot:
        if d.get("schema") != SCHEMA_VERSION:
            raise InputError(f"unsupported snapshot schema {d.get('schema')!r}")
        return cls(
            as_of=dt.date.fromisoformat(d["as_of"]),
            quotes=d["quotes"],
            curves={k: DiscountCurve.from_dict(x) for k, x in d["curves"].items()},
            surfaces={k: VolSurface.from_dict(x) for k, x in d["surfaces"].items()},
            fixings={k: {dt.date.fromisoformat(s): float(x) for s, x in f.items()}
                     for k, f in d["fixings"].items()},
            label=d.get("label", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> MarketSnapshot:
        return cls.from_dict(json.loads(s))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MarketSnapshot) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(self.to_json())
