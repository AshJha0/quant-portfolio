"""The trade-layer contract: three methods and two value types.

Plain idea
----------
Every instrument, whatever it is, answers three questions: what is it worth
in this market (``pv``), what does it pay and when (``cashflows``), and how
does its value move when the market moves (``risk``). Downstream code (the
OMS, risk grids, the backtester) only talks to these three methods, so a new
product needs no change anywhere else. This is the QuantLib split between an
instrument and its pricing engine reduced to the minimum, and the "product"
and "trade" layers of the FINOS Common Domain Model.

Formal statement
----------------
``pv(m) -> Money``: present value as of ``m.as_of`` in the instrument's own
currency, the currency of the returned :class:`Money`.

``cashflows() -> list[Cashflow]``: the contractual flows known today, signed
from the holder's side (receive > 0). Contingent payoffs (options) list only
what is fixed; an empty list is allowed.

``risk(m) -> dict[str, float]``: sensitivities of ``pv`` in the PV currency.
Keys follow :func:`risk_key`: ``delta:FX/EURUSD`` (per unit of the quote),
``pv01:USD`` (per +1bp parallel zero shift), ``vega:EQ/SPX`` (per +1 vol
point = 0.01). The default implementation, :func:`bump_risk`, is central
finite differences on the snapshot; instruments with closed forms may
override it, and tests hold them to the bump numbers.

Worked example: :mod:`qp_core.instruments` module docstring.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..errors import InputError
from ..market.snapshot import MarketSnapshot, Shock

__all__ = [
    "Cashflow",
    "Instrument",
    "Maturity",
    "Money",
    "bump_risk",
    "maturity_from_json",
    "maturity_to_json",
    "risk_key",
]

#: A payment time: a calendar date, or years from the snapshot date.
Maturity = dt.date | float

#: Bump sizes used by :func:`bump_risk`.
QUOTE_BUMP_REL = 1e-4
CURVE_BUMP = 1e-4
VOL_BUMP = 1e-4


@dataclass(frozen=True)
class Money:
    """An amount in a currency."""

    amount: float
    ccy: str

    def __add__(self, other: Money) -> Money:
        if other.ccy != self.ccy:
            raise InputError(f"cannot add {other.ccy} to {self.ccy}; convert first")
        return Money(self.amount + other.amount, self.ccy)

    def scaled(self, k: float) -> Money:
        return Money(self.amount * k, self.ccy)

    def to(self, ccy: str, market: MarketSnapshot) -> Money:
        """Convert at today's value of one unit (:meth:`MarketSnapshot.fx_today`)."""
        return Money(self.amount * market.fx_today(self.ccy, ccy), ccy)


@dataclass(frozen=True)
class Cashflow:
    """One contractual payment, signed from the holder's side."""

    when: Maturity
    ccy: str
    amount: float
    kind: str = ""


@runtime_checkable
class Instrument(Protocol):
    """Anything with ``pv``, ``cashflows`` and ``risk`` (see module docstring)."""

    def pv(self, market: MarketSnapshot) -> Money: ...

    def cashflows(self) -> list[Cashflow]: ...

    def risk(self, market: MarketSnapshot) -> dict[str, float]: ...


def risk_key(measure: str, target: str) -> str:
    return f"{measure}:{target}"


def bump_risk(
    pv: Any,
    market: MarketSnapshot,
    quotes: Iterable[str] = (),
    curves: Iterable[str] = (),
    surfaces: Iterable[str] = (),
) -> dict[str, float]:
    """Central-difference sensitivities of ``pv(market) -> Money``.

    ``delta`` per unit of quote (relative bump 1e-4), ``pv01`` per +1bp
    parallel zero shift (bump +-1bp), ``vega`` per vol point (bump +-1e-4,
    scaled to 0.01). Error is O(h^2).
    """
    out: dict[str, float] = {}

    def central(up: Shock, dn: Shock) -> float:
        return float(pv(market.shock(up)).amount - pv(market.shock(dn)).amount)

    for q in quotes:
        h = QUOTE_BUMP_REL * abs(market.quote(q)) or QUOTE_BUMP_REL
        out[risk_key("delta", q)] = central(Shock("quote", q, h), Shock("quote", q, -h)) / (2 * h)
    for c in curves:
        out[risk_key("pv01", c)] = central(Shock("curve", c, CURVE_BUMP), Shock("curve", c, -CURVE_BUMP)) / 2
    for s in surfaces:
        out[risk_key("vega", s)] = central(Shock("vol", s, VOL_BUMP), Shock("vol", s, -VOL_BUMP)) / (2 * VOL_BUMP) * 0.01
    return out


def maturity_to_json(m: Maturity) -> Any:
    return {"date": m.isoformat()} if isinstance(m, dt.date) else float(m)


def maturity_from_json(x: Any) -> Maturity:
    if isinstance(x, Mapping):
        return dt.date.fromisoformat(x["date"])
    return float(x)
