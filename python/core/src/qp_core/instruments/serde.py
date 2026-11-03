"""JSON for instruments, trades and portfolios.

Plain idea
----------
A trade is stored as plain JSON with a ``type`` tag naming its instrument
class; reading it back gives an equal object. Python's ``json`` writes floats
with ``repr``, which round-trips every finite double exactly (Steele and
White 1990; Python ``float.__repr__``), so a stored book prices to the same
bits after a round trip.

Formal statement
----------------
``loads(dumps(x)) == x`` for every registered instrument, :class:`Trade`
and :class:`Portfolio`. Registration maps a tag to a class with
``to_dict()`` and ``from_dict(d)``. New products call :func:`register`.

Worked example
--------------
>>> from qp_core.instruments import Cash, Trade
>>> t = Trade("T1", Cash("EUR", 0.1), quantity=3.0, book="FX")
>>> loads(dumps(t)) == t
True
>>> dumps(Cash("EUR", 0.1))
'{"amount":0.1,"ccy":"EUR","type":"Cash","when":0.0}'
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from typing import Any, TypeVar

from ..errors import InputError
from .base import Instrument
from .products import Cash, EquityOption, EquityPosition, FixedCashflows, FXForward, FXSwap
from .trade import Portfolio, Trade

__all__ = ["dumps", "from_json", "loads", "register", "to_json"]

SCHEMA_VERSION = 1
_REGISTRY: dict[str, type[Any]] = {}
_T = TypeVar("_T", bound=type[Any])


def register(tag: str | None = None) -> Callable[[_T], _T]:
    """Class decorator: make an instrument class serialisable under ``tag``."""

    def deco(cls: _T) -> _T:
        name = tag or cls.__name__
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise InputError(f"instrument tag {name!r} already registered")
        _REGISTRY[name] = cls
        return cls

    return deco


for _cls in (Cash, FixedCashflows, FXForward, FXSwap, EquityPosition, EquityOption):
    register()(_cls)


def _tag(obj: Any) -> str:
    for k, v in _REGISTRY.items():
        if type(obj) is v:
            return k
    raise InputError(f"{type(obj).__name__} is not registered; use qp_core.instruments.register")


def to_json(obj: Instrument | Trade | Portfolio) -> dict[str, Any]:
    """JSON-ready dict for an instrument, a trade or a portfolio."""
    if isinstance(obj, Portfolio):
        return {"type": "Portfolio", "schema": SCHEMA_VERSION, "name": obj.name,
                "trades": [to_json(t) for t in obj.trades]}
    if isinstance(obj, Trade):
        return {"type": "Trade", "trade_id": obj.trade_id, "quantity": obj.quantity, "book": obj.book,
                "counterparty": obj.counterparty,
                "trade_date": obj.trade_date.isoformat() if obj.trade_date else None,
                "instrument": to_json(obj.instrument)}
    return {"type": _tag(obj), **obj.to_dict()}  # type: ignore[attr-defined]


def from_json(d: dict[str, Any]) -> Any:
    """Inverse of :func:`to_json`."""
    kind = d.get("type")
    if kind == "Portfolio":
        if d.get("schema") != SCHEMA_VERSION:
            raise InputError(f"unsupported portfolio schema {d.get('schema')!r}")
        return Portfolio([from_json(t) for t in d["trades"]], d.get("name", ""))
    if kind == "Trade":
        td = d.get("trade_date")
        return Trade(d["trade_id"], from_json(d["instrument"]), d["quantity"], d.get("book", ""),
                     d.get("counterparty", ""), dt.date.fromisoformat(td) if td else None)
    if kind not in _REGISTRY:
        raise InputError(f"unknown instrument type {kind!r}")
    body = {k: v for k, v in d.items() if k != "type"}
    return _REGISTRY[kind].from_dict(body)


def dumps(obj: Instrument | Trade | Portfolio) -> str:
    return json.dumps(to_json(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def loads(s: str) -> Any:
    return from_json(json.loads(s))
