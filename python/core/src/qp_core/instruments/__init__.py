"""Instruments, trades and portfolios behind one three-method contract.

``Instrument`` has ``pv(market)``, ``cashflows()`` and ``risk(market)``;
``Trade`` scales an instrument by a quantity; ``Portfolio`` aggregates in a
reporting currency. Adapters from the older per-package trade classes live
in :mod:`qp_core.instruments.adapters` and are imported on demand, so
``qp_core`` keeps no dependency on the product packages.

Worked example
--------------
>>> import datetime as dt
>>> from qp_core.market import DiscountCurve, MarketSnapshot
>>> m = MarketSnapshot(dt.date(2025, 6, 2), quotes={"FX/EURUSD": 1.10},
...     curves={"USD": DiscountCurve.flat(0.04), "EUR": DiscountCurve.flat(0.02)})
>>> fwd = FXForward("EURUSD", notional_base=1e6, strike=1.12, delivery=dt.date(2026, 6, 2))
>>> book = Portfolio([Trade("T1", fwd), Trade("T2", Cash("EUR", 1e5))])
>>> round(fwd.pv(m).amount, 2)
2016.21
>>> round(book.pv(m, "USD").amount, 2)
112004.16
"""

from __future__ import annotations

from .base import Cashflow, Instrument, Maturity, Money, bump_risk, risk_key
from .products import Cash, EquityOption, EquityPosition, FixedCashflows, FXForward, FXSwap
from .serde import dumps, from_json, loads, register, to_json
from .trade import Portfolio, Trade

__all__ = [
    "Cash",
    "Cashflow",
    "EquityOption",
    "EquityPosition",
    "FXForward",
    "FXSwap",
    "FixedCashflows",
    "Instrument",
    "Maturity",
    "Money",
    "Portfolio",
    "Trade",
    "bump_risk",
    "dumps",
    "from_json",
    "loads",
    "register",
    "risk_key",
    "to_json",
]
