"""Trades and portfolios on top of :class:`Instrument`.

Plain idea
----------
An instrument is the contract; a trade is our holding of it: an id, a
quantity, a book, a counterparty and a trade date. A trade is itself an
instrument (its numbers are the instrument's times the quantity), and a
portfolio is a list of trades that reports in one currency.

Formal statement
----------------
``Trade.pv(m) = q * I.pv(m)``, ``Trade.cashflows() = q * I.cashflows()``,
``Trade.risk(m) = q * I.risk(m)``. ``Portfolio.pv(m, c) = sum_i
convert(Trade_i.pv(m), c)`` using :meth:`MarketSnapshot.fx_today`; risk is
summed key by key after the same conversion. Converting risk at today's FX
rate is first order: an FX-delta of a book in a reporting currency also
moves with the conversion rate, which a multi-currency risk engine (W14d)
handles; here each trade's own ``delta:FX/...`` is kept as reported.

Worked example: :mod:`qp_core.instruments` module docstring.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace

from ..errors import InputError
from ..market.snapshot import MarketSnapshot
from .base import Cashflow, Instrument, Money

__all__ = ["Portfolio", "Trade"]


@dataclass(frozen=True)
class Trade:
    """A holding of ``quantity`` units of ``instrument``."""

    trade_id: str
    instrument: Instrument
    quantity: float = 1.0
    book: str = ""
    counterparty: str = ""
    trade_date: dt.date | None = None

    def __post_init__(self) -> None:
        if not self.trade_id:
            raise InputError("trade_id must not be empty")
        if not isinstance(self.instrument, Instrument):
            raise InputError(f"{type(self.instrument).__name__} is not an Instrument")

    def pv(self, market: MarketSnapshot) -> Money:
        return self.instrument.pv(market).scaled(self.quantity)

    def cashflows(self) -> list[Cashflow]:
        return [Cashflow(c.when, c.ccy, c.amount * self.quantity, c.kind) for c in self.instrument.cashflows()]

    def risk(self, market: MarketSnapshot) -> dict[str, float]:
        return {k: v * self.quantity for k, v in self.instrument.risk(market).items()}

    def replace(self, **changes: object) -> Trade:
        return _dc_replace(self, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Portfolio:
    """An ordered set of trades with unique ids."""

    trades: tuple[Trade, ...] = field(default_factory=tuple)
    name: str = ""

    def __init__(self, trades: Iterable[Trade] = (), name: str = "") -> None:
        t = tuple(trades)
        ids = [x.trade_id for x in t]
        if len(set(ids)) != len(ids):
            raise InputError("duplicate trade_id in portfolio")
        object.__setattr__(self, "trades", t)
        object.__setattr__(self, "name", name)

    def __iter__(self) -> Iterator[Trade]:
        return iter(self.trades)

    def __len__(self) -> int:
        return len(self.trades)

    def __getitem__(self, trade_id: str) -> Trade:
        for t in self.trades:
            if t.trade_id == trade_id:
                return t
        raise KeyError(trade_id)

    def add(self, trade: Trade) -> Portfolio:
        return Portfolio((*self.trades, trade), self.name)

    def remove(self, trade_id: str) -> Portfolio:
        self[trade_id]
        return Portfolio((t for t in self.trades if t.trade_id != trade_id), self.name)

    def filter(self, book: str) -> Portfolio:
        return Portfolio((t for t in self.trades if t.book == book), f"{self.name}/{book}")

    def pv_by_trade(self, market: MarketSnapshot, ccy: str) -> dict[str, float]:
        return {t.trade_id: t.pv(market).to(ccy, market).amount for t in self.trades}

    def pv(self, market: MarketSnapshot, ccy: str = "USD") -> Money:
        return Money(sum(self.pv_by_trade(market, ccy).values()), ccy)

    def cashflows(self) -> list[Cashflow]:
        return [c for t in self.trades for c in t.cashflows()]

    def risk(self, market: MarketSnapshot, ccy: str = "USD") -> dict[str, float]:
        out: dict[str, float] = {}
        for t in self.trades:
            fx = market.fx_today(t.pv(market).ccy, ccy)
            for k, v in t.risk(market).items():
                out[k] = out.get(k, 0.0) + v * fx
        return out
