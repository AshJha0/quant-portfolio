"""Order interface shared by the backtester (W24) and the order path (W28).

Plain idea
----------
A strategy does not move positions directly. It sends an order; something
answers with execution reports; the position is what the reports add up to.
The backtester (``qp_bt``) and the real order path (W28) must speak the same
language, so that a strategy tested in one runs in the other without a
rewrite. This module is that language and nothing else: no state, no I/O, no
third-party imports.

Formal statement
----------------
The types mirror the FIX 4.4 order messages the W28 session will carry:

* :class:`Order` is a NewOrderSingle (35=D): ClOrdID (11), Symbol (55), Side
  (54), OrderQty (38), OrdType (40), Price (44), TimeInForce (59).
* :class:`ExecutionReport` is an ExecutionReport (35=8): ExecID (17),
  OrdStatus (39), ExecType (150), LastQty (32), LastPx (31), CumQty (14),
  LeavesQty (151), AvgPx (6).
* The invariant ``CumQty + LeavesQty = OrderQty`` holds on every report of a
  live order; once the order is done (filled, cancelled, rejected, expired)
  ``LeavesQty = 0``. It is checked when a report is built.
* :class:`OrderGateway` is what a strategy calls; :class:`ExecutionListener`
  is what it is called back on. ``qp_bt`` implements both sides in
  simulation; W28 implements the gateway over FIX.

Worked example
--------------
Buy 100 shares, filled 60 then 40::

    >>> o = Order("c1", "AAPL", Side.BUY, 100.0)
    >>> r1 = ExecutionReport.fill(o, "e1", last_qty=60.0, last_px=10.0, cum_qty=60.0)
    >>> r1.ord_status, r1.leaves_qty
    (<OrdStatus.PARTIALLY_FILLED: '1'>, 40.0)
    >>> r2 = ExecutionReport.fill(o, "e2", last_qty=40.0, last_px=10.5, cum_qty=100.0,
    ...                           avg_px=10.2)
    >>> r2.ord_status, r2.leaves_qty
    (<OrdStatus.FILLED: '2'>, 0.0)

Review notes for W28 (the 0.5 dd interface review, Round 10)
------------------------------------------------------------
1. Quantities are unsigned with a separate side, as in FIX. The backtester's
   internal signed share deltas convert with :func:`signed_qty`.
2. Target-weight and target-position intents are a backtest convenience: they
   are resolved into plain orders at the fill point and never reach this
   interface. W28 needs no "target" order type.
3. ``decision_time`` is not a FIX field. It travels with the order inside the
   process so that signal-to-fill lag can be measured; a FIX gateway drops it
   (or carries it in a user-defined tag, W28's choice).
4. Amend (35=G) and cancel (35=F) take the original ClOrdID and a new one, so
   ClOrdID chains are explicit. The simulator only needs ``cancel``.
5. Nothing here is thread-safe or persistent; W28a's event log owns that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

__all__ = [
    "ExecType",
    "ExecutionListener",
    "ExecutionReport",
    "OrdStatus",
    "OrdType",
    "Order",
    "OrderGateway",
    "Side",
    "TimeInForce",
    "signed_qty",
]


class Side(StrEnum):
    """FIX tag 54."""

    BUY = "1"
    SELL = "2"
    SELL_SHORT = "5"


class OrdType(StrEnum):
    """FIX tag 40 (the subset the backtester models)."""

    MARKET = "1"
    LIMIT = "2"
    MARKET_ON_CLOSE = "5"


class TimeInForce(StrEnum):
    """FIX tag 59 (the subset the backtester models)."""

    DAY = "0"
    GTC = "1"
    IOC = "3"
    AT_THE_CLOSE = "7"


class OrdStatus(StrEnum):
    """FIX tag 39."""

    NEW = "0"
    PARTIALLY_FILLED = "1"
    FILLED = "2"
    CANCELED = "4"
    REJECTED = "8"
    EXPIRED = "C"


class ExecType(StrEnum):
    """FIX tag 150."""

    NEW = "0"
    CANCELED = "4"
    REJECTED = "8"
    EXPIRED = "C"
    TRADE = "F"


_DONE = {OrdStatus.FILLED, OrdStatus.CANCELED, OrdStatus.REJECTED, OrdStatus.EXPIRED}


@dataclass(frozen=True)
class Order:
    """A new order (FIX 35=D).

    ``quantity`` is unsigned and positive; ``side`` carries the direction.
    ``limit_price`` is required for ``LIMIT`` and must be absent otherwise.
    """

    cl_ord_id: str
    symbol: str
    side: Side
    quantity: float
    ord_type: OrdType = OrdType.MARKET
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    decision_time: object | None = None

    def __post_init__(self) -> None:
        if not self.cl_ord_id:
            raise ValueError("cl_ord_id must be non-empty")
        if not math.isfinite(self.quantity) or self.quantity <= 0:
            raise ValueError(f"quantity must be finite and > 0, got {self.quantity!r}")
        if self.ord_type is OrdType.LIMIT:
            if self.limit_price is None or not math.isfinite(self.limit_price):
                raise ValueError("a LIMIT order needs a finite limit_price")
        elif self.limit_price is not None:
            raise ValueError(f"{self.ord_type.name} orders carry no limit_price")


def signed_qty(order: Order) -> float:
    """Signed share change of ``order``: positive buys, negative sells."""
    return order.quantity if order.side is Side.BUY else -order.quantity


@dataclass(frozen=True)
class ExecutionReport:
    """An execution report (FIX 35=8) with the quantity invariants checked."""

    cl_ord_id: str
    exec_id: str
    exec_type: ExecType
    ord_status: OrdStatus
    order_qty: float
    cum_qty: float
    leaves_qty: float
    last_qty: float = 0.0
    last_px: float | None = None
    avg_px: float | None = None
    transact_time: object | None = None

    def __post_init__(self) -> None:
        tol = 1e-9 * max(1.0, abs(self.order_qty))
        if self.cum_qty > self.order_qty + tol:
            raise ValueError("cum_qty exceeds order_qty (overfill)")
        if self.cum_qty < -tol or self.leaves_qty < -tol or self.last_qty < -tol:
            raise ValueError("cum_qty, leaves_qty and last_qty must be >= 0")
        if self.ord_status in _DONE:
            if abs(self.leaves_qty) > tol:
                raise ValueError(f"leaves_qty must be 0 once {self.ord_status.name}")
        elif abs(self.cum_qty + self.leaves_qty - self.order_qty) > tol:
            raise ValueError("CumQty + LeavesQty must equal OrderQty on a live order")
        if self.exec_type is ExecType.TRADE:
            if self.last_qty <= 0 or self.last_px is None or not math.isfinite(self.last_px):
                raise ValueError("a TRADE report needs last_qty > 0 and a finite last_px")

    @classmethod
    def fill(cls, order: Order, exec_id: str, *, last_qty: float, last_px: float,
             cum_qty: float, avg_px: float | None = None,
             transact_time: object | None = None) -> ExecutionReport:
        """A TRADE report; status is FILLED when nothing is left, else PARTIALLY_FILLED."""
        leaves = order.quantity - cum_qty
        if abs(leaves) <= 1e-9 * max(1.0, order.quantity):
            leaves = 0.0
        status = OrdStatus.FILLED if leaves == 0.0 else OrdStatus.PARTIALLY_FILLED
        return cls(order.cl_ord_id, exec_id, ExecType.TRADE, status, order.quantity,
                   cum_qty, leaves, last_qty, last_px,
                   last_px if avg_px is None else avg_px, transact_time)


@runtime_checkable
class ExecutionListener(Protocol):
    """Called back with every execution report for orders it sent."""

    def on_execution_report(self, report: ExecutionReport) -> None: ...


@runtime_checkable
class OrderGateway(Protocol):
    """What a strategy sends orders through: the simulator or the W28 session."""

    def submit(self, order: Order) -> None: ...

    def cancel(self, orig_cl_ord_id: str, cl_ord_id: str) -> None: ...
