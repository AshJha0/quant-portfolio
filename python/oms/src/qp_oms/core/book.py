"""Journaled order book: every change is logged before it takes effect (W28a).

Plain idea
----------
:class:`OrderBook` keeps the current :class:`~qp_oms.core.state.Order`
for each order and a hash-chained :class:`~qp_core.eventlog.EventLog`.
Each public method does three things in a fixed order:

1. compute the new order value with the pure state machine (an illegal
   event raises here, and nothing is logged or changed);
2. append the event to the log (write-ahead: durable before step 3 and
   before the caller sends anything to a venue);
3. swap the new value in.

Because the log holds every input and the state machine is pure,
:meth:`OrderBook.replay` rebuilds the identical book from the log alone.
That is the crash-recovery primitive W28d extends.

Worked example
--------------
>>> from qp_core.eventlog import EventLog
>>> book = OrderBook(EventLog.in_memory())
>>> _ = book.submit("A1", side="1", symbol="XYZ", order_qty=100, price=10.0)
>>> _ = book.on_report(ExecReport(ExecType.NEW, "e1", "A1"))
>>> _ = book.request_cancel("A1", "A2")
>>> book.on_report(ExecReport(ExecType.CANCELED, "e2", "A2")).ord_status.name
'CANCELED'
>>> [r.kind for r in book.log]
['oms.submit', 'oms.report', 'oms.cancel_request', 'oms.report']
>>> OrderBook.replay(book.log).snapshot() == book.snapshot()
True
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from qp_core.eventlog import EventLog, LogRecord

from qp_oms.core.state import (
    ExecReport,
    ExecType,
    IllegalTransition,
    Order,
    apply_cancel_reject,
    apply_report,
    request_cancel,
    request_replace,
)

__all__ = ["OrderBook", "UnknownOrder"]


class UnknownOrder(KeyError):
    """No order in the book has this ClOrdID (current, chained or requested)."""


def _report_payload(r: ExecReport) -> dict[str, Any]:
    return {
        "exec_type": r.exec_type.value,
        "exec_id": r.exec_id,
        "clordid": r.clordid,
        "last_qty": r.last_qty,
        "last_px": r.last_px,
        "order_qty": r.order_qty,
        "price": r.price,
        "exec_ref_id": r.exec_ref_id,
    }


def _report_from(p: Mapping[str, Any]) -> ExecReport:
    return ExecReport(
        ExecType(p["exec_type"]),
        p["exec_id"],
        p["clordid"],
        last_qty=int(p["last_qty"]),
        last_px=float(p["last_px"]),
        order_qty=p["order_qty"],
        price=p["price"],
        exec_ref_id=p["exec_ref_id"],
    )


class OrderBook:
    """Orders keyed by their first ClOrdID, with an alias for every ClOrdID in the chain."""

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self._orders: dict[str, Order] = {}
        self._alias: dict[str, str] = {}

    # -- lookup ---------------------------------------------------------
    def _root(self, clordid: str) -> str:
        try:
            return self._alias[clordid]
        except KeyError:
            raise UnknownOrder(clordid) from None

    def get(self, clordid: str) -> Order:
        return self._orders[self._root(clordid)]

    def orders(self) -> Iterable[Order]:
        return tuple(self._orders.values())

    def snapshot(self) -> dict[str, Order]:
        return dict(self._orders)

    def _fresh(self, clordid: str) -> None:
        if clordid in self._alias:
            raise IllegalTransition(f"ClOrdID {clordid!r} already used")

    def _log_then_set(
        self, root: str, new: Order, kind: str, payload: Mapping[str, Any], ts_ns: int | None
    ) -> Order:
        self.log.append(kind, payload, ts_ns=ts_ns)
        self._orders[root] = new
        return new

    # -- commands -------------------------------------------------------
    def submit(
        self,
        clordid: str,
        *,
        side: str,
        symbol: str,
        order_qty: int,
        price: float | None,
        ts_ns: int | None = None,
    ) -> Order:
        """Create an order (35=D). Log it; the caller sends only after this returns."""
        self._fresh(clordid)
        o = Order.new(clordid, side=side, symbol=symbol, order_qty=order_qty, price=price)
        payload = {"clordid": clordid, "side": side, "symbol": symbol, "order_qty": order_qty, "price": price}
        self._log_then_set(clordid, o, "oms.submit", payload, ts_ns)
        self._alias[clordid] = clordid
        return o

    def request_cancel(self, clordid: str, new_clordid: str, *, ts_ns: int | None = None) -> Order:
        root = self._root(clordid)
        self._fresh(new_clordid)
        new = request_cancel(self._orders[root], new_clordid)
        payload = {"orig_clordid": clordid, "clordid": new_clordid}
        self._log_then_set(root, new, "oms.cancel_request", payload, ts_ns)
        self._alias[new_clordid] = root
        return new

    def request_replace(
        self,
        clordid: str,
        new_clordid: str,
        *,
        order_qty: int,
        price: float | None,
        ts_ns: int | None = None,
    ) -> Order:
        root = self._root(clordid)
        self._fresh(new_clordid)
        new = request_replace(self._orders[root], new_clordid, order_qty=order_qty, price=price)
        payload = {"orig_clordid": clordid, "clordid": new_clordid, "order_qty": order_qty, "price": price}
        self._log_then_set(root, new, "oms.replace_request", payload, ts_ns)
        self._alias[new_clordid] = root
        return new

    def on_report(self, r: ExecReport, *, ts_ns: int | None = None) -> Order:
        """Apply a 35=8. A duplicate ExecID is a no-op and is not logged again."""
        root = self._root(r.clordid)
        old = self._orders[root]
        new = apply_report(old, r)
        if new is old:
            return old
        return self._log_then_set(root, new, "oms.report", _report_payload(r), ts_ns)

    def on_cancel_reject(self, clordid: str, *, ts_ns: int | None = None) -> Order:
        """Apply a 35=9 for our outstanding request ``clordid``."""
        root = self._root(clordid)
        new = apply_cancel_reject(self._orders[root], clordid)
        return self._log_then_set(root, new, "oms.cancel_reject", {"clordid": clordid}, ts_ns)

    # -- recovery -------------------------------------------------------
    @classmethod
    def replay(cls, records: Iterable[LogRecord], log: EventLog | None = None) -> OrderBook:
        """Rebuild a book from logged records (other kinds are skipped).

        The rebuilt book writes new events to ``log`` (a fresh in-memory
        log if omitted); replay itself appends nothing.
        """
        book = cls(EventLog.in_memory())
        for rec in records:
            p = rec.payload
            if rec.kind == "oms.submit":
                book.submit(
                    p["clordid"],
                    side=p["side"],
                    symbol=p["symbol"],
                    order_qty=p["order_qty"],
                    price=p["price"],
                )
            elif rec.kind == "oms.cancel_request":
                book.request_cancel(p["orig_clordid"], p["clordid"])
            elif rec.kind == "oms.replace_request":
                book.request_replace(
                    p["orig_clordid"], p["clordid"], order_qty=p["order_qty"], price=p["price"]
                )
            elif rec.kind == "oms.report":
                book.on_report(_report_from(p))
            elif rec.kind == "oms.cancel_reject":
                book.on_cancel_reject(p["clordid"])
        book.log = log if log is not None else EventLog.in_memory()
        return book
