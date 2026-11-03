"""Order state machine and journaled order book (W28a).

This is the published interface other lanes code against (W24 backtest
order interface, W28c OMS/EMS, W28d risk and recovery, W29 surveillance):

* :mod:`qp_oms.core.state` - ``Order`` (immutable), ``ExecReport``,
  ``ExecType``, ``OrdStatus``, ``apply_report``, ``request_cancel``,
  ``request_replace``, ``apply_cancel_reject``, ``IllegalTransition``.
* :mod:`qp_oms.core.book` - ``OrderBook`` (log-then-apply, ``replay``).
* :mod:`qp_core.eventlog` - ``EventLog`` (hash chain, fsync policy).

Event kinds written by ``OrderBook``: ``oms.submit``,
``oms.cancel_request``, ``oms.replace_request``, ``oms.report``,
``oms.cancel_reject``. Other lanes should use their own prefix
(``fix.``, ``risk.``, ``kill.``) in the same log.
"""

from __future__ import annotations

from qp_oms.core.book import OrderBook, UnknownOrder
from qp_oms.core.state import (
    ExecReport,
    ExecType,
    IllegalTransition,
    Order,
    OrdStatus,
    Phase,
    Request,
    apply_cancel_reject,
    apply_report,
    request_cancel,
    request_replace,
)

__all__ = [
    "ExecReport",
    "ExecType",
    "IllegalTransition",
    "OrdStatus",
    "Order",
    "OrderBook",
    "Phase",
    "Request",
    "UnknownOrder",
    "apply_cancel_reject",
    "apply_report",
    "request_cancel",
    "request_replace",
]
