"""FIX 4.4 order state machine (W28a).

Plain idea
----------
A venue tells us what happened to an order through execution reports.
Each report carries an ``ExecType`` (what just happened: a fill, a cancel,
a replace) and an ``OrdStatus`` (where the order now stands). This module
holds one order as an immutable value and a pure function
:func:`apply_report` that moves it from one state to the next. A report
that is not allowed from the current state raises
:class:`IllegalTransition` and the old value is untouched, so a bad report
can never half-update an order.

Formal statement
----------------
State: ``order_qty``, ``cum_qty``, ``avg_px``, a *phase* (``PENDING_NEW``,
``LIVE`` or a terminal phase ``REJECTED``, ``CANCELED``, ``EXPIRED``,
``DONE_FOR_DAY``), an optional acknowledged *pending* request (cancel or
replace) and an optional outstanding (sent, not yet acknowledged) request.
Fills are kept by ``ExecID`` so a trade cancel (``ExecType=H``) can reverse
one exactly.

Invariants, checked after every transition (:meth:`Order.check`):

* ``0 <= cum_qty <= order_qty``;
* while the order is working (``PENDING_NEW`` or ``LIVE``)
  ``cum_qty + leaves_qty = order_qty``; once it is terminal or filled,
  ``leaves_qty = 0`` (FIX 4.4 Vol. 4, "Order State Changes": LeavesQty is
  zero for a closed order, so the identity holds only while working);
* ``avg_px = sum(q_i * p_i) / cum_qty`` over the live fills (0 if none);
* terminal phases are absorbing, except that a trade correction
  (``ExecType=H``) may still reduce ``cum_qty`` on a closed order.

``OrdStatus`` is derived, not stored, using the FIX 4.4 precedence table
(Vol. 4, "OrdStatus" precedence): Pending Cancel > Pending Replace >
Done for Day > Filled > Canceled = Expired > Partially Filled > New =
Rejected = Pending New.

ClOrdID chains: a cancel or replace request carries a new ``ClOrdID`` and
``OrigClOrdID`` = the current one. On ``Replaced`` / ``Canceled`` the new
ClOrdID becomes current and the old one is appended to ``chain``.

Worked example (Appendix D, "Filled order", in miniature)
--------------------------------------------------------
>>> o = Order.new("A1", side="1", symbol="XYZ", order_qty=1000, price=10.0)
>>> o = apply_report(o, ExecReport(ExecType.NEW, "e1", "A1"))
>>> o = apply_report(o, ExecReport(ExecType.TRADE, "e2", "A1", last_qty=400, last_px=10.0))
>>> o.ord_status, o.cum_qty, o.leaves_qty
(<OrdStatus.PARTIALLY_FILLED: '1'>, 400, 600)
>>> o = apply_report(o, ExecReport(ExecType.TRADE, "e3", "A1", last_qty=600, last_px=10.5))
>>> o.ord_status, o.leaves_qty, o.avg_px
(<OrdStatus.FILLED: '2'>, 0, 10.3)

Scope: single-leg orders; quantities are integers (shares or contracts);
one outstanding cancel/replace request at a time (a second one while the
first is unanswered raises; FIX allows chaining, which W28c may add).

Reference: FIX 4.4 Specification Vol. 4, Appendix D "Order State Change
Matrices" (FIX Trading Community, 2003).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

__all__ = [
    "ExecReport",
    "ExecType",
    "IllegalTransition",
    "OrdStatus",
    "Order",
    "Phase",
    "Request",
    "apply_cancel_reject",
    "apply_report",
    "request_cancel",
    "request_replace",
]


class IllegalTransition(ValueError):
    """The report or request is not allowed from the order's current state."""


class ExecType(enum.StrEnum):
    """FIX tag 150."""

    NEW = "0"
    DONE_FOR_DAY = "3"
    CANCELED = "4"
    REPLACED = "5"
    PENDING_CANCEL = "6"
    REJECTED = "8"
    PENDING_NEW = "A"
    EXPIRED = "C"
    RESTATED = "D"
    PENDING_REPLACE = "E"
    TRADE = "F"
    TRADE_CANCEL = "H"


class OrdStatus(enum.StrEnum):
    """FIX tag 39 (the values this state machine can produce)."""

    NEW = "0"
    PARTIALLY_FILLED = "1"
    FILLED = "2"
    DONE_FOR_DAY = "3"
    CANCELED = "4"
    PENDING_CANCEL = "6"
    REJECTED = "8"
    PENDING_NEW = "A"
    EXPIRED = "C"
    PENDING_REPLACE = "E"


class Phase(enum.Enum):
    PENDING_NEW = "pending_new"
    LIVE = "live"
    REJECTED = "rejected"
    CANCELED = "canceled"
    EXPIRED = "expired"
    DONE_FOR_DAY = "done_for_day"


_TERMINAL = frozenset({Phase.REJECTED, Phase.CANCELED, Phase.EXPIRED, Phase.DONE_FOR_DAY})


@dataclass(frozen=True)
class Request:
    """A cancel (35=F) or cancel/replace (35=G) request we sent."""

    kind: str  # "cancel" | "replace"
    clordid: str
    order_qty: int | None = None
    price: float | None = None


@dataclass(frozen=True)
class ExecReport:
    """The fields of a 35=8 the state machine needs."""

    exec_type: ExecType
    exec_id: str
    clordid: str
    last_qty: int = 0
    last_px: float = 0.0
    order_qty: int | None = None  # REPLACED / RESTATED: the new quantity
    price: float | None = None  # REPLACED / RESTATED: the new price
    exec_ref_id: str | None = None  # TRADE_CANCEL: the ExecID being busted


@dataclass(frozen=True)
class Order:
    clordid: str
    side: str
    symbol: str
    order_qty: int
    price: float | None
    phase: Phase = Phase.PENDING_NEW
    cum_qty: int = 0
    fills: Mapping[str, tuple[int, float]] = field(default_factory=lambda: MappingProxyType({}))
    pending: str | None = None  # acknowledged: "cancel" | "replace"
    outstanding: Request | None = None
    chain: tuple[str, ...] = ()
    exec_ids: frozenset[str] = frozenset()

    @classmethod
    def new(cls, clordid: str, *, side: str, symbol: str, order_qty: int, price: float | None) -> Order:
        if not clordid:
            raise ValueError("clordid must be non-empty")
        if order_qty <= 0:
            raise ValueError("order_qty must be positive")
        if side not in ("1", "2", "5"):
            raise ValueError("side must be FIX 54 value 1 (buy), 2 (sell) or 5 (sell short)")
        o = cls(clordid, side, symbol, int(order_qty), price)
        o.check()
        return o

    # -- derived fields -------------------------------------------------
    @property
    def is_filled(self) -> bool:
        return self.cum_qty == self.order_qty

    @property
    def is_working(self) -> bool:
        return self.phase in (Phase.PENDING_NEW, Phase.LIVE) and not self.is_filled

    @property
    def is_closed(self) -> bool:
        return not self.is_working

    @property
    def leaves_qty(self) -> int:
        return self.order_qty - self.cum_qty if self.is_working else 0

    @property
    def avg_px(self) -> float:
        if self.cum_qty == 0:
            return 0.0
        return sum(q * p for q, p in self.fills.values()) / self.cum_qty

    @property
    def ord_status(self) -> OrdStatus:
        if self.pending == "cancel":
            return OrdStatus.PENDING_CANCEL
        if self.pending == "replace":
            return OrdStatus.PENDING_REPLACE
        if self.phase is Phase.DONE_FOR_DAY:
            return OrdStatus.DONE_FOR_DAY
        if self.is_filled:
            return OrdStatus.FILLED
        if self.phase is Phase.CANCELED:
            return OrdStatus.CANCELED
        if self.phase is Phase.EXPIRED:
            return OrdStatus.EXPIRED
        if self.phase is Phase.REJECTED:
            return OrdStatus.REJECTED
        if self.cum_qty > 0:
            return OrdStatus.PARTIALLY_FILLED
        if self.phase is Phase.PENDING_NEW:
            return OrdStatus.PENDING_NEW
        return OrdStatus.NEW

    def check(self) -> None:
        """Raise AssertionError if any invariant fails."""
        assert self.order_qty > 0, "order_qty > 0"
        assert 0 <= self.cum_qty <= self.order_qty, "0 <= cum_qty <= order_qty"
        assert self.cum_qty == sum(q for q, _ in self.fills.values()), "cum_qty = sum of fills"
        if self.is_working:
            assert self.cum_qty + self.leaves_qty == self.order_qty, "cum + leaves = order_qty"
        else:
            assert self.leaves_qty == 0, "closed order has leaves 0"
        if self.pending is not None:
            assert self.phase is Phase.LIVE and not self.is_filled, "pending only on a working live order"
        assert self.clordid not in self.chain, "clordid not reused in its chain"


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise IllegalTransition(msg)


def _matches(o: Order, clordid: str) -> bool:
    return clordid == o.clordid or (o.outstanding is not None and clordid == o.outstanding.clordid)


def request_cancel(o: Order, new_clordid: str) -> Order:
    """Record a 35=F sent with ClOrdID ``new_clordid``, OrigClOrdID = current."""
    _need(o.is_working, f"cancel of a closed order ({o.ord_status.name})")
    _need(o.outstanding is None, "a cancel/replace request is already outstanding")
    _need(
        bool(new_clordid) and new_clordid != o.clordid and new_clordid not in o.chain,
        "new ClOrdID must be fresh",
    )
    return _commit(o, outstanding=Request("cancel", new_clordid))


def request_replace(o: Order, new_clordid: str, *, order_qty: int, price: float | None) -> Order:
    """Record a 35=G sent with ClOrdID ``new_clordid``, OrigClOrdID = current."""
    _need(
        o.phase is Phase.LIVE and not o.is_filled, f"replace needs a live working order ({o.ord_status.name})"
    )
    _need(o.outstanding is None, "a cancel/replace request is already outstanding")
    _need(
        bool(new_clordid) and new_clordid != o.clordid and new_clordid not in o.chain,
        "new ClOrdID must be fresh",
    )
    _need(order_qty > 0, "order_qty must be positive")
    return _commit(o, outstanding=Request("replace", new_clordid, int(order_qty), price))


def apply_cancel_reject(o: Order, clordid: str) -> Order:
    """Apply a 35=9 for our outstanding request; the order returns to its prior status."""
    _need(
        o.outstanding is not None and o.outstanding.clordid == clordid,
        "cancel reject for no outstanding request",
    )
    return _commit(o, outstanding=None, pending=None)


def _commit(o: Order, **changes: object) -> Order:
    n = replace(o, **changes)  # type: ignore[arg-type]
    n.check()
    return n


def _advance_chain(o: Order) -> dict[str, object]:
    assert o.outstanding is not None
    return {
        "clordid": o.outstanding.clordid,
        "chain": (*o.chain, o.clordid),
        "outstanding": None,
        "pending": None,
    }


def apply_report(o: Order, r: ExecReport) -> Order:
    """Return the order after execution report ``r``; raise IllegalTransition if not allowed.

    A report whose ExecID was already applied returns ``o`` unchanged
    (PossDup / resend safety): check ``r.exec_id in o.exec_ids`` to tell.
    """
    if r.exec_id in o.exec_ids:
        return o
    _need(bool(r.exec_id), "ExecID required")
    _need(_matches(o, r.clordid), f"ClOrdID {r.clordid!r} is not this order's")
    et = r.exec_type
    seen = o.exec_ids | {r.exec_id}
    ch: dict[str, object] = {"exec_ids": seen}

    if et is ExecType.PENDING_NEW:
        _need(o.phase is Phase.PENDING_NEW, "PendingNew only before acceptance")
    elif et is ExecType.NEW:
        _need(o.phase is Phase.PENDING_NEW, "New only from PendingNew")
        ch["phase"] = Phase.LIVE
    elif et is ExecType.REJECTED:
        _need(
            o.phase is Phase.PENDING_NEW, "ExecType=8 rejects a new order only (use 35=9 for cancel/replace)"
        )
        ch.update(phase=Phase.REJECTED, outstanding=None)
    elif et is ExecType.TRADE:
        _need(o.phase is Phase.LIVE, "fill needs an accepted order")
        _need(r.last_qty > 0, "LastQty must be positive")
        _need(o.cum_qty + r.last_qty <= o.order_qty, "overfill: CumQty would exceed OrderQty")
        fills = dict(o.fills)
        fills[r.exec_id] = (int(r.last_qty), float(r.last_px))
        ch.update(cum_qty=o.cum_qty + r.last_qty, fills=MappingProxyType(fills))
        if o.cum_qty + r.last_qty == o.order_qty:
            # Filled: any pending or outstanding request is moot (Appendix D:
            # a fill that completes the order during PendingCancel).
            ch.update(pending=None)
    elif et is ExecType.TRADE_CANCEL:
        _need(r.exec_ref_id is not None and r.exec_ref_id in o.fills, "trade cancel must reference a fill")
        assert r.exec_ref_id is not None
        fills = dict(o.fills)
        q, _ = fills.pop(r.exec_ref_id)
        ch.update(cum_qty=o.cum_qty - q, fills=MappingProxyType(fills))
    elif et in (ExecType.PENDING_CANCEL, ExecType.PENDING_REPLACE):
        kind = "cancel" if et is ExecType.PENDING_CANCEL else "replace"
        _need(o.phase is Phase.LIVE and not o.is_filled, "pending state needs a live working order")
        _need(
            o.outstanding is not None and o.outstanding.kind == kind and r.clordid == o.outstanding.clordid,
            f"Pending{kind.title()} without a matching {kind} request",
        )
        ch["pending"] = kind
    elif et is ExecType.CANCELED:
        _need(o.is_working, "cancel of a closed order")
        if (
            o.outstanding is not None
            and o.outstanding.kind == "cancel"
            and r.clordid == o.outstanding.clordid
        ):
            ch.update(_advance_chain(o))
        else:
            _need(r.clordid == o.clordid, "unsolicited cancel must carry the current ClOrdID")
            ch.update(outstanding=None, pending=None)
        ch["phase"] = Phase.CANCELED
    elif et is ExecType.REPLACED:
        req = o.outstanding
        _need(
            req is not None and req.kind == "replace" and r.clordid == req.clordid,
            "Replaced without a matching replace request",
        )
        _need(o.phase is Phase.LIVE and not o.is_filled, "replace needs a live working order")
        assert req is not None
        new_qty = r.order_qty if r.order_qty is not None else req.order_qty
        assert new_qty is not None
        _need(new_qty >= o.cum_qty, "replace below CumQty")
        ch.update(_advance_chain(o))
        ch.update(order_qty=int(new_qty), price=r.price if r.price is not None else req.price)
    elif et is ExecType.RESTATED:
        _need(o.phase is Phase.LIVE, "restatement needs a live order")
        _need(r.clordid == o.clordid, "restatement carries the current ClOrdID")
        if r.order_qty is not None:
            _need(r.order_qty >= o.cum_qty and r.order_qty > 0, "restated qty below CumQty")
            ch["order_qty"] = int(r.order_qty)
        if r.price is not None:
            ch["price"] = r.price
    elif et in (ExecType.EXPIRED, ExecType.DONE_FOR_DAY):
        _need(o.phase is Phase.LIVE and not o.is_filled, f"{et.name} needs a live working order")
        _need(r.clordid == o.clordid, f"{et.name} carries the current ClOrdID")
        ch.update(
            phase=Phase.EXPIRED if et is ExecType.EXPIRED else Phase.DONE_FOR_DAY,
            outstanding=None,
            pending=None,
        )
    else:  # pragma: no cover - enum is closed
        raise IllegalTransition(f"unsupported ExecType {et!r}")

    try:
        return _commit(o, **ch)
    except AssertionError as exc:  # an invariant the guards above did not catch
        raise IllegalTransition(f"invariant violated: {exc}") from exc
