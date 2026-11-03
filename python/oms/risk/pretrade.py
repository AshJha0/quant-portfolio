"""Pre-trade risk checks (W28d).

Plain idea: before an order leaves, ask "if every working order and this one
filled in full, would any limit break?" If yes, reject. Exposure is counted
conservatively while a change is in flight: a pending cancel still counts in
full until it is acknowledged, and a pending amend counts as the larger of
the old and new quantity until the venue answers. That is what makes the
limit hold under amend/cancel races.

Formal statement, per account a and symbol s, with signed position q,
working buy quantity B and working sell quantity S (pending amends at
max(old, new)):

    worst_long  = q + B (+ new buy qty)     <= position_limit
    worst_short = -q + S (+ new sell qty)   <= position_limit
    sum over working orders of qty * px (+ new notional) <= credit_limit

Sources: SEC Rule 15c3-5(c)(1) (credit and capital thresholds, erroneous
orders by price or size, duplicative orders); FIA Best Practices (2024):
price bands, fat-finger, message throttles, self-trade prevention.

Throttle rule (stated): a message over the per-session limit is REJECTED,
never queued. Queuing hides latency from the strategy and can release a
burst after the market has moved.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class OrderIntent:
    """The fields risk needs. Seam: maps from the r12-l2-oms core Order."""

    cl_ord_id: str
    account: str
    session: str
    symbol: str
    side: Side
    qty: float
    px: float
    parent_id: str = ""  # set for EMS child slices


@dataclass(frozen=True)
class Limits:
    max_order_qty: float
    max_order_notional: float
    price_band: float  # fraction of reference price, e.g. 0.05
    position_limit: float  # absolute worst-case position per symbol
    credit_limit: float  # working + new notional per account
    throttle_msgs: int
    throttle_window_ns: int
    dup_window_ns: int = 1_000_000_000


@dataclass
class _Working:
    intent: OrderIntent
    qty: float  # open qty counted for exposure
    pending_qty: float | None = None  # amend in flight


@dataclass
class RiskBook:
    limits: Limits
    ref_px: dict[str, float] = field(default_factory=dict)
    restricted: set[str] = field(default_factory=set)
    positions: dict[tuple[str, str], float] = field(default_factory=dict)
    _working: dict[str, _Working] = field(default_factory=dict)
    _seen_ids: set[str] = field(default_factory=set)
    _recent: deque[tuple[int, tuple[object, ...]]] = field(default_factory=deque)
    _msgs: dict[str, deque[int]] = field(default_factory=dict)

    # ---- exposure -------------------------------------------------------
    def _exposure_qty(self, w: _Working) -> float:
        return max(w.qty, w.pending_qty) if w.pending_qty is not None else w.qty

    def worst_case(self, account: str, symbol: str) -> tuple[float, float]:
        """(worst long, worst short) including working orders."""
        q = self.positions.get((account, symbol), 0.0)
        b = s = 0.0
        for w in self._working.values():
            if w.intent.account == account and w.intent.symbol == symbol:
                if w.intent.side is Side.BUY:
                    b += self._exposure_qty(w)
                else:
                    s += self._exposure_qty(w)
        return q + b, -q + s

    def working_notional(self, account: str) -> float:
        return sum(
            self._exposure_qty(w) * w.intent.px for w in self._working.values() if w.intent.account == account
        )

    # ---- checks ---------------------------------------------------------
    def _throttle(self, session: str, now_ns: int) -> str | None:
        dq = self._msgs.setdefault(session, deque())
        while dq and dq[0] <= now_ns - self.limits.throttle_window_ns:
            dq.popleft()
        if len(dq) >= self.limits.throttle_msgs:
            return "THROTTLE"
        dq.append(now_ns)
        return None

    def check_new(self, o: OrderIntent, now_ns: int) -> str | None:
        """Return None to accept (and book as working), else a reject reason."""
        L = self.limits
        if (r := self._throttle(o.session, now_ns)) is not None:
            return r
        if o.qty <= 0 or o.px <= 0:
            return "BAD_QTY_OR_PX"
        if o.symbol in self.restricted:
            return "RESTRICTED"
        if o.cl_ord_id in self._seen_ids:
            return "DUP_CLORDID"
        key = (o.account, o.symbol, o.side, o.qty, o.px)
        while self._recent and self._recent[0][0] <= now_ns - L.dup_window_ns:
            self._recent.popleft()
        # Content duplicates catch a manual double submit. EMS slices of one parent
        # repeat size and price on purpose, so they are exempt (ClOrdID dedupe
        # still applies to them).
        if not o.parent_id and any(k == key for _, k in self._recent):
            return "DUP_CONTENT"
        if o.qty > L.max_order_qty:
            return "FAT_FINGER_QTY"
        if o.qty * o.px > L.max_order_notional:
            return "MAX_NOTIONAL"
        ref = self.ref_px.get(o.symbol)
        if ref is None:
            return "NO_REF_PRICE"
        if abs(o.px - ref) > L.price_band * ref:
            return "PRICE_BAND"
        if self._self_trade(o):
            return "SELF_TRADE"
        lng, sht = self.worst_case(o.account, o.symbol)
        if (lng + o.qty if o.side is Side.BUY else sht + o.qty) > L.position_limit:
            return "POSITION_LIMIT"
        if self.working_notional(o.account) + o.qty * o.px > L.credit_limit:
            return "CREDIT_LIMIT"
        self._seen_ids.add(o.cl_ord_id)
        self._recent.append((now_ns, key))
        self._working[o.cl_ord_id] = _Working(o, o.qty)
        return None

    def _self_trade(self, o: OrderIntent) -> bool:
        """Cancel-newest STP: reject an order that would cross our own resting order."""
        for w in self._working.values():
            r = w.intent
            if r.account != o.account or r.symbol != o.symbol or r.side is o.side:
                continue
            if (o.side is Side.BUY and o.px >= r.px) or (o.side is Side.SELL and o.px <= r.px):
                return True
        return False

    def check_amend(self, cl_ord_id: str, new_qty: float, now_ns: int) -> str | None:
        """An amend is checked as if the larger quantity were new exposure."""
        w = self._working.get(cl_ord_id)
        if w is None:
            return "UNKNOWN_ORDER"
        if w.pending_qty is not None:
            return "AMEND_IN_FLIGHT"
        if (r := self._throttle(w.intent.session, now_ns)) is not None:
            return r
        if new_qty <= 0 or new_qty > self.limits.max_order_qty:
            return "FAT_FINGER_QTY"
        extra = max(0.0, new_qty - w.qty)
        lng, sht = self.worst_case(w.intent.account, w.intent.symbol)
        if (lng if w.intent.side is Side.BUY else sht) + extra > self.limits.position_limit:
            return "POSITION_LIMIT"
        if self.working_notional(w.intent.account) + extra * w.intent.px > self.limits.credit_limit:
            return "CREDIT_LIMIT"
        w.pending_qty = new_qty
        return None

    # ---- venue feedback -------------------------------------------------
    def on_amend_result(self, cl_ord_id: str, accepted: bool) -> None:
        w = self._working[cl_ord_id]
        if accepted and w.pending_qty is not None:
            w.qty = w.pending_qty
        w.pending_qty = None

    def on_fill(self, cl_ord_id: str, qty: float) -> None:
        w = self._working[cl_ord_id]
        sign = 1.0 if w.intent.side is Side.BUY else -1.0
        k = (w.intent.account, w.intent.symbol)
        self.positions[k] = self.positions.get(k, 0.0) + sign * qty
        w.qty -= qty
        if w.pending_qty is not None:
            w.pending_qty = max(0.0, w.pending_qty - qty)
        if w.qty <= 1e-12 and not w.pending_qty:
            del self._working[cl_ord_id]

    def on_terminal(self, cl_ord_id: str) -> None:
        """Cancel ack, reject or expiry: exposure released only now."""
        self._working.pop(cl_ord_id, None)
