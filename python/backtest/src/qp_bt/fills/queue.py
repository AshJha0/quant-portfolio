"""Queue-position fills for resting orders and book-walking fills for takers.

Plain idea
----------
A simulated ("virtual") limit order is not in the real book, so the model
tracks one number for it: ``queue_ahead``, the visible quantity that was at
its price when it arrived and is still in front of it.

* A trade at the order's price eats ``queue_ahead`` first; only the excess
  fills the order (partial fills follow naturally).
* A trade at a price *through* the order (below a resting buy, above a
  resting sell) fills the remainder: in a price-time book the order would
  have been matched before the worse price traded.
* A drop in visible depth that is not a trade is a cancellation. Where in
  the queue it happened is unknown, so the model moves ``queue_ahead``
  forward by the expected share of the cancel that came from in front.

Formal statement
----------------
Let ``A`` be the quantity ahead and ``B`` the quantity behind (``A + B`` is
the tracked level depth). For a cancel of size ``c`` the update is
``A <- max(A - c * p(A, B), 0)``, with the power rule

    p(A, B) = A^n / (A^n + B^n)

of the probabilistic queue models used in open-source HFT backtesters
(hftbacktest ``PowerProbQueueFunc``; Rigtorp's "probabilistic queue
position" note). ``n = 1`` is pro-rata, large ``n`` pushes cancels to the
longer side, and ``cancel_ahead=0`` / ``cancel_ahead=1`` give the two
bounds (all cancels behind: most conservative; all in front: most
optimistic). Trades at the level reduce ``A`` first and the tracked depth
by the full trade size.

Worked example
--------------
A buy for 3 joins a 10-lot bid (``A = 10``, ``B = 0``). Two lots cancel at
that level: with ``B = 0`` they must have come from in front, so ``A = 8``.
A sell trade of 9 hits the bid: 8 clear the queue, 1 fills us. Another sell
trade of 5 at a lower price trades through us: the remaining 2 fill.

Limits stated plainly: the model never removes liquidity from the real book
(its fills do not change what later traders see), and latency is handled by
the caller submitting the order at arrival time, not decision time.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from qp_core.model import model

from .book_view import BookView, Side

__all__ = [
    "Fill",
    "MarketableResult",
    "PassiveOrder",
    "QueueAwareFillModel",
    "fill_marketable",
    "prob_ahead_power",
]

_EPS = 1e-12


@dataclass(frozen=True)
class Fill:
    """One execution of a simulated order."""

    oid: int
    side: Side
    price: float
    qty: float


@dataclass(frozen=True)
class MarketableResult:
    """Result of walking the book with a marketable order."""

    fills: tuple[Fill, ...]
    filled: float
    unfilled: float

    @property
    def avg_price(self) -> float:
        """Quantity-weighted fill price; NaN when nothing filled."""
        if self.filled <= 0:
            return math.nan
        return sum(f.price * f.qty for f in self.fills) / self.filled


def _opposite(side: Side) -> Side:
    return "sell" if side == "buy" else "buy"


def _check_qty(qty: float) -> None:
    if not (math.isfinite(qty) and qty > 0):
        raise ValueError(f"qty must be positive and finite, got {qty}")


@model(id="BT-FILL-QUEUE")
def fill_marketable(
    book: BookView, side: Side, qty: float, limit: float | None = None, oid: int = 0
) -> MarketableResult:
    """Walk the opposite side of ``book`` (IOC semantics).

    A buy takes ask levels best first while ``price <= limit``; a sell takes
    bid levels while ``price >= limit``. What cannot fill is reported in
    ``unfilled`` and is not left resting.
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    _check_qty(qty)
    remaining = qty
    fills: list[Fill] = []
    for lvl in book.levels(_opposite(side)):
        if remaining <= _EPS:
            break
        if limit is not None and (
            (side == "buy" and lvl.price > limit) or (side == "sell" and lvl.price < limit)
        ):
            break
        take = min(remaining, lvl.qty)
        if take > 0:
            fills.append(Fill(oid, side, lvl.price, take))
            remaining -= take
    remaining = max(remaining, 0.0)
    return MarketableResult(tuple(fills), qty - remaining, remaining)


@model(id="BT-FILL-QUEUE")
def prob_ahead_power(ahead: float, behind: float, n: float = 1.0) -> float:
    """Share of a cancel that came from in front: ``A^n / (A^n + B^n)``."""
    if ahead < 0 or behind < 0 or n < 0:
        raise ValueError("ahead, behind and n must be non-negative")
    if ahead <= 0:
        return 0.0
    if behind <= 0:
        return 1.0
    # ratio form avoids overflow for large n
    return 1.0 / (1.0 + (behind / ahead) ** n)


@dataclass
class PassiveOrder:
    """A resting simulated limit order and its estimated queue position."""

    oid: int
    side: Side
    price: float
    qty: float
    queue_ahead: float
    filled: float = 0.0

    @property
    def remaining(self) -> float:
        return self.qty - self.filled

    @property
    def done(self) -> bool:
        return self.remaining <= _EPS


@model(id="BT-FILL-QUEUE")
@dataclass
class QueueAwareFillModel:
    """Tracks the queue position of resting simulated orders against a book.

    Parameters
    ----------
    book : BookView
        The book the orders rest in (read on submit only).
    cancel_ahead : callable or float
        ``p(ahead, behind)`` in [0, 1], or a constant. Default: pro-rata
        (power rule with ``n = 1``).

    Event contract: call :meth:`on_trade` for every print and
    :meth:`on_depth` with the new visible depth after every level change
    that is *not* a trade (adds and cancels). The model reconciles its own
    tracked depth so a trade followed by a depth update is not counted twice.
    """

    book: BookView
    cancel_ahead: Callable[[float, float], float] | float = field(
        default=prob_ahead_power  # type: ignore[assignment]
    )
    _orders: dict[int, PassiveOrder] = field(default_factory=dict, init=False)
    _depth: dict[tuple[Side, float], float] = field(default_factory=dict, init=False)
    _seq: int = field(default=0, init=False)

    # --------------------------------------------------------------- orders
    def submit(self, side: Side, price: float, qty: float) -> PassiveOrder:
        """Rest a limit order at the back of its level; returns it.

        Raises ``ValueError`` if the order would cross the book: route
        marketable orders through :func:`fill_marketable`.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        _check_qty(qty)
        if not math.isfinite(price):
            raise ValueError("price must be finite")
        opp = self.book.levels(_opposite(side))
        if opp and ((side == "buy" and price >= opp[0].price) or (side == "sell" and price <= opp[0].price)):
            raise ValueError("order is marketable; use fill_marketable")
        depth = float(self.book.depth_at(side, price))
        self._seq += 1
        order = PassiveOrder(self._seq, side, price, qty, queue_ahead=depth)
        self._orders[order.oid] = order
        self._depth[(side, price)] = depth
        return order

    def cancel(self, oid: int) -> PassiveOrder | None:
        """Remove an order; returns it (with what had filled) or None."""
        return self._orders.pop(oid, None)

    @property
    def open_orders(self) -> list[PassiveOrder]:
        return [o for o in self._orders.values() if not o.done]

    # --------------------------------------------------------------- events
    def on_trade(self, price: float, qty: float, aggressor: Side) -> list[Fill]:
        """A print of ``qty`` at ``price``; the aggressor hit the other side."""
        _check_qty(qty)
        resting: Side = _opposite(aggressor)
        fills: list[Fill] = []
        at_level_left: dict[float, float] = {}
        for o in sorted(self._orders.values(), key=lambda x: x.oid):
            if o.done or o.side != resting:
                continue
            through = (resting == "buy" and price < o.price) or (resting == "sell" and price > o.price)
            if through:
                take = o.remaining
                o.queue_ahead = 0.0
            elif price == o.price:
                left = at_level_left.get(o.price, qty)
                eaten = min(left, o.queue_ahead)
                o.queue_ahead -= eaten
                left -= eaten
                take = min(left, o.remaining)
                at_level_left[o.price] = left - take
            else:
                continue
            if take > _EPS:
                o.filled += take
                fills.append(Fill(o.oid, o.side, o.price, take))
        key = (resting, price)
        if key in self._depth:
            self._depth[key] = max(self._depth[key] - qty, 0.0)
        return fills

    def on_depth(self, side: Side, price: float, new_depth: float) -> None:
        """Visible depth at a level changed for a non-trade reason."""
        key = (side, price)
        old = self._depth.get(key)
        if old is None:
            return
        new_depth = max(float(new_depth), 0.0)
        drop = old - new_depth
        if drop > _EPS:
            for o in self._orders.values():
                if o.side != side or o.price != price or o.done:
                    continue
                ahead = min(o.queue_ahead, old)
                behind = old - ahead
                p = (
                    float(self.cancel_ahead)
                    if isinstance(self.cancel_ahead, int | float)
                    else float(self.cancel_ahead(ahead, behind))
                )
                p = min(max(p, 0.0), 1.0)
                o.queue_ahead = max(ahead - drop * p, 0.0)
        for o in self._orders.values():
            if o.side == side and o.price == price:
                o.queue_ahead = min(o.queue_ahead, new_depth)
        self._depth[key] = new_depth
