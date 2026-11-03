"""Bind the W25 reference book (``qp_exec.lob.ReferenceBook``) to :class:`BookView`.

Plain idea
----------
The queue-aware fill model reads a book through two calls, ``levels`` and
``depth_at``, in float prices. The W25 reference book speaks integer ticks and
integer lots and has its own ``Side`` enum. This adapter converts between the
two and nothing else, so ``qp_bt`` does not import ``qp_exec`` at all: the
book is duck-typed (it needs ``depth(side, levels)`` and ``order_state(id)``).

Formal statement
----------------
For a book side ``s`` and tick size ``k``, with the displayed L2 depth
``D_s = [(t_i, q_i)]`` from the best price outward,

    levels(s)        = [(t_i * k, q_i - own_i) for each i with q_i - own_i > 0]
    depth_at(s, p)   = q_j - own_j  where t_j = round(p / k), else 0

where ``own_i`` is the visible quantity of the excluded (our own, simulated)
orders resting at ``t_i``. Excluding them is what makes the view *public*:
the model must not see itself in the queue.

Worked example
--------------
Bids at tick 99 hold 10 lots, of which our order 7 is 3. With ``k = 0.01``
and ``exclude={7}``, ``depth_at("buy", 0.99)`` is 7.0 and ``levels("buy")``
is ``[Level(0.99, 7.0)]``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from .book_view import Level, Side

__all__ = ["ReferenceBookView"]

# Numeric values of qp_exec.lob.Side (part of its published cross-language
# contract: BUY = 1, SELL = 2). IntEnum members compare equal to these.
_SIDE_CODE: dict[str, int] = {"buy": 1, "sell": 2}


class _LobLike(Protocol):
    def depth(self, side: Any, levels: int) -> list[tuple[int, int]]: ...

    def order_state(self, order_id: int) -> tuple[int, int, int, bool] | None: ...


class ReferenceBookView:
    """Read-only :class:`BookView` over a ``qp_exec.lob.ReferenceBook``.

    Parameters
    ----------
    book : ReferenceBook (or anything with ``depth`` and ``order_state``)
    tick_size : float
        Price of one tick; float prices are ``ticks * tick_size``.
    exclude : iterable of int
        Order ids hidden from the view (the simulated order itself).
        :meth:`hide` adds more later.
    """

    def __init__(self, book: _LobLike, tick_size: float = 1.0,
                 exclude: Iterable[int] = ()) -> None:
        if not tick_size > 0:
            raise ValueError(f"tick_size must be > 0, got {tick_size!r}")
        self.book = book
        self.tick_size = float(tick_size)
        self.exclude: set[int] = set(exclude)

    def hide(self, order_id: int) -> None:
        """Hide an order (ours) from the public view."""
        self.exclude.add(order_id)

    def _own(self, tick: int) -> int:
        # order_state gives (price, visible, reserve, active). It carries no
        # side, but a resting buy and sell cannot share a tick in an uncrossed
        # book, so the tick alone places the order.
        own = 0
        for oid in self.exclude:
            st = self.book.order_state(oid)
            if st is not None and st[0] == tick and st[3]:
                own += st[1]
        return own

    def _rows(self, side: Side) -> list[tuple[int, int]]:
        if side not in _SIDE_CODE:
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        return self.book.depth(_SIDE_CODE[side], 1 << 30)

    def levels(self, side: Side) -> Sequence[Level]:
        out: list[Level] = []
        for tick, qty in self._rows(side):
            q = qty - self._own(tick)
            if q > 0:
                out.append(Level(tick * self.tick_size, float(q)))
        return out

    def depth_at(self, side: Side, price: float) -> float:
        tick = round(price / self.tick_size)
        for t, qty in self._rows(side):
            if t == tick:
                return float(max(qty - self._own(t), 0))
        return 0.0
