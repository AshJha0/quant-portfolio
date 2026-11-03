"""The read-only seam between the fill model and an order book (W25).

Integration note: this protocol is the whole contract the fill model needs
from the W25 reference book. Keep it this small; anything richer belongs in
the book, not here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple, Protocol, runtime_checkable

Side = Literal["buy", "sell"]


class Level(NamedTuple):
    """One visible price level: price and total resting quantity."""

    price: float
    qty: float


@runtime_checkable
class BookView(Protocol):
    """Read-only view of a limit order book.

    ``levels("sell")`` is the ask side, best (lowest) price first;
    ``levels("buy")`` is the bid side, best (highest) price first.
    ``depth_at`` returns 0.0 for an empty level.
    """

    def levels(self, side: Side) -> Sequence[Level]: ...

    def depth_at(self, side: Side, price: float) -> float: ...
