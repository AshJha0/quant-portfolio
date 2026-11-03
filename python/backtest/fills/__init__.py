"""Queue-aware fill model for the ``qp_bt`` backtest core (W24b).

Plain idea. A resting limit order does not fill the moment the price touches
it: it waits behind the orders that were already at that price. It fills only
after trades have eaten the queue ahead of it, or when the price trades
through its level. A marketable order walks the visible book and may fill
only partly.

The model sees the book only through :class:`BookView`, the small read-only
seam it shares with the W25 reference book (``qp_exec.lob``). Integration
binds the W25 book to that protocol; the tests here use a price-time FIFO
book written to the W25 semantics (price-time priority) as a stand-in.
"""

from .book_view import BookView, Level, Side
from .queue import (
    Fill,
    MarketableResult,
    PassiveOrder,
    QueueAwareFillModel,
    fill_marketable,
    prob_ahead_power,
)

__all__ = [
    "BookView",
    "Fill",
    "Level",
    "MarketableResult",
    "PassiveOrder",
    "QueueAwareFillModel",
    "Side",
    "fill_marketable",
    "prob_ahead_power",
]
