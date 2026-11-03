"""Fill models for the ``qp_bt`` backtest core.

:class:`NextBarFill` (W24a, :mod:`qp_bt.fills.nextbar`) is the bar-level
fill: when, and at what price, a decision fills. The rest of this package
(W24b) is the queue-aware fill model for resting and marketable orders.


Plain idea. A resting limit order does not fill the moment the price touches
it: it waits behind the orders that were already at that price. It fills only
after trades have eaten the queue ahead of it, or when the price trades
through its level. A marketable order walks the visible book and may fill
only partly.

The model sees the book only through :class:`BookView`, the small read-only
seam it shares with the W25 reference book (``qp_exec.lob``). Integration
binds the W25 book to that protocol through
:class:`qp_bt.fills.lob_view.ReferenceBookView`.
"""

from .book_view import BookView, Level, Side
from .lob_view import ReferenceBookView
from .nextbar import NextBarFill
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
    "NextBarFill",
    "PassiveOrder",
    "QueueAwareFillModel",
    "ReferenceBookView",
    "Side",
    "fill_marketable",
    "prob_ahead_power",
]
