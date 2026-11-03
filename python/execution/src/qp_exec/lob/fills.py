"""Fills with latency and queue position, and markouts.

Plain idea
----------
A passive order fills only when the queue ahead of it has traded or
cancelled away, and it is filled most often just before the price moves
against it (adverse selection). Markouts measure that: the mid some time
after a fill against the fill price.

Formal statement
----------------
Markout of a fill at price ``p`` and time ``t`` for horizon ``h``, from the
point of view of the side that received the fill (``s = +1`` buy, ``-1``
sell), in ticks::

    m(h) = s * (mid(t + h) - p)

``mid(t + h)`` is the last mid at or before ``t + h`` (as-of join). A
negative mean markout on passive fills is adverse selection.

Worked example
--------------
>>> import numpy as np
>>> markouts(np.array([1.0]), np.array([100.0]), np.array([1]),
...          np.array([0.0, 1.5, 3.0]), np.array([100.5, 99.5, 99.0]), (1.0, 2.0))
array([[-0.5, -1. ]])
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .book import ReferenceBook
from .events import ADD, Event, apply_event
from .types import Fill, Order

MARKOUT_HORIZONS_S = (1.0, 10.0, 60.0)


def markouts(fill_t: NDArray[np.float64], fill_px: NDArray[np.float64], sign: NDArray[np.int64],
             mid_t: NDArray[np.float64], mid: NDArray[np.float64],
             horizons: tuple[float, ...] = MARKOUT_HORIZONS_S) -> NDArray[np.float64]:
    """Markouts in price units, one row per fill, one column per horizon."""
    out = np.full((fill_t.size, len(horizons)), np.nan)
    for j, h in enumerate(horizons):
        idx = np.searchsorted(mid_t, fill_t + h, side="right") - 1
        ok = idx >= 0
        out[ok, j] = sign[ok] * (mid[idx[ok]] - fill_px[ok])
    return out


@dataclass
class PassiveRun:
    fills: list[Fill] = field(default_factory=list)
    fill_times: list[float] = field(default_factory=list)
    queue_ahead: list[tuple[float, int]] = field(default_factory=list)
    posted_at: float | None = None


def simulate_passive_order(events: list[Event], times: list[float], order: Order,
                           submit_time: float, latency: float) -> PassiveRun:
    """Replay recorded/generated flow and insert our order after ``latency``.

    The order reaches the book at ``submit_time + latency``; everything in the
    stream up to then has already happened (that is the cost of latency: the
    queue it joins is the one at arrival, not at decision). Returns our fills
    (partial fills included) and our queue position after every event.

    Our order id must not collide with ids in the stream.
    """
    book = ReferenceBook()
    run = PassiveRun()
    arrive = submit_time + latency
    posted = False
    for ev, t in zip(events, times, strict=True):
        if not posted and t >= arrive:
            r = book.add(order, int(arrive * 1e6))
            posted = True
            run.posted_at = arrive
            for f in r.fills:
                run.fills.append(f)
                run.fill_times.append(arrive)
        # Our order may have been fully filled; stream events referencing
        # other ids are unaffected.
        if ev[0] == ADD and ev[2] == order.order_id:
            raise ValueError("order id collides with the event stream")
        for f in apply_event(book, ev):
            if order.order_id in (f.taker_id, f.maker_id):
                run.fills.append(f)
                run.fill_times.append(t)
        if posted:
            qa = book.queue_ahead(order.order_id)
            if qa is not None:
                run.queue_ahead.append((t, qa))
    return run

