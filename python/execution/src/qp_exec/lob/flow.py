"""Order-flow generators that drive the reference book.

Plain idea
----------
To study fills and impact without exchange data, generate order flow whose
statistics are known. Two standard models:

* **Zero intelligence** (Smith, Farmer, Gillemot and Krishnamurthy 2003;
  Cont, Stoikov and Talreja 2010): limit orders arrive at a Poisson rate at a
  random distance from the opposite best, market orders at a Poisson rate,
  and each resting order is cancelled at a constant per-order rate.
* **Queue reactive** (Huang, Lehalle and Rosenbaum 2015): the rates at the
  best quotes depend on the size of the queue there. A small queue attracts
  limit orders; a large one attracts cancels and market orders.

Formal statement
----------------
Both are continuous-time Markov chains on the book; the event chosen next is
drawn with probability proportional to its rate and the clock advances by an
exponential with the total rate (Gillespie). Integer timestamps are the event
time in microseconds.

Worked example
--------------
>>> from qp_exec.lob.flow import ZIParams, zero_intelligence
>>> sim = zero_intelligence(ZIParams(), n_events=2000, seed=1)
>>> sim.book.check_invariants()
>>> bool(min(sim.spreads) >= 1)
True
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .book import ReferenceBook
from .events import ADD, CANCEL, Event, apply_event
from .types import Fill, Side


@dataclass(frozen=True)
class ZIParams:
    """Rates per unit time. ``depth`` is how many ticks from the opposite best
    a limit order may land (uniformly in 1..depth)."""

    limit_rate: float = 1.0  # per side
    market_rate: float = 0.2  # per side
    cancel_rate: float = 0.05  # per resting order
    depth: int = 10
    lot: int = 1
    mid: int = 10_000


@dataclass
class FlowResult:
    book: ReferenceBook
    events: list[Event] = field(default_factory=list)
    times: list[float] = field(default_factory=list)  # one per event
    mids: list[float] = field(default_factory=list)  # mid after each event (nan if one-sided)
    spreads: list[int] = field(default_factory=list)  # only when two-sided
    trades: list[tuple[float, Fill]] = field(default_factory=list)


def _seed_book(book: ReferenceBook, mid: int, depth: int, lot: int, nid: int) -> int:
    for k in range(1, depth + 1):
        nid += 1
        apply_event(book, (ADD, 0, nid, int(Side.BUY), lot, mid - k, 1, 1, 0, 0, 0))
        nid += 1
        apply_event(book, (ADD, 0, nid, int(Side.SELL), lot, mid + k, 1, 1, 0, 0, 0))
    return nid


def _run(book: ReferenceBook, n_events: int, seed: int,
         rates: Callable[[ReferenceBook], tuple[float, float, float, float, float]],
         params: ZIParams, nid: int) -> FlowResult:
    """Gillespie loop. ``rates`` returns (limit buy, limit sell, market buy,
    market sell, total cancel)."""
    rng = np.random.default_rng(seed)
    res = FlowResult(book)
    t = 0.0
    for _ in range(n_events):
        r = np.array(rates(book))
        tot = float(r.sum())
        t += rng.exponential(1.0 / tot)
        ts = int(t * 1e6)
        k = int(np.searchsorted(np.cumsum(r), rng.random() * tot, side="right"))
        k = min(k, 4)
        ev: Event
        bb, ba = book.best_bid(), book.best_ask()
        if k in (0, 1):
            side = Side.BUY if k == 0 else Side.SELL
            off = int(rng.integers(1, params.depth + 1))
            if side == Side.BUY:
                ref = ba if ba is not None else (bb + 1 if bb is not None else params.mid)
                px = ref - off
            else:
                ref = bb if bb is not None else (ba - 1 if ba is not None else params.mid)
                px = ref + off
            nid += 1
            ev = (ADD, ts, nid, int(side), params.lot, max(px, 1), 1, 1, 0, 0, 0)
        elif k in (2, 3):
            side = Side.BUY if k == 2 else Side.SELL
            nid += 1
            ev = (ADD, ts, nid, int(side), params.lot, 0, 2, 2, 0, 0, 0)
        else:
            ids = list(book._orders)
            if not ids:
                continue
            ev = (CANCEL, ts, ids[int(rng.integers(len(ids)))], 0, 0, 0, 0, 0, 0, 0, 0)
        fills = apply_event(book, ev)
        res.events.append(ev)
        res.times.append(t)
        for f in fills:
            res.trades.append((t, f))
        bb, ba = book.best_bid(), book.best_ask()
        if bb is not None and ba is not None:
            res.mids.append(0.5 * (bb + ba))
            res.spreads.append(ba - bb)
        else:
            res.mids.append(float("nan"))
    return res


def zero_intelligence(params: ZIParams, n_events: int, seed: int) -> FlowResult:
    book = ReferenceBook()
    nid = _seed_book(book, params.mid, params.depth, params.lot, 10**9)

    def rates(b: ReferenceBook) -> tuple[float, float, float, float, float]:
        return (params.limit_rate, params.limit_rate, params.market_rate, params.market_rate,
                params.cancel_rate * len(b._orders))

    return _run(book, n_events, seed, rates, params, nid)


def queue_reactive(params: ZIParams, n_events: int, seed: int, q_ref: int = 5) -> FlowResult:
    """Huang-Lehalle-Rosenbaum style: rates at the best depend on queue size.

    Limit arrival at a side scales with ``q_ref / (q + 1)`` (small queue
    attracts liquidity), market orders with ``q / q_ref`` capped at 2x.
    Cancels stay per-order.
    """
    book = ReferenceBook()
    nid = _seed_book(book, params.mid, params.depth, params.lot, 10**9)

    def q(b: ReferenceBook, side: Side) -> int:
        d = b.depth(side, 1)
        return d[0][1] if d else 0

    def rates(b: ReferenceBook) -> tuple[float, float, float, float, float]:
        qb, qa = q(b, Side.BUY), q(b, Side.SELL)
        lb = params.limit_rate * q_ref / (qb + 1)
        la = params.limit_rate * q_ref / (qa + 1)
        mb = params.market_rate * min(qa / q_ref, 2.0)  # buys hit the ask queue
        ma = params.market_rate * min(qb / q_ref, 2.0)
        return (lb, la, mb, ma, params.cancel_rate * len(b._orders))

    return _run(book, n_events, seed, rates, params, nid)
