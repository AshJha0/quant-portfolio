"""Book events, a seeded random event stream, and replay with per-event digests.

An event is eleven signed 64-bit integers::

    op, ts, order_id, side, qty, price, ord_type, tif, flags, display_qty, peg

``op``: 1 add, 2 cancel, 3 modify (``qty`` is the new remaining quantity),
4 start auction, 5 uncross (``price`` is the reference price).
``flags``: bit 0 post-only, bit 1 hidden. Unused fields are 0.

The binary file form is the events packed little-endian (``<11q`` each). The
C++ replay tool ``exec_core_replay`` reads that file and prints one digest per
event; ``replay_digests`` here produces the same list.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from .book import ReferenceBook
from .types import Fill, Order, OrdType, Peg, Side, Tif

ADD, CANCEL, MODIFY, START_AUCTION, UNCROSS = 1, 2, 3, 4, 5
_REC = struct.Struct("<11q")

Event = tuple[int, int, int, int, int, int, int, int, int, int, int]


def apply_event(book: ReferenceBook, ev: Event) -> tuple[Fill, ...]:
    op, ts, oid, side, qty, price, ot, tif, flags, disp, peg = ev
    if op == ADD:
        o = Order(oid, Side(side), qty, price, OrdType(ot), Tif(tif), bool(flags & 1),
                  bool(flags & 2), disp, Peg(peg))
        return book.add(o, ts).fills
    if op == CANCEL:
        return book.cancel(oid, ts)[1]
    if op == MODIFY:
        return book.modify(oid, qty, ts)[1]
    if op == START_AUCTION:
        book.start_auction()
        return ()
    if op == UNCROSS:
        return book.uncross(ts, price)
    raise ValueError(f"unknown op {op}")


def random_events(n: int, seed: int, mid: int = 1000, width: int = 12,
                  auction_every: int = 5000) -> Iterator[Event]:
    """A seeded stream exercising every order type, cancels, modifies, auctions.

    Prices sit in ``mid +- width`` ticks so the book stays small and dense and
    the queues are long enough for priority mistakes to show.
    """
    rng = np.random.default_rng(seed)
    # A private book tells the generator which ids are still live, so cancels
    # and modifies hit real orders and the book stays near ``target`` orders.
    shadow = ReferenceBook()
    nid = 0
    in_auction = False
    target = 4 * width
    for ts in range(1, n + 1):
        if auction_every and ts % auction_every == 0:
            in_auction = not in_auction
            ev: Event
            if in_auction:
                ev = (START_AUCTION, ts, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            else:
                ev = (UNCROSS, ts, 0, 0, 0, mid, 0, 0, 0, 0, 0)
            apply_event(shadow, ev)
            yield ev
            continue
        u = rng.random()
        live = shadow._orders
        p_cancel = 0.15 if len(live) < target else 0.45
        if live and u < p_cancel + 0.07:
            ids = list(live)
            oid = ids[int(rng.integers(len(ids)))]
            if u < p_cancel:
                ev = (CANCEL, ts, oid, 0, 0, 0, 0, 0, 0, 0, 0)
            else:
                ev = (MODIFY, ts, oid, 0, int(rng.integers(0, 12)), 0, 0, 0, 0, 0, 0)
            apply_event(shadow, ev)
            yield ev
            continue
        nid += 1
        side = int(rng.integers(1, 3))
        qty = int(rng.integers(1, 10))
        off = int(rng.integers(-width, width + 1))
        # buys centred slightly below mid, sells above, with overlap so orders cross
        price = mid + off + (-2 if side == 1 else 2)
        v = rng.random()
        ot, tif, flags, disp, peg = 1, 1, 0, 0, 0
        if v < 0.06:
            ot, tif = 2, (2 if rng.random() < 0.7 else 3)
            price = 0
        elif v < 0.12:
            tif = 2
        elif v < 0.16:
            tif = 3
        elif v < 0.22:
            flags = 1
        elif v < 0.27:
            flags = 2
        elif v < 0.33:
            disp = int(rng.integers(1, 4))
            qty += 6
        elif v < 0.36:
            peg, price = (1 if rng.random() < 0.6 else 2), 0
        # In the auction call, non-DAY types are rejected: that is the reject path.
        ev = (ADD, ts, nid, side, qty, price, ot, tif, flags, disp, peg)
        apply_event(shadow, ev)
        yield ev


def write_events(path: Path, events: Iterable[Event]) -> int:
    n = 0
    with open(path, "wb") as f:
        for ev in events:
            f.write(_REC.pack(*ev))
            n += 1
    return n


def read_events(path: Path) -> Iterator[Event]:
    data = Path(path).read_bytes()
    yield from _REC.iter_unpack(data)


def replay_digests(events: Iterable[Event], check: bool = False) -> list[int]:
    book = ReferenceBook()
    out: list[int] = []
    for ev in events:
        fills = apply_event(book, ev)
        if check:
            book.check_invariants()
        out.append(book.digest(fills))
    return out
