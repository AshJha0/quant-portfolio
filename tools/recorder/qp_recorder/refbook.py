"""Python reference books for replaying live recordings.

This module shares no code with the C++ (cpp/ull/feed) or Rust
(rust/ull/feed) handlers. It is the third, independent opinion the replay
tests compare both against.

Plain idea. A handler's book can only be judged at moments when the handler
itself says it is fresh. So for each recording we compute the book the venue
*meant*, keyed by something both sides can name:

- Binance diff-depth: the last update id ``u`` folded into the book. The
  reference takes every distinct event from both lines, orders them by
  ``u`` regardless of arrival, starts from the REST snapshot and applies
  the chain (Binance, "How to manage a local order book correctly": drop
  ``u <= lastUpdateId``; the first kept event must have
  ``U <= lastUpdateId + 1 <= u``; each next one ``U == previous u + 1``).
  The chain stops at the first hole, because past it nobody knows the book.
- Coinbase ``level2``/``level2_batch``: the channel has no sequence number,
  so the key is the record index. One line only; the in-band ``snapshot``
  replaces the book and each ``l2update`` sets absolute sizes.

Formal statement. Prices and sizes are int64 with 8 decimals. For a level
``(side, px, qty)`` let ``h = mix(mix(px) ^ qty ^ tag(side))``, with ``mix``
the SplitMix64 finaliser, ``tag(bid) = 0`` and
``tag(ask) = 0xA5A5A5A5A5A5A5A5``, all arithmetic mod 2^64. The book digest
is the sum of ``h`` over the levels with ``qty > 0``, mod 2^64. A sum can be
updated in O(1) per level change, which keeps the Python reference fast on
books of thousands of levels; the C++ and Rust replay tools recompute it
from the whole book after every record, so an error in their bookkeeping
cannot hide behind a shared incremental shortcut. Two different books
share a digest with probability about 2^-64 per comparison. This is a test
oracle, not a security boundary.

Worked example. Snapshot ``bids [["100.10","2"]]``, ``asks []``, then an
update setting bid 100.1 to 0.5: the book is one level, so its digest is
``level_hash("B", 10_010_000_000, 50_000_000)`` and the text form
(:meth:`L2.dump`) is ``"B 100.1 0.5\\n"``.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation

from .qprc import KIND_SNAPSHOT, Record

SCALE = 10**8
MASK = (1 << 64) - 1
ASK_TAG = 0xA5A5A5A5A5A5A5A5


def fixed(s: str) -> int:
    """Exchange decimal text to int64 with 8 decimals; refuses to round."""
    try:
        d = Decimal(s)
    except InvalidOperation as e:
        raise ValueError(f"bad decimal {s!r}") from e
    v = d * SCALE
    if v != v.to_integral_value():
        raise ValueError(f"more than 8 decimals: {s!r}")
    return int(v)


def fmt(v: int) -> str:
    neg = v < 0
    u = -v if neg else v
    f = f"{u % SCALE:08d}".rstrip("0")
    s = ("-" if neg else "") + str(u // SCALE)
    return s + "." + f if f else s


def mix(x: int) -> int:
    """SplitMix64 (Steele, Lea and Flood 2014), mod 2^64."""
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def level_hash(side: str, px: int, qty: int) -> int:
    return mix(mix(px & MASK) ^ (qty & MASK) ^ (0 if side == "B" else ASK_TAG))


class L2:
    def __init__(self) -> None:
        self.bids: dict[int, int] = {}
        self.asks: dict[int, int] = {}
        self.h = 0

    def set(self, side: str, px: int, qty: int) -> None:
        if qty < 0:
            raise ValueError("negative size")
        d = self.bids if side == "B" else self.asks
        old = d.pop(px, 0)
        if old:
            self.h = (self.h - level_hash(side, px, old)) & MASK
        if qty:
            d[px] = qty
            self.h = (self.h + level_hash(side, px, qty)) & MASK

    def dump(self) -> str:
        o = "".join(f"B {fmt(p)} {fmt(q)}\n" for p, q in sorted(self.bids.items(), reverse=True))
        return o + "".join(f"A {fmt(p)} {fmt(q)}\n" for p, q in sorted(self.asks.items()))


def digest(book: L2) -> int:
    return book.h


# ---- Binance ------------------------------------------------------------------


def _binance_event(data: bytes, symbol: str) -> dict | None:
    try:
        m = json.loads(data)
    except ValueError:
        return None
    if isinstance(m, dict) and "stream" in m and "data" in m:
        m = m["data"]
    if not isinstance(m, dict) or m.get("e") != "depthUpdate" or m.get("s") != symbol:
        return None
    return m


def binance_reference(records: list[Record], symbol: str) -> dict[int, int]:
    """Map update id ``u`` -> digest of the venue book after it.

    Uses the first snapshot in the recording; the snapshot itself is keyed
    by its ``lastUpdateId``.
    """
    snap = next((r for r in records if r.kind == KIND_SNAPSHOT), None)
    if snap is None:
        return {}
    s = json.loads(snap.data)
    book = L2()
    for p, q in s["bids"]:
        book.set("B", fixed(p), fixed(q))
    for p, q in s["asks"]:
        book.set("A", fixed(p), fixed(q))
    last = int(s["lastUpdateId"])
    out = {last: digest(book)}
    events: dict[int, dict] = {}
    for r in records:
        if r.kind != KIND_SNAPSHOT:
            ev = _binance_event(r.data, symbol)
            if ev is not None:
                events.setdefault(int(ev["u"]), ev)
    first = True
    for u in sorted(events):
        ev = events[u]
        big_u = int(ev["U"])
        if u <= last:
            continue
        ok = big_u <= last + 1 if first else big_u == last + 1
        if not ok:
            break  # a hole on both lines: the chain ends here
        first = False
        for p, q in ev["b"]:
            book.set("B", fixed(p), fixed(q))
        for p, q in ev["a"]:
            book.set("A", fixed(p), fixed(q))
        last = u
        out[u] = digest(book)
    return out


# ---- Coinbase level2 ------------------------------------------------------------


def coinbase_level2_reference(records: list[Record], product: str) -> dict[int, int]:
    """Map record index -> digest after that record, for records at which the
    reference book is live (an in-band snapshot has been seen)."""
    book = L2()
    live = False
    out: dict[int, int] = {}
    for i, r in enumerate(records):
        try:
            m = json.loads(r.data)
        except ValueError:
            m = None
        if isinstance(m, dict) and m.get("product_id", product) == product:
            t = m.get("type")
            if t == "snapshot":
                book = L2()
                for p, q in m["bids"]:
                    book.set("B", fixed(p), fixed(q))
                for p, q in m["asks"]:
                    book.set("A", fixed(p), fixed(q))
                live = True
            elif t == "l2update" and live:
                for side, p, q in m["changes"]:
                    book.set("B" if side == "buy" else "A", fixed(p), fixed(q))
        if live:
            out[i] = digest(book)
    return out
