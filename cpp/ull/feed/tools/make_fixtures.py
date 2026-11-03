"""Generate the feed handler's offline fixtures (SYNTHETIC, seeded).

These are not captures of a live exchange. They are seeded simulations
written in the exact message shapes of the public Coinbase Exchange `full`
channel (+ REST level-3 book) and the Binance spot diff-depth stream
(+ REST /api/v3/depth), wrapped in the QPRC recording format that
cpp/ull/feed/include/qp/ull/recorder.hpp reads. This script keeps its own
reference book, independent of the C++ code, and writes the expected final
book next to each recording.

What the recordings exercise, by construction:
  - the stream starts with a book already populated (orders opened before
    we subscribed), so the handler must wait for the REST snapshot;
  - the snapshot body arrives in the recording several frames AFTER its own
    sequence, so frames on both sides of it are buffered and some must be
    discarded;
  - for Binance, the snapshot's lastUpdateId falls inside an event's [U, u]
    range, which is the case Binance's rule U <= lastUpdateId+1 <= u covers.

Run from the repo root:  python cpp/ull/feed/tools/make_fixtures.py
"""

from __future__ import annotations

import json
import random
import struct
from decimal import Decimal
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
SCALE = 10**8


def fx(d: Decimal) -> int:
    v = d * SCALE
    assert v == v.to_integral_value(), d
    return int(v)


def fmt(v: int) -> str:
    """Same text as qp::ull::format_fixed."""
    neg = v < 0
    u = -v if neg else v
    f = f"{u % SCALE:08d}".rstrip("0")
    s = ("-" if neg else "") + str(u // SCALE)
    return s + "." + f if f else s


def dump(bids: dict[int, int], asks: dict[int, int]) -> str:
    o = "".join(f"B {fmt(p)} {fmt(q)}\n" for p, q in sorted(bids.items(), reverse=True) if q)
    return o + "".join(f"A {fmt(p)} {fmt(q)}\n" for p, q in sorted(asks.items()) if q)


def qprc(records: list[tuple[int, int, int, bytes]]) -> bytes:
    out = bytearray(b"QPRC" + struct.pack("<HH", 1, 0))
    for ns, line, kind, body in records:
        out += struct.pack("<QBBI", ns, line, kind, len(body)) + body
    return bytes(out)


def dec(d: Decimal) -> str:
    """Plain decimal text as exchanges send it (never exponent form)."""
    return format(d, "f")


def js(d: dict) -> bytes:
    return json.dumps(d, separators=(",", ":")).encode()


# ---- Coinbase full ------------------------------------------------------------


def coinbase(seed: int, pre: int, n: int, snap_at: int, snap_delay: int, product: str = "BTC-USD") -> None:
    rng = random.Random(seed)
    orders: dict[str, tuple[str, Decimal, Decimal]] = {}  # id -> (side, px, size)
    seq = 5_000
    oid = 0
    frames: list[tuple[int, dict]] = []

    def px(side: str) -> Decimal:
        c = rng.randint(1, 400)
        return Decimal(10000) - Decimal(c) / 100 if side == "buy" else Decimal(10000) + Decimal(c) / 100

    def step() -> list[dict]:
        nonlocal oid
        msgs: list[dict] = []
        r = rng.random()
        if r < 0.45 or len(orders) < 10:
            oid += 1
            o = f"o{seed}-{oid}"
            side = rng.choice(["buy", "sell"])
            p = px(side)
            sz = Decimal(rng.randint(1, 500_000)) / 10**6
            msgs.append({"type": "received", "order_id": o, "side": side, "price": dec(p), "size": dec(sz)})
            msgs.append(
                {"type": "open", "order_id": o, "side": side, "price": dec(p), "remaining_size": dec(sz)}
            )
            orders[o] = (side, p, sz)
        elif r < 0.70:
            o = rng.choice(sorted(orders))
            side, p, sz = orders.pop(o)
            msgs.append(
                {
                    "type": "done",
                    "order_id": o,
                    "side": side,
                    "price": dec(p),
                    "remaining_size": dec(sz),
                    "reason": "canceled",
                }
            )
        elif r < 0.80:
            o = rng.choice(sorted(orders))
            side, p, sz = orders[o]
            ns = (sz * Decimal(rng.randint(1, 99)) / 100).quantize(Decimal("0.00000001"))
            if ns > 0:
                orders[o] = (side, p, ns)
                msgs.append(
                    {
                        "type": "change",
                        "order_id": o,
                        "side": side,
                        "price": dec(p),
                        "old_size": dec(sz),
                        "new_size": dec(ns),
                    }
                )
        else:
            maker_side = rng.choice(["buy", "sell"])
            book = [(p, o) for o, (s, p, _) in orders.items() if s == maker_side]
            if not book:
                return msgs
            p, maker = max(book) if maker_side == "buy" else min(book)
            _, _, msz = orders[maker]
            oid += 1
            taker = f"t{seed}-{oid}"
            fill = (
                msz
                if rng.random() < 0.4
                else (msz * Decimal(rng.randint(1, 99)) / 100).quantize(Decimal("0.00000001"))
            )
            if fill <= 0:
                return msgs
            tside = "sell" if maker_side == "buy" else "buy"
            msgs.append(
                {
                    "type": "received",
                    "order_id": taker,
                    "side": tside,
                    "size": dec(fill),
                    "order_type": "market",
                }
            )
            msgs.append(
                {
                    "type": "match",
                    "maker_order_id": maker,
                    "taker_order_id": taker,
                    "side": maker_side,
                    "price": dec(p),
                    "size": dec(fill),
                    "trade_id": oid,
                }
            )
            if fill == msz:
                del orders[maker]
                msgs.append(
                    {
                        "type": "done",
                        "order_id": maker,
                        "side": maker_side,
                        "price": dec(p),
                        "remaining_size": "0",
                        "reason": "filled",
                    }
                )
            else:
                orders[maker] = (maker_side, p, msz - fill)
            msgs.append(
                {"type": "done", "order_id": taker, "side": tside, "remaining_size": "0", "reason": "filled"}
            )
        return msgs

    snapshot = None
    t = 1_000_000
    count = 0
    while count < pre + n:
        for m in step():
            seq += 1
            m.update({"product_id": product, "sequence": seq, "time": "2026-11-03T00:00:00.000000Z"})
            if count >= pre:
                t += rng.randint(20_000, 400_000)
                frames.append((t, m))
        # Snapshot after the whole step, so its sequence matches its contents.
        if count == pre + snap_at and snapshot is None and seq > 5_000:
            bids = [[dec(p), dec(s), o] for o, (sd, p, s) in sorted(orders.items()) if sd == "buy"]
            asks = [[dec(p), dec(s), o] for o, (sd, p, s) in sorted(orders.items()) if sd == "sell"]
            snapshot = {"sequence": seq, "bids": bids, "asks": asks}
        count += 1
    recs: list[tuple[int, int, int, bytes]] = []
    snap_idx = (
        next(i for i, (_, m) in enumerate(frames) if m["sequence"] == snapshot["sequence"]) + snap_delay
    )
    hb = {"type": "heartbeat", "sequence": 0, "last_trade_id": 0, "product_id": product, "time": "x"}
    for i, (ts, m) in enumerate(frames):
        recs.append((ts, 0, 0, js(m)))
        if i == 3:
            recs.append(
                (
                    ts + 1,
                    0,
                    0,
                    js({"type": "subscriptions", "channels": [{"name": "full", "product_ids": [product]}]}),
                )
            )
        if i % 50 == 7:
            recs.append((ts + 2, 0, 0, js({**hb, "sequence": m["sequence"]})))
        if i == snap_idx:
            recs.append((ts + 3, 255, 1, js(snapshot)))
    b: dict[int, int] = {}
    a: dict[int, int] = {}
    for sd, p, s in orders.values():
        d = b if sd == "buy" else a
        d[fx(p)] = d.get(fx(p), 0) + fx(s)
    name = f"coinbase_full_{seed}"
    (OUT / f"{name}.qprc").write_bytes(qprc(recs))
    (OUT / f"{name}.expected").write_text(
        f"orders {len(orders)}\nlast_seq {seq}\n" + dump(b, a), encoding="utf-8", newline="\n"
    )
    print(name, len(recs), "records,", len(orders), "orders")


# ---- Binance diff-depth -----------------------------------------------------------


def binance(seed: int, pre: int, n: int, snap_at: int, snap_delay: int, symbol: str = "BTCUSDT") -> None:
    rng = random.Random(seed)
    bids: dict[int, int] = {}
    asks: dict[int, int] = {}
    uid = 70_000
    frames: list[tuple[int, dict]] = []
    snapshot = None
    t = 2_000_000
    for k in range(pre + n):
        changes = []
        for _ in range(rng.randint(1, 6)):
            side = rng.choice(["b", "a"])
            c = rng.randint(1, 300)
            p = Decimal(60000) - Decimal(c) / 100 if side == "b" else Decimal(60000) + Decimal(c) / 100
            q = Decimal(0) if rng.random() < 0.3 else Decimal(rng.randint(1, 9_000_000)) / 10**8
            changes.append((side, p, q))
        first = uid + 1
        uid += len(changes)
        ev = {
            "e": "depthUpdate",
            "E": 1_700_000_000_000 + k,
            "s": symbol,
            "U": first,
            "u": uid,
            "b": [],
            "a": [],
        }
        # Snapshot taken part-way through this event's id range: the snapshot
        # contains the first `cut` changes of the event, not the rest.
        cut = rng.randint(0, len(changes) - 1) if k == pre + snap_at else len(changes)
        for i, (side, p, q) in enumerate(changes):
            ev[side].append([f"{p:.8f}", f"{q:.8f}"])
            d = bids if side == "b" else asks
            if i == cut and snapshot is None:
                snapshot = {
                    "lastUpdateId": first + cut - 1,
                    "bids": [
                        [f"{Decimal(x) / SCALE:.8f}", f"{Decimal(y) / SCALE:.8f}"]
                        for x, y in sorted(bids.items(), reverse=True)
                    ],
                    "asks": [
                        [f"{Decimal(x) / SCALE:.8f}", f"{Decimal(y) / SCALE:.8f}"]
                        for x, y in sorted(asks.items())
                    ],
                }
            if q == 0:
                d.pop(fx(p), None)
            else:
                d[fx(p)] = fx(q)
        if k >= pre:
            t += rng.randint(50_000, 150_000)
            frames.append((t, ev))
    assert snapshot is not None
    recs: list[tuple[int, int, int, bytes]] = []
    snap_idx = snap_at + snap_delay
    for i, (ts, ev) in enumerate(frames):
        body = {"stream": f"{symbol.lower()}@depth@100ms", "data": ev} if i % 2 else ev
        recs.append((ts, 0, 0, js(body)))
        if i == snap_idx:
            recs.append((ts + 3, 255, 1, js(snapshot)))
    name = f"binance_depth_{seed}"
    (OUT / f"{name}.qprc").write_bytes(qprc(recs))
    (OUT / f"{name}.expected").write_text(
        f"last_seq {uid}\n" + dump(bids, asks), encoding="utf-8", newline="\n"
    )
    print(name, len(recs), "records,", len(bids) + len(asks), "levels")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    coinbase(seed=11, pre=300, n=1500, snap_at=40, snap_delay=25)
    coinbase(seed=12, pre=50, n=800, snap_at=5, snap_delay=60)
    binance(seed=21, pre=200, n=1200, snap_at=30, snap_delay=20)
    binance(seed=22, pre=40, n=600, snap_at=3, snap_delay=50)


if __name__ == "__main__":
    main()
