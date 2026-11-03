"""Seeded synthetic audit log: benign flow plus optional planted abusers.

The benign population is built to be hard for the detectors, not easy:
market makers quote both sides and cancel most quotes within seconds (high
cancel ratios are normal for them), and directional traders work orders that
fill late in the day. Planted agents follow the textbook patterns the
detectors look for. Everything is "synthetic venue"; no real order flow.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from qp_posttrade.model import EventKind, OrderEvent, Side

SESSION = 23_400.0  # 6.5 h in seconds
TICK = 0.01


@dataclass(frozen=True)
class Planted:
    kind: str
    trader: str
    symbol: str


class _Log:
    def __init__(self) -> None:
        self.events: list[OrderEvent] = []
        self.n = 0

    def oid(self) -> str:
        self.n += 1
        return f"O{self.n:07d}"

    def add(
        self,
        ts: float,
        tr: str,
        sym: str,
        oid: str,
        kind: EventKind,
        side: Side,
        qty: float,
        px: float,
        owner: str = "",
    ) -> None:
        self.events.append(OrderEvent(ts, tr, f"ACC-{tr}", sym, oid, kind, side, qty, px, owner))


def _benign(log: _Log, rng: random.Random, symbols: list[str]) -> None:
    for s in symbols:
        mid = 100.0
        # market makers: two-sided quotes every ~20 s, cancelled 1-3 s later; some fill
        for mm in ("MM1", "MM2", "MM3"):
            t = rng.uniform(0, 20)
            while t < SESSION:
                mid += rng.gauss(0, 0.02)
                quotes: tuple[tuple[Side, float], ...] = (("BUY", mid - TICK), ("SELL", mid + TICK))
                for side, px in quotes:
                    o = log.oid()
                    q = rng.choice([100, 200, 300])
                    log.add(t, mm, s, o, "NEW", side, q, round(px, 2))
                    if rng.random() < 0.1:
                        log.add(t + rng.uniform(0.1, 1.0), mm, s, o, "FILL", side, q, round(px, 2))
                    else:
                        log.add(t + rng.uniform(1, 3), mm, s, o, "CANCEL", side, q, round(px, 2))
                t += rng.uniform(10, 30)
        # directional traders: work a parent all day, children fill
        for d in ("DT1", "DT2", "DT3", "DT4", "DT5"):
            dside: Side = rng.choice(["BUY", "SELL"])
            t = rng.uniform(0, 600)
            while t < SESSION:
                o = log.oid()
                q = rng.choice([100, 200, 500])
                px = round(100 + rng.gauss(0, 0.5), 2)
                log.add(t, d, s, o, "NEW", dside, q, px)
                if rng.random() < 0.7:
                    log.add(t + rng.uniform(0.5, 30), d, s, o, "FILL", dside, q, px)
                else:
                    log.add(t + rng.uniform(5, 60), d, s, o, "CANCEL", dside, q, px)
                t += rng.uniform(60, 400)


def _plant_spoof(log: _Log, rng: random.Random, tr: str, s: str, levels: int) -> None:
    t = rng.uniform(1000, SESSION - 1000)
    real: Side = rng.choice(["BUY", "SELL"])
    fake: Side = "SELL" if real == "BUY" else "BUY"
    step = TICK if fake == "BUY" else -TICK
    ids = []
    for k in range(levels):
        o = log.oid()
        ids.append(o)
        log.add(t + 0.1 * k, tr, s, o, "NEW", fake, 2000, round(100 - step * (k + 1), 2))
    o = log.oid()
    log.add(t + 1, tr, s, o, "NEW", real, 200, 100.0)
    log.add(t + 1.5, tr, s, o, "FILL", real, 200, 100.0)
    for k, i in enumerate(ids):
        log.add(t + 2 + 0.1 * k, tr, s, i, "CANCEL", fake, 2000, 0.0)


def _plant_wash(log: _Log, rng: random.Random, tr: str, s: str) -> None:
    t = rng.uniform(1000, SESSION - 1000)
    q = rng.choice([700, 900])
    px = round(100 + rng.gauss(0, 0.2), 2)
    b, se = log.oid(), log.oid()
    log.add(t, tr, s, b, "NEW", "BUY", q, px)
    log.add(t, f"{tr}x", s, se, "NEW", "SELL", q, px, owner=f"ACC-{tr}")
    log.add(t + 0.2, tr, s, b, "FILL", "BUY", q, px)
    log.add(t + 0.2, f"{tr}x", s, se, "FILL", "SELL", q, px, owner=f"ACC-{tr}")


def _plant_close(log: _Log, rng: random.Random, tr: str, s: str) -> None:
    t = SESSION - 240
    for _ in range(6):
        o = log.oid()
        log.add(t, tr, s, o, "NEW", "BUY", 1000, 100.2)
        log.add(t + 1, tr, s, o, "FILL", "BUY", 1000, 100.2)
        t += rng.uniform(20, 35)


def generate(
    seed: int, symbols: list[str] | None = None, plant: bool = True
) -> tuple[list[OrderEvent], list[Planted]]:
    rng = random.Random(seed)
    syms = symbols or ["AAA", "BBB"]
    log = _Log()
    _benign(log, rng, syms)
    planted: list[Planted] = []
    if plant:
        _plant_spoof(log, rng, "BAD1", syms[0], 1)
        planted.append(Planted("SPOOFING", "BAD1", syms[0]))
        _plant_spoof(log, rng, "BAD2", syms[-1], 4)
        planted.append(Planted("LAYERING", "BAD2", syms[-1]))
        _plant_wash(log, rng, "BAD3", syms[0])
        planted.append(Planted("WASH", "BAD3+BAD3x", syms[0]))
        _plant_close(log, rng, "BAD4", syms[-1])
        planted.append(Planted("MARKING_CLOSE", "BAD4", syms[-1]))
    log.events.sort(key=lambda e: e.ts)
    return log.events, planted
