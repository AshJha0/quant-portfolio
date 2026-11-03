"""Operational controls between a quoting model and the book.

Plain idea
----------
A model quote is only an opinion. Before it reaches the venue a desk applies
a few hard rules that do not depend on the model being right:

* **Stale data pulls quotes.** If the book feed (or the hedge venue) is
  flagged stale, the maker cannot see the market it is quoting into, so both
  quotes are cancelled and nothing new is posted until the flag clears. A
  one-sided or crossed book counts as stale too.
* **Position limits.** Past ``widen_at`` lots the side that would grow the
  position is moved ``widen_ticks`` further away; at ``max_position`` that
  side is pulled.
* **Throttle.** At most ``max_updates`` new quote messages per ``window``.
  Pulls are never throttled: removing risk always goes through.
* **Tick rounding.** Bids round down and asks round up to the tick, and a bid
  never sits at or above the best ask (post-only would reject it anyway).

Formal statement
----------------
``decide`` maps (model quote in price units, inventory, book top, flags, time)
to a ``Decision``: integer-tick bid and ask, each possibly ``None`` (pulled),
plus the reason. It is a pure function of its inputs and the throttle history.

Worked example
--------------
>>> c = QuoteController(ControlParams(tick=0.01, max_position=5, widen_at=3, widen_ticks=2))
>>> c.decide(model_bid=99.987, model_ask=100.013, q=0, best_bid=9998, best_ask=10002,
...          book_stale=False, t=0.0)
Decision(bid=9998, ask=10002, reason='ok')
>>> c.decide(model_bid=99.987, model_ask=100.013, q=0, best_bid=9998, best_ask=10002,
...          book_stale=True, t=1.0)
Decision(bid=None, ask=None, reason='stale')
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ControlParams:
    tick: float
    max_position: int  # pull the inventory-growing side at |q| >= this
    widen_at: int | None = None  # widen the inventory-growing side at |q| >= this
    widen_ticks: int = 1
    max_updates: int | None = None  # throttle: new quote messages per window
    window: float = 1.0

    def __post_init__(self) -> None:
        if self.tick <= 0 or self.max_position < 1:
            raise ValueError("tick > 0 and max_position >= 1 required")
        if self.widen_at is not None and not 0 < self.widen_at <= self.max_position:
            raise ValueError("0 < widen_at <= max_position required")


@dataclass(frozen=True, slots=True)
class Decision:
    bid: int | None  # ticks
    ask: int | None
    reason: str  # 'ok', 'stale', 'throttled', 'no_model'


class QuoteController:
    def __init__(self, p: ControlParams) -> None:
        self.p = p
        self._sent: deque[float] = deque()
        self.last = Decision(None, None, "init")

    def _to_ticks(self, bid: float | None, ask: float | None) -> tuple[int | None, int | None]:
        t = self.p.tick
        # A small tolerance keeps 99.99/0.01 from landing on 9998.999...
        b = None if bid is None else math.floor(bid / t + 1e-9)
        a = None if ask is None else math.ceil(ask / t - 1e-9)
        return b, a

    def decide(self, model_bid: float | None, model_ask: float | None, q: int,
               best_bid: int | None, best_ask: int | None, book_stale: bool, t: float,
               hedge_stale: bool = False) -> Decision:
        p = self.p
        if book_stale or hedge_stale or best_bid is None or best_ask is None or best_bid >= best_ask:
            return self._commit(Decision(None, None, "stale"), t, pull=True)
        b, a = self._to_ticks(model_bid, model_ask)
        if p.widen_at is not None:
            if q >= p.widen_at and b is not None:
                b -= p.widen_ticks
            if q <= -p.widen_at and a is not None:
                a += p.widen_ticks
        if q >= p.max_position:
            b = None
        if q <= -p.max_position:
            a = None
        # Never cross the opposite best (post-only safety) and keep bid < ask.
        if b is not None:
            b = min(b, best_ask - 1)
        if a is not None:
            a = max(a, best_bid + 1)
        if b is not None and a is not None and b >= a:
            a = b + 1
        if b is None and a is None:
            return self._commit(Decision(None, None, "no_model"), t, pull=True)
        return self._commit(Decision(b, a, "ok"), t, pull=False)

    def _commit(self, d: Decision, t: float, pull: bool) -> Decision:
        if (d.bid, d.ask) == (self.last.bid, self.last.ask):
            self.last = d
            return d
        p = self.p
        if not pull and p.max_updates is not None:
            while self._sent and self._sent[0] <= t - p.window:
                self._sent.popleft()
            if len(self._sent) >= p.max_updates:
                # Keep what is live, except that a side the new decision pulls is pulled.
                kept = Decision(self.last.bid if d.bid is not None else None,
                                self.last.ask if d.ask is not None else None, "throttled")
                self.last = kept
                return kept
            self._sent.append(t)
        self.last = d
        return d
