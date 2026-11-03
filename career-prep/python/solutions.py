"""Reference solutions for part B of the Python pack.

Each function is tested against a brute-force version on random inputs in
``career-prep/tests/test_python_pack.py``.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict, deque
from collections.abc import Iterable, Iterator, Sequence

import pandas as pd


def max_drawdown(prices: Sequence[float]) -> float:
    """YB1: largest fall from a running peak, as a fraction of that peak."""
    peak = float("-inf")
    worst = 0.0
    for p in prices:
        peak = max(peak, p)
        worst = max(worst, (peak - p) / peak)
    return worst


def vwap(trades: Iterable[tuple[float, float]]) -> float:
    """YB2: sum(price x qty) / sum(qty)."""
    notional = volume = 0.0
    for price, qty in trades:
        notional += price * qty
        volume += qty
    if volume == 0:
        raise ValueError("no volume")
    return notional / volume


def rolling_max(xs: Sequence[float], k: int) -> list[float]:
    """YB3: monotone deque of indices whose values decrease front to back."""
    if k <= 0:
        raise ValueError("k must be positive")
    out: list[float] = []
    dq: deque[int] = deque()
    for i, x in enumerate(xs):
        while dq and xs[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(xs[dq[0]])
    return out


class LRUCache:
    """YB4: OrderedDict keeps recency order; both operations are O(1)."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._d: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int | None:
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key: int, value: int) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)


def merge_sorted(streams: Iterable[Iterable[tuple[int, float]]]) -> Iterator[tuple[int, float]]:
    """YB5: k-way merge with a heap; ties keep stream order (heapq.merge is stable)."""
    return heapq.merge(*streams, key=lambda tv: tv[0])


def asof_join(trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """YB6: last quote at or before each trade. Both frames need a sorted ``time`` column."""
    return pd.merge_asof(trades.sort_values("time"), quotes.sort_values("time"),
                         on="time", direction="backward")
