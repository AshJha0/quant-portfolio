"""Independent checks for the trading-system design pack.

The key states each sizing answer from a one-line formula. Here each one is
recomputed by a different route: an event-by-event simulation (ring buffer,
token bucket, snapshot replay), a brute-force search (router allocation),
explicit matching of client orders (central risk book), a numerical
indifference-price solve (Avellaneda-Stoikov), or a share-by-share P&L
(implementation shortfall). ``career-prep/tests/test_final_packs.py``
compares them with ``keys/answers.md``.

Run ``python career-prep/system-design/verify.py`` to print the table.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
from scipy.optimize import brentq

Number = Fraction | float


def _feed_bandwidth() -> float:
    # One second of traffic, byte by byte would be silly; count bits per message.
    bits_per_msg = 64 * 8
    return sum(itertools.repeat(bits_per_msg, 5_000_000)) / 1e9


def _ring_capacity() -> Fraction:
    # Simulate in 1 microsecond steps: 1 arrival per us, 0.4 drained per us.
    depth = Fraction(0)
    peak = Fraction(0)
    for _ in range(2000):
        depth += 1
        depth = max(depth - Fraction(2, 5), Fraction(0))
        peak = max(peak, depth)
    cap = 1
    while cap < peak:
        cap *= 2
    return Fraction(cap)


def _reservation_price() -> float:
    """Average of bid and ask indifference prices under CARA utility.

    Value of holding q units with terminal price ~ N(s, sigma^2 tau) is
    -exp(-gamma x) E[exp(-gamma q S_T)]; the expectation is taken by
    Gauss-Hermite quadrature, and each indifference price by root-finding.
    """
    s, q, gamma, sigma, tau = 100.0, 10, 0.1, 2.0, 0.5
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    weights = weights / weights.sum()
    st = s + sigma * math.sqrt(tau) * nodes

    def value(cash: float, units: int) -> float:
        return float(
            -np.sum(weights * np.exp(-gamma * (cash + units * (st - s)))) * math.exp(-gamma * units * s)
        )

    base = value(0.0, q)
    bid = brentq(lambda p: value(-p, q + 1) - base, 50, 150)
    ask = brentq(lambda p: value(p, q - 1) - base, 50, 150)
    return 0.5 * (bid + ask)


def _internalised() -> Fraction:
    buys = [300, 20]
    sells = [120, 50]
    matched = 0
    i = j = 0
    while i < len(buys) and j < len(sells):
        m = min(buys[i], sells[j])
        matched += 2 * m  # both sides of the cross are internalised
        buys[i] -= m
        sells[j] -= m
        i += buys[i] == 0
        j += sells[j] == 0
    return Fraction(matched, 300 + 120 + 50 + 20)


def _router() -> Fraction:
    venues = [  # (size, price + fee) in exact cents
        (3000, Fraction(100) + Fraction(3, 1000)),
        (5000, Fraction(100) + Fraction(1, 1000)),
        (4000, Fraction(10001, 100) - Fraction(2, 1000)),
    ]
    best: Fraction | None = None
    step = 500
    for a in range(0, 3001, step):
        for b in range(0, 5001, step):
            c = 10_000 - a - b
            if 0 <= c <= 4000:
                cost = sum((n * p for n, (_s, p) in zip((a, b, c), venues, strict=True)), Fraction(0))
                avg = cost / 10_000
                best = avg if best is None else min(best, avg)
    assert best is not None
    return best


def _replay_ms() -> Fraction:
    worst = Fraction(0)
    for crash_at in range(0, 30_000, 7):  # events since start when the crash hits
        since_snapshot = crash_at % 10_000
        worst = max(worst, Fraction(since_snapshot * 2, 1000))
    for crash_at in (9_999, 19_999, 29_999):
        worst = max(worst, Fraction((crash_at % 10_000) * 2, 1000))
    return worst


def _token_bucket() -> Fraction:
    tokens = Fraction(100)
    rejected = 0
    for _ in range(100):
        if tokens >= 1:
            tokens -= 1
        else:
            rejected += 1
    t_prev = Fraction(0)
    for k in range(1, 801):
        t = Fraction(k, 80)
        tokens = min(Fraction(100), tokens + 50 * (t - t_prev))
        t_prev = t
        if tokens >= 1:
            tokens -= 1
        else:
            rejected += 1
    return Fraction(rejected)


def _shortfall_bp() -> Fraction:
    decision = Fraction(100)
    fills = [Fraction(10005, 100)] * 6000 + [Fraction(10010, 100)] * 3000
    paper_pnl = sum(Fraction(10030, 100) - decision for _ in range(10_000))
    real_pnl = sum(Fraction(10030, 100) - p for p in fills)
    return (paper_pnl - real_pnl) / (10_000 * decision) * 10_000


EXACT: dict[str, Callable[[], Number]] = {
    "X01": _feed_bandwidth,
    "X02": _reservation_price,
    "X03": _internalised,
    "X04": _router,
    "X05": _replay_ms,
    "X06": _token_bucket,
    "X07": _shortfall_bp,
    "X08": _ring_capacity,
}

TOL: dict[str, float] = {"X02": 1e-9}


if __name__ == "__main__":
    for pid, fn in EXACT.items():
        print(pid, fn())
