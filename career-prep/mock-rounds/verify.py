"""Independent checks for the mock rounds.

Keys give the pen-and-paper argument. Here every graded answer is
recomputed another way: by enumeration (dice, inversions, stairs), a linear
solve of the Markov chain (two heads), simulation of the search or the
array growth, quadrature against a density (option prices, expected
shortfall, E[max(W, 0)]), ``ctypes`` struct layout on this machine (the
struct sizes; x86-64 and AArch64 agree), address arithmetic (cache lines),
and fill-by-fill accounting (VWAP, P&L). ``career-prep/tests/test_final_packs.py``
compares each with ``keys/round-N.md``.

Run ``python career-prep/mock-rounds/verify.py`` to print the table.
"""

from __future__ import annotations

import bisect
import ctypes
import itertools
import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import binom

Number = Fraction | float
SQ2PI = math.sqrt(2 * math.pi)


def _density(x: float, var: float = 1.0) -> float:
    return math.exp(-0.5 * x * x / var) / math.sqrt(var) / SQ2PI


def _cdf(x: float) -> float:
    v, _ = quad(_density, 0.0, x, epsabs=1e-13, epsrel=1e-12)
    return 0.5 + v  # symmetric density: P(Z < 0) = 1/2


# Round 1
def _max_two_dice() -> Fraction:
    rolls = list(itertools.product(range(1, 7), repeat=2))
    return Fraction(sum(max(r) for r in rolls), len(rolls))


def _two_heads() -> Fraction:
    # E = 1 + P E over transient states {0 heads, 1 head}: (I - P) E = 1.
    p = [[Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 2), Fraction(0)]]
    a = [[1 - p[0][0], -p[0][1]], [-p[1][0], 1 - p[1][1]]]
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    return (a[1][1] - a[0][1]) / det


def _corr_x_xy() -> float:
    # Exact covariance matrix of (X, X + Y) from the linear map, unit variances.
    m = np.array([[1.0, 0.0], [1.0, 1.0]])
    c = m @ m.T
    return float(c[0, 1] / math.sqrt(c[0, 0] * c[1, 1]))


def _t_stat() -> float:
    daily_mean, daily_sd = 1.0 / 252, 1.0 / math.sqrt(252)
    n = 4 * 252
    return daily_mean / (daily_sd / math.sqrt(n))


# Round 2
def _call_r0() -> float:
    sd = 0.2

    def f(z: float) -> float:
        return max(100 * math.exp(-0.5 * sd * sd + sd * z) - 100, 0.0) * _density(z)

    v, _ = quad(f, 0.5 * sd, 12, epsabs=1e-13, epsrel=1e-13)
    return v


def _digital_put_r0() -> float:
    return _cdf(0.1)  # P(S_T < 100) = P(Z < sigma/2)


def _fwd_rate() -> Fraction:
    # Lend 0.90 from today to year 2, borrow 0.95 to year 1 per unit at year 1.
    df1, df2 = Fraction(95, 100), Fraction(90, 100)
    return (1 / df2 - 1 / df1) / (1 / df1)


def _e_max_w4() -> float:
    v, _ = quad(lambda x: x * _density(x, 4.0), 0, 60, epsabs=1e-13, epsrel=1e-12)
    return v


# Round 3
def _z(level: float) -> float:
    return brentq(lambda x: _cdf(x) - level, 0, 6, xtol=1e-13)


def _var_r3() -> float:
    return 200_000 * _z(0.99)


def _es_r3() -> float:
    z = _z(0.975)
    v, _ = quad(lambda x: x * _density(x), z, 40, epsabs=1e-13, epsrel=1e-12)
    return 10_000 * v / 0.025


def _port_vol() -> float:
    cov = np.diag([0.1**2, 0.2**2])
    w = np.array([0.5, 0.5])
    return float(math.sqrt(w @ cov @ w))


def _kupiec_tail() -> float:
    return sum(math.comb(250, k) * 0.01**k * 0.99 ** (250 - k) for k in range(6, 251))


# Round 4
def _stairs() -> Fraction:
    count = 0
    for length in range(5, 11):
        count += sum(1 for steps in itertools.product((1, 2), repeat=length) if sum(steps) == 10)
    return Fraction(count)


def _bsearch_probes() -> Fraction:
    worst = 0
    n = 1_000_000
    for target in (-1, n, 2 * 524_288 - 1, n // 3 + 0.5, n - 0.5, 0.5):
        lo, hi, probes = 0, n - 1, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            probes += 1
            if mid < target:
                lo = mid + 1
            else:
                hi = mid - 1
        worst = max(worst, probes)
    return Fraction(worst)


def _inversions() -> Fraction:
    a = [3, 1, 4, 1, 5, 9, 2, 6]
    seen: list[int] = []
    count = 0
    for x in a:  # count earlier elements strictly greater than x
        count += len(seen) - bisect.bisect_right(seen, x)
        bisect.insort(seen, x)
    return Fraction(count)


def _reallocations() -> Fraction:
    cap, size, moves = 1, 0, 0
    for _ in range(1000):
        if size == cap:
            cap *= 2
            moves += 1
        size += 1
    return Fraction(moves)


# Round 5
class _S1(ctypes.Structure):
    _fields_ = [("a", ctypes.c_char), ("b", ctypes.c_double), ("c", ctypes.c_int32)]


class _S2(ctypes.Structure):
    _fields_ = [("b", ctypes.c_double), ("c", ctypes.c_int32), ("a", ctypes.c_char)]


def _lines() -> Fraction:
    return Fraction(len({(8 * i) // 64 for i in range(1000)}))


def _slot() -> Fraction:
    return Fraction(5000 & (1024 - 1))


# Round 6
def _vwap() -> Fraction:
    fills = [(100, Fraction(1000, 100)), (300, Fraction(1004, 100)), (600, Fraction(1002, 100))]
    return sum((q * p for q, p in fills), Fraction(0)) / sum(q for q, _ in fills)


def _participation_minutes() -> Fraction:
    per_min = Fraction(2_000_000, 390)
    done, minute = Fraction(0), 0
    while done + per_min / 10 < 50_000:
        done += per_min / 10
        minute += 1
    return minute + (50_000 - done) / (per_min / 10)


def _dice_fair() -> Fraction:
    rolls = list(itertools.product(range(1, 7), repeat=2))
    return Fraction(sum(6 + a + b for a, b in rolls), len(rolls))


def _fifo_pnl() -> Fraction:
    lots = [Fraction(10002, 100)] * 1000
    realised = sum((Fraction(10005, 100) - lots.pop(0) for _ in range(600)), Fraction(0))
    unrealised = sum((Fraction(9998, 100) - p for p in lots), Fraction(0))
    return realised + unrealised


EXACT: dict[str, Callable[[], Number]] = {
    "M1.1": _max_two_dice,
    "M1.2": _two_heads,
    "M1.3": _corr_x_xy,
    "M1.4": _t_stat,
    "M2.1": _call_r0,
    "M2.2": _digital_put_r0,
    "M2.3": _fwd_rate,
    "M2.4": _e_max_w4,
    "M3.1": _var_r3,
    "M3.2": _es_r3,
    "M3.3": _port_vol,
    "M3.4": _kupiec_tail,
    "M4.1": _stairs,
    "M4.2": _bsearch_probes,
    "M4.3": _inversions,
    "M4.4": _reallocations,
    "M5.1": lambda: Fraction(ctypes.sizeof(_S1)),
    "M5.2": lambda: Fraction(ctypes.sizeof(_S2)),
    "M5.3": _lines,
    "M5.4": _slot,
    "M6.1": _vwap,
    "M6.2": _participation_minutes,
    "M6.3": _dice_fair,
    "M6.4": _fifo_pnl,
}

TOL: dict[str, float] = {"M3.4": 1e-9}

SECOND_ROUTE: dict[str, Callable[[], float]] = {"M3.4": lambda: float(binom.sf(5, 250, 0.01))}


if __name__ == "__main__":
    for pid, fn in EXACT.items():
        print(pid, fn())
