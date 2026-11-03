"""Independent checks for the probability and puzzles pack.

Every answer in ``keys/answers.md`` is recomputed here by a route that does
not copy the written solution: brute-force enumeration, an exact linear
solve of a Markov chain, a dynamic programme, or a closed form evaluated
numerically. A second, seeded Monte Carlo estimate is given for the problems
where a simulation is cheap; ``career-prep/tests`` checks both.

Run ``python career-prep/probability/verify.py`` to print the table.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from fractions import Fraction
from functools import cache

import numpy as np
from scipy.stats import norm

Number = Fraction | float


# ---------------------------------------------------------------- helpers
def _absorb_solve(states: list[str], step: Callable[[str], list[tuple[str, Fraction]]],
                  terminal: Callable[[str], bool], reward: Callable[[str], Fraction],
                  cost: Fraction = Fraction(0)) -> dict[str, Fraction]:
    """Solve v(s) = cost + sum p v(s') with v(terminal) = reward, exactly.

    Gaussian elimination over Fractions, so the result is exact.
    """
    live = [s for s in states if not terminal(s)]
    idx = {s: i for i, s in enumerate(live)}
    n = len(live)
    a = [[Fraction(0)] * (n + 1) for _ in range(n)]
    for s in live:
        i = idx[s]
        a[i][i] += 1
        a[i][n] += cost
        for nxt, p in step(s):
            if terminal(nxt):
                a[i][n] += p * reward(nxt)
            else:
                a[i][idx[nxt]] -= p
    for c in range(n):
        piv = next(r for r in range(c, n) if a[r][c] != 0)
        a[c], a[piv] = a[piv], a[c]
        for r in range(n):
            if r != c and a[r][c] != 0:
                f = a[r][c] / a[c][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c], strict=True)]
    return {s: a[idx[s]][n] / a[idx[s]][idx[s]] for s in live}


def _pattern_chain(pattern: str) -> dict[str, Fraction]:
    """Expected fair-coin flips until ``pattern`` appears (prefix states)."""
    states = [pattern[:k] for k in range(len(pattern) + 1)]

    def nxt(s: str, c: str) -> str:
        t = s + c
        while not pattern.startswith(t):
            t = t[1:]
        return t

    half = Fraction(1, 2)
    return _absorb_solve(states, lambda s: [(nxt(s, "H"), half), (nxt(s, "T"), half)],
                         lambda s: s == pattern, lambda s: Fraction(0), Fraction(1))


def _race(p1: str, p2: str) -> Fraction:
    """P(pattern p1 appears before p2) for a fair coin, from the empty history."""
    k = max(len(p1), len(p2))
    states = [""] + ["".join(t) for m in range(1, k + 1) for t in itertools.product("HT", repeat=m)]
    half = Fraction(1, 2)

    def done(s: str) -> bool:
        return s.endswith(p1) or s.endswith(p2)

    def step(s: str) -> list[tuple[str, Fraction]]:
        return [((s + c)[-k:], half) for c in "HT"]

    v = _absorb_solve(states, step, done, lambda s: Fraction(int(s.endswith(p1))))
    return v[""]


def _ruin(start: int, target: int, p: Fraction, duration: bool = False) -> Fraction:
    states = [str(i) for i in range(target + 1)]
    q = 1 - p
    v = _absorb_solve(
        states,
        lambda s: [(str(int(s) + 1), p), (str(int(s) - 1), q)],
        lambda s: s in ("0", str(target)),
        lambda s: Fraction(0) if duration else Fraction(int(s == str(target))),
        Fraction(1) if duration else Fraction(0),
    )
    return v[str(start)]


def _harmonic(n: int) -> Fraction:
    return sum((Fraction(1, k) for k in range(1, n + 1)), Fraction(0))


def _secretary(n: int) -> float:
    best = 0.0
    for r in range(2, n + 1):
        best = max(best, (r - 1) / n * sum(1 / (i - 1) for i in range(r, n + 1)))
    return best


@cache
def _polya(reds: int, blues: int, draws: int, target: int) -> Fraction:
    """P(exactly ``target`` more reds after ``draws`` Polya-urn draws)."""
    if draws == 0:
        return Fraction(int(target == 0))
    tot = reds + blues
    out = Fraction(0)
    if target > 0:
        out += Fraction(reds, tot) * _polya(reds + 1, blues, draws - 1, target - 1)
    out += Fraction(blues, tot) * _polya(reds, blues + 1, draws - 1, target)
    return out


def _dice2() -> list[int]:
    return [a + b for a in range(1, 7) for b in range(1, 7)]


def _craps() -> Fraction:
    sums = _dice2()
    pr = {s: Fraction(sums.count(s), 36) for s in set(sums)}
    win = pr[7] + pr[11]
    for point in (4, 5, 6, 8, 9, 10):
        win += pr[point] * pr[point] / (pr[point] + pr[7])
    return win


def _reroll_value(rerolls: int) -> Fraction:
    v = Fraction(7, 2)
    for _ in range(rerolls):
        v = sum((max(Fraction(f), v) for f in range(1, 7)), Fraction(0)) / 6
    return v


def _two_state_stationary() -> Fraction:
    a, b = Fraction(3, 10), Fraction(1, 10)  # P(0->1), P(1->0)
    return a / (a + b)


def _ballot() -> Fraction:
    good = total = 0
    for pos in itertools.combinations(range(10), 6):
        total += 1
        lead, ok = 0, True
        for i in range(10):
            lead += 1 if i in pos else -1
            ok = ok and lead > 0
        good += ok
    return Fraction(good, total)


def _rw_max_ge(n: int, a: int) -> Fraction:
    hit = 0
    for steps in itertools.product((1, -1), repeat=n):
        s, m = 0, 0
        for x in steps:
            s += x
            m = max(m, s)
        hit += m >= a
    return Fraction(hit, 2**n)


def _dyck(n_pairs: int) -> int:
    cnt = 0
    for steps in itertools.product((1, -1), repeat=2 * n_pairs):
        s = 0
        ok = True
        for x in steps:
            s += x
            if s < 0:
                ok = False
                break
        cnt += ok and s == 0
    return cnt


def _reflect_to_n(n: int) -> Fraction:
    states = [str(i) for i in range(n + 1)]

    def step(s: str) -> list[tuple[str, Fraction]]:
        i = int(s)
        if i == 0:
            return [("1", Fraction(1))]
        return [(str(i + 1), Fraction(1, 2)), (str(i - 1), Fraction(1, 2))]

    return _absorb_solve(states, step, lambda s: s == str(n), lambda s: Fraction(0),
                         Fraction(1))["0"]


def _enum_flips(n: int, f: Callable[[tuple[int, ...]], int]) -> Fraction:
    return Fraction(sum(f(t) for t in itertools.product((0, 1), repeat=n)), 2**n)


def _cards(pred: Callable[[tuple[tuple[int, int], ...]], bool]) -> Fraction:
    deck = [(r, s) for r in range(13) for s in range(4)]
    hits = sum(pred(h) for h in itertools.combinations(deck, 5))
    return Fraction(hits, math.comb(52, 5))


def _flush(h: tuple[tuple[int, int], ...]) -> bool:
    return len({s for _, s in h}) == 1


def _full_house(h: tuple[tuple[int, int], ...]) -> bool:
    counts = sorted([r for r, _ in h].count(r) for r in {r for r, _ in h})
    return counts == [2, 3]


def _expected_distinct(rolls: int) -> Fraction:
    tot = Fraction(0)
    for t in itertools.product(range(6), repeat=rolls):
        tot += len(set(t))
    return tot / 6**rolls


# ---------------------------------------------------------------- exact table
EXACT: dict[str, Callable[[], Number]] = {
    "P01": lambda: Fraction(sum(range(1, 7)), 6),
    "P02": lambda: Fraction(sum(max(a, b) for a in range(1, 7) for b in range(1, 7)), 36),
    "P03": lambda: Fraction(6),  # geometric mean 1/p; MC below
    "P04": lambda: 6 * _harmonic(6),
    "P05": lambda: _pattern_chain("HH")[""],
    "P06": lambda: _pattern_chain("HT")[""],
    "P07": lambda: _race("HH", "HT"),
    "P08": lambda: _race("THH", "HHH"),
    "P09": lambda: Fraction(sum(f == (1, 1) for f in itertools.product((0, 1), repeat=2)),
                           sum(max(f) == 1 for f in itertools.product((0, 1), repeat=2))),
    "P10": lambda: Fraction(sum(1 for car in range(3) for pick in range(3) if car != pick), 9),
    "P11": lambda: Fraction(1, 100) * Fraction(99, 100)
    / (Fraction(1, 100) * Fraction(99, 100) + Fraction(99, 100) * Fraction(5, 100)),
    "P12": lambda: _ruin(3, 10, Fraction(1, 2)),
    "P13": lambda: _ruin(3, 10, Fraction(3, 5)),
    "P14": lambda: _ruin(3, 10, Fraction(1, 2), duration=True),
    "P15": lambda: _ruin(5, 10, Fraction(1, 2), duration=True),
    "P16": lambda: 10 * _harmonic(10),
    "P17": lambda: float(next(n for n in range(1, 366)
                              if 1 - math.prod((365 - i) / 365 for i in range(n)) > 0.5)),
    "P18": lambda: 1 - math.prod((365 - i) / 365 for i in range(23)),
    "P19": lambda: Fraction(sum(sum(p[i] == i for i in range(6))
                                for p in itertools.permutations(range(6))), math.factorial(6)),
    "P20": lambda: Fraction(sum(all(p[i] != i for i in range(8))
                                for p in itertools.permutations(range(8))), math.factorial(8)),
    "P21": lambda: Fraction(3, 4),
    "P22": lambda: Fraction(1, 4),
    "P23": lambda: Fraction(2, 6),
    "P24": lambda: Fraction(1, 4),
    "P25": lambda: Fraction(3, 4),
    "P26": lambda: Fraction(1, 3),
    "P27": lambda: math.e,
    "P28": lambda: math.exp(-1.5),
    "P29": lambda: Fraction(1, 2) ** 4,
    "P30": lambda: Fraction(2, 5),
    "P31": lambda: Fraction(1, 5),
    "P32": lambda: Fraction(10),
    "P33": lambda: Fraction(20),
    "P34": _two_state_stationary,
    "P35": lambda: 1 / (1 - _two_state_stationary()),
    "P36": lambda: _reflect_to_n(4),
    "P37": lambda: _rw_max_ge(10, 3),
    "P38": _ballot,
    "P39": lambda: Fraction(_dyck(5)),
    "P40": lambda: _ruin(2, 3, Fraction(1, 2)),
    "P41": lambda: 6 * Fraction(7, 2),
    "P42": lambda: _polya(1, 1, 10, 5),
    "P43": lambda: Fraction(1),
    "P44": lambda: _enum_flips(10, lambda t: 1 + sum(t[i] != t[i + 1] for i in range(9))),
    "P45": lambda: _enum_flips(10, lambda t: sum(t[i] == t[i + 1] == 1 for i in range(9))),
    "P46": lambda: Fraction(math.comb(4, 2), math.comb(52, 2)),
    "P47": lambda: _cards(_flush),
    "P48": lambda: _cards(_full_house),
    "P49": lambda: Fraction(_dice2().count(7), _dice2().count(7) + _dice2().count(6)),
    "P50": _craps,
    "P51": lambda: _expected_distinct(6),
    "P52": lambda: Fraction(2 * 6 - 10, 10),  # (bp - q)/b with b=1
    "P53": lambda: _reroll_value(1),
    "P54": lambda: _reroll_value(2),
    "P55": lambda: _secretary(10),
    "P56": lambda: 1 / math.sqrt(math.pi),
    "P57": lambda: 0.25 + math.asin(0.5) / (2 * math.pi),
    "P58": lambda: math.sqrt(2 / math.pi),
    "P59": lambda: 2 * (1 - float(norm.cdf(1.0))),
    "P60": lambda: _pattern_chain("HHH")[""],
}


# ---------------------------------------------------------------- Monte Carlo
def _mc(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


N_MC = 200_000


def _mc_geom() -> float:
    return float(_mc(3).geometric(1 / 6, N_MC).mean())


def _mc_coupon(n: int, reps: int = 20_000) -> float:
    rng = _mc(4 + n)
    tot = 0
    for _ in range(reps):
        seen: set[int] = set()
        k = 0
        draws = rng.integers(0, n, 400)
        while len(seen) < n:
            seen.add(int(draws[k]))
            k += 1
        tot += k
    return tot / reps


def _mc_uniform_stat(fn: Callable[[np.ndarray], np.ndarray], m: int, seed: int) -> float:
    return float(fn(_mc(seed).random((N_MC, m))).mean())


def _mc_stick() -> float:
    u = np.sort(_mc(24).random((N_MC, 2)), axis=1)
    a, b, c = u[:, 0], u[:, 1] - u[:, 0], 1 - u[:, 1]
    return float(((a < 0.5) & (b < 0.5) & (c < 0.5)).mean())


def _mc_until_sum() -> float:
    u = _mc(27).random((N_MC, 20)).cumsum(axis=1)
    return float(((u <= 1).sum(axis=1) + 1).mean())


def _mc_bvn() -> float:
    rng = _mc(57)
    x = rng.standard_normal(N_MC)
    y = 0.5 * x + math.sqrt(0.75) * rng.standard_normal(N_MC)
    return float(((x > 0) & (y > 0)).mean())


def _mc_bm_max() -> float:
    steps = 2000
    w = _mc(59).standard_normal((20_000, steps)).cumsum(axis=1) / math.sqrt(steps)
    return float((w.max(axis=1) >= 1).mean())


def _mc_monty() -> float:
    rng = _mc(10)
    car = rng.integers(0, 3, N_MC)
    pick = rng.integers(0, 3, N_MC)
    return float((car != pick).mean())  # switching wins iff first pick was wrong


def _mc_bayes() -> float:
    rng = _mc(11)
    d = rng.random(2_000_000) < 0.01
    pos = np.where(d, rng.random(d.size) < 0.99, rng.random(d.size) < 0.05)
    return float(d[pos].mean())


def _mc_two_kids() -> float:
    k = _mc(9).integers(0, 2, (N_MC, 2))
    one = k.max(axis=1) == 1
    return float((k.min(axis=1)[one] == 1).mean())


MONTE_CARLO: dict[str, tuple[Callable[[], float], float]] = {
    # id: (estimator, absolute tolerance)
    "P03": (_mc_geom, 0.06),
    "P04": (lambda: _mc_coupon(6), 0.2),
    "P09": (_mc_two_kids, 0.01),
    "P10": (_mc_monty, 0.01),
    "P11": (_mc_bayes, 0.01),
    "P16": (lambda: _mc_coupon(10), 0.3),
    "P21": (lambda: _mc_uniform_stat(lambda u: u.max(axis=1), 3, 21), 0.005),
    "P22": (lambda: _mc_uniform_stat(lambda u: u.min(axis=1), 3, 22), 0.005),
    "P23": (lambda: _mc_uniform_stat(lambda u: np.sort(u, axis=1)[:, 1], 5, 23), 0.005),
    "P24": (_mc_stick, 0.005),
    "P25": (lambda: _mc_uniform_stat(lambda u: np.where(u[:, 0] < 0.5, 1 - u[:, 0], u[:, 0]),
                                     1, 25), 0.005),
    "P26": (lambda: _mc_uniform_stat(lambda u: np.abs(u[:, 0] - u[:, 1]), 2, 26), 0.005),
    "P27": (_mc_until_sum, 0.01),
    "P31": (lambda: float(np.minimum(_mc(31).exponential(1 / 2, N_MC),
                                     _mc(131).exponential(1 / 3, N_MC)).mean()), 0.005),
    "P56": (lambda: float(_mc(56).standard_normal((N_MC, 2)).max(axis=1).mean()), 0.01),
    "P57": (_mc_bvn, 0.005),
    "P58": (lambda: float(np.abs(_mc(58).standard_normal(N_MC)).mean()), 0.01),
    "P59": (_mc_bm_max, 0.02),
}


def as_float(x: Number) -> float:
    return float(x)


if __name__ == "__main__":  # pragma: no cover
    for pid, fn in EXACT.items():
        val = fn()
        mc = MONTE_CARLO.get(pid)
        extra = f"  MC {mc[0]():.4f}" if mc else ""
        print(f"{pid}  {val!s:>24}  {float(val):.6f}{extra}")
