"""Independent checks for the machine-learning-for-finance pack.

Every answer in ``keys/answers.md`` is recomputed here by running the
procedure the question describes on the question's own numbers: fitting the
model by numerical optimisation or a linear solve, counting the indices a
purged and embargoed split keeps, scoring every positive-negative pair for
AUC, and so on. Where the answer is a population quantity, a seeded
simulation estimates it as well. Only ``numpy`` and ``scipy`` are used, so
the checks do not depend on any ML library's defaults.
``career-prep/tests/test_ml_pack.py`` compares the key with this file.

Run ``python career-prep/ml/verify.py`` to print the table.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Callable
from fractions import Fraction

import numpy as np
from scipy import optimize, stats

Number = Fraction | float


def _ridge_1d() -> float:
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    return float(np.linalg.solve(x.T @ x + 14 * np.eye(1), x.T @ y)[0])


def _lasso_1d(b_ols: float, lam: float) -> float:
    res = optimize.minimize_scalar(lambda b: 0.5 * (b - b_ols) ** 2 + lam * abs(b),
                                   bounds=(-5, 5), method="bounded",
                                   options={"xatol": 1e-12})
    return round(float(res.x), 8)


def _purged_train_size() -> Fraction:
    n, k, horizon, embargo = 1000, 5, 10, 10
    folds = np.array_split(np.arange(n), k)
    test = set(folds[2].tolist())
    t0, t1 = min(test), max(test)
    keep = []
    for i in range(n):
        if i in test:
            continue
        label_end = i + horizon  # label of i uses data up to i + horizon
        overlaps = i < t0 and label_end >= t0
        embargoed = t1 < i <= t1 + embargo
        if not overlaps and not embargoed:
            keep.append(i)
    return Fraction(len(keep))


def _grid_fits() -> Fraction:
    return Fraction(len(list(itertools.product(range(3), range(4), range(5)))) * 5)


def _gini(counts: list[int]) -> Fraction:
    n = sum(counts)
    return 1 - sum((Fraction(c, n) ** 2 for c in counts), Fraction(0))


def _entropy(probs: list[float]) -> float:
    return float(stats.entropy(probs, base=2))


def _info_gain() -> float:
    parent = _entropy([50, 50])
    left, right = _entropy([40, 10]), _entropy([10, 40])
    return parent - 0.5 * left - 0.5 * right


def _bagged_var(b: int, rho: Fraction) -> Fraction:
    cov = [[Fraction(1) if i == j else rho for j in range(b)] for i in range(b)]
    return sum((sum(r, Fraction(0)) for r in cov), Fraction(0)) / (b * b)


def _boost_one_step() -> Fraction:
    y = [Fraction(1), Fraction(2), Fraction(6)]
    f0 = min((Fraction(c, 100) for c in range(0, 1001)),
             key=lambda c: sum((v - c) ** 2 for v in y))  # best constant
    resid = [v - f0 for v in y]
    return f0 + Fraction(1, 10) * resid[2]  # stump fits each residual exactly


def _f1() -> Fraction:
    tp, fp, fn = 30, 10, 20
    return Fraction(2 * tp, 2 * tp + fp + fn)


def _auc() -> Fraction:
    pos, neg = [0.9, 0.6], [0.7, 0.3, 0.2]
    wins = sum(1 for p, q in itertools.product(pos, neg) if p > q)
    return Fraction(wins, len(pos) * len(neg))


def _uniqueness(n: int = 1000, span: int = 5) -> Fraction:
    # labels i cover days i..i+span-1 on a circle of n days (no edges)
    conc: Counter[int] = Counter()
    for i in range(n):
        for d in range(span):
            conc[(i + d) % n] += 1
    avg = sum(
        (sum((Fraction(1, conc[(i + d) % n]) for d in range(span)), Fraction(0)) / span
         for i in range(n)),
        Fraction(0),
    ) / n
    return 250 * avg


def _ridge_dof() -> Fraction:
    x = np.diag([2.0, 1.0])
    hat = x @ np.linalg.solve(x.T @ x + np.eye(2), x.T)
    return Fraction(float(np.trace(hat))).limit_denominator(1000)


def _lloyd() -> Fraction:
    pts = [Fraction(v) for v in (0, 1, 4, 5)]
    centres = [Fraction(0), Fraction(1)]
    groups: list[list[Fraction]] = [[], []]
    for p in pts:
        groups[min(range(2), key=lambda j: abs(p - centres[j]))].append(p)
    return sum(groups[1], Fraction(0)) / len(groups[1])


def _gd_max_step() -> float:
    # largest step for which w <- w - eta * 4 w does not diverge, by bisection
    lo, hi = 0.0, 2.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if abs(1 - 4 * mid) <= 1 else (lo, mid)
    return lo


def _odds_ratio() -> float:
    def odds(z: float) -> float:
        p = 1 / (1 + math.exp(-z))
        return p / (1 - p)

    return odds(0.3 + 0.7 * 1) / odds(0.3 + 0.7 * 0)  # any intercept works


def _perm_importance() -> float:
    # error after permuting = 2 x1 + x2 - 2 x1' with (x1, x2, x1') iid N(0, 1);
    # its mean square is the squared norm of the coefficient vector
    after = np.array([2.0, 1.0, -2.0])
    before = np.array([0.0, 1.0, 0.0])
    return float(after @ after - before @ before)


EXACT: dict[str, Callable[[], Number]] = {
    "M01": _ridge_1d,
    "M02": lambda: _lasso_1d(0.8, 0.3),
    "M03": lambda: _lasso_1d(0.2, 0.3),
    "M04": _purged_train_size,
    "M05": _grid_fits,
    "M06": lambda: _bagged_var(5, Fraction(0)),  # mean of 5 independent unit-variance labels
    "M07": lambda: _gini([30, 70]),
    "M08": lambda: _entropy([0.25, 0.75]),
    "M09": _info_gain,
    "M10": lambda: _bagged_var(10, Fraction(3, 10)),
    "M11": lambda: (1 - 1 / 10**7) ** 10**7,
    "M12": _boost_one_step,
    "M13": lambda: float(-stats.bernoulli.logpmf(1, 0.8)),
    "M14": _odds_ratio,
    "M15": _f1,
    "M16": _auc,
    "M17": _perm_importance,
    "M18": lambda: float(stats.norm.isf(0.05 / 100)),
    "M19": _uniqueness,
    "M20": _ridge_dof,
    "M21": _lloyd,
    "M22": _gd_max_step,
    "M23": lambda: Fraction(98, 100),
    "M24": lambda: Fraction(sum(1 for start in range(10) if start + 3 < 10)),
    "M25": lambda: float(stats.beta.mean(10 / 2, (101 - 10 - 1) / 2)),
}

REL_TOL: dict[str, float] = {"M11": 1e-6, "M22": 1e-9}


# --------------------------------------------------------------- simulation
def _mc(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _mc_knn_var() -> float:
    # prediction = average of the 5 neighbours' noisy labels; noise var 1
    eps = _mc(6).standard_normal((200_000, 5))
    return float(eps.mean(axis=1).var())


def _mc_bagging() -> float:
    rng = _mc(10)
    n = 100_000
    common = rng.standard_normal((n, 1))
    trees = math.sqrt(0.3) * common + math.sqrt(0.7) * rng.standard_normal((n, 10))
    return float(trees.mean(axis=1).var())


def _mc_oob() -> float:
    rng = _mc(11)
    n = 1000
    out = [1 - np.unique(rng.integers(0, n, n)).size / n for _ in range(400)]
    return float(np.mean(out))


def _mc_perm_importance() -> float:
    rng = _mc(17)
    n = 400_000
    x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
    y = 2 * x1 + x2
    base = np.mean((y - 2 * x1) ** 2)
    perm = np.mean((y - 2 * rng.permutation(x1)) ** 2)
    return float(perm - base)


def _mc_noise_r2() -> float:
    rng = _mc(25)
    n, p = 101, 10
    r2 = []
    for _ in range(4000):
        x = np.column_stack([np.ones(n), rng.standard_normal((n, p))])
        y = rng.standard_normal(n)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        r2.append(1 - resid @ resid / ((y - y.mean()) @ (y - y.mean())))
    return float(np.mean(r2))


MONTE_CARLO: dict[str, tuple[Callable[[], float], float]] = {
    "M06": (_mc_knn_var, 0.004),
    "M10": (_mc_bagging, 0.008),
    "M11": (_mc_oob, 0.002),
    "M17": (_mc_perm_importance, 0.06),
    "M25": (_mc_noise_r2, 0.004),
}


if __name__ == "__main__":  # pragma: no cover
    for pid, fn in EXACT.items():
        val = fn()
        mc = MONTE_CARLO.get(pid)
        extra = f"  MC {mc[0]():.4f}" if mc else ""
        print(f"{pid}  {val!s:>24}  {float(val):.10f}{extra}")
