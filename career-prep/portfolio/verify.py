"""Independent checks for the portfolio pack.

The key uses closed forms; this file reaches each value by numerical
optimisation, a matrix computation or a root-find instead, so a slip in a
formula shows up as a mismatch. ``career-prep/tests/test_numeric_packs.py``
compares the two.

Run ``python career-prep/portfolio/verify.py`` to print the table.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
from scipy import optimize

Number = Fraction | float

VOL = np.array([0.10, 0.20])
RHO = 0.2
COV = np.array([[VOL[0] ** 2, RHO * VOL[0] * VOL[1]], [RHO * VOL[0] * VOL[1], VOL[1] ** 2]])
MU = np.array([0.08, 0.12])
RF = 0.02


def _minvar_w1() -> float:
    res = optimize.minimize_scalar(lambda w: np.array([w, 1 - w]) @ COV @ np.array([w, 1 - w]),
                                   bounds=(-2, 3), method="bounded", options={"xatol": 1e-12})
    return float(res.x)


def _minvar_vol() -> float:
    w = np.array([_minvar_w1(), 1 - _minvar_w1()])
    return float(math.sqrt(w @ COV @ w))


def _neg_sharpe(w1: float) -> float:
    w = np.array([w1, 1 - w1])
    return -float((w @ MU - RF) / math.sqrt(w @ COV @ w))


def _tangency_w1() -> float:
    return float(optimize.minimize_scalar(_neg_sharpe, bounds=(-2, 3), method="bounded",
                                          options={"xatol": 1e-12}).x)


def _equal_weight_vol(n: int) -> float:
    c = np.full((n, n), 0.3 * 0.04)
    np.fill_diagonal(c, 0.04)
    w = np.full(n, 1 / n)
    return float(math.sqrt(w @ c @ w))


def _kelly_cont() -> float:
    # maximise f mu - f^2 sigma^2 / 2 numerically
    return float(optimize.minimize_scalar(lambda f: -(f * 0.06 - 0.5 * f * f * 0.04),
                                          bounds=(0, 10), method="bounded",
                                          options={"xatol": 1e-12}).x)


def _kelly_bet() -> float:
    return float(optimize.minimize_scalar(
        lambda f: -(0.6 * math.log(1 + f) + 0.4 * math.log(1 - f)),
        bounds=(0, 0.99), method="bounded", options={"xatol": 1e-12}).x)


def _risk_contrib(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return w * (cov @ w) / math.sqrt(w @ cov @ w)


def _rp2() -> float:
    def gap(w1: float) -> float:
        rc = _risk_contrib(np.array([w1, 1 - w1]), COV)
        return float(rc[0] - rc[1])

    return float(optimize.brentq(gap, 0.01, 0.99, xtol=1e-14))


COV3 = (np.array([0.10, 0.20, 0.30])[:, None] * np.array([0.10, 0.20, 0.30])[None, :]
        * np.array([[1, 0.5, 0.2], [0.5, 1, 0.4], [0.2, 0.4, 1]]))


def _rp3_w1() -> float:
    def eqs(x: np.ndarray) -> np.ndarray:
        w = np.append(x, 1 - x.sum())
        rc = _risk_contrib(w, COV3)
        return np.array([rc[0] - rc[1], rc[1] - rc[2]])

    sol = optimize.fsolve(eqs, np.array([0.5, 0.3]), xtol=1e-14)
    return float(sol[0])


def _te() -> float:
    a = np.array([0.1, -0.1])
    return float(math.sqrt(a @ COV @ a))


def _div_ratio() -> float:
    w = np.array([_minvar_w1(), 1 - _minvar_w1()])
    return float(w @ VOL / math.sqrt(w @ COV @ w))


EXACT: dict[str, Callable[[], Number]] = {
    "O01": _minvar_w1,
    "O02": _minvar_vol,
    "O03": _tangency_w1,
    "O04": lambda: -_neg_sharpe(_tangency_w1()),
    "O05": lambda: _equal_weight_vol(10),
    "O06": lambda: _equal_weight_vol(20_000),
    "O07": _kelly_cont,
    "O08": _kelly_bet,
    "O09": _rp2,
    "O10": _rp3_w1,
    "O11": _te,
    "O12": lambda: Fraction(15, 1000) / Fraction(3, 100),
    "O13": _div_ratio,
    "O14": lambda: 1 / sum(Fraction(x, 10) ** 2 for x in (5, 3, 2)),
    "O15": lambda: 0.01 * 12 / (0.04 * math.sqrt(12)),
}

# Looser tolerance for values reached by an optimiser or a large-n limit.
TOLERANCE: dict[str, float] = {"O06": 1e-4}

MONTE_CARLO: dict[str, tuple[Callable[[], float], float]] = {}


if __name__ == "__main__":  # pragma: no cover
    for pid, fn in EXACT.items():
        print(f"{pid}  {float(fn()):.10g}")
