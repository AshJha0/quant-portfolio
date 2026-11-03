"""Independent checks for the case pack.

The key uses closed forms (N(d1), put-call parity, the Kelly formula, Bayes,
a sum of squares, the two-asset variance). Here: the hedge ratio is a
Richardson finite difference of a call price integrated against the
lognormal density; the arbitrage is the present value of a replicated
payoff checked to be constant over a grid of terminal prices; the Kelly
stake maximises expected log growth numerically; the market-maker ask is
found by enumerating trader types; the impact cost is a slice-by-slice sum;
and the VaR is a seeded simulation plus a covariance-matrix quantile.

Run ``python career-prep/case/verify.py`` to print the table.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import norm

Number = Fraction | float
R, SIG, T = 0.05, 0.2, 1.0


def _call_quad(s0: float, k: float = 100.0) -> float:
    m = math.log(s0) + (R - 0.5 * SIG * SIG) * T
    sd = SIG * math.sqrt(T)

    def f(x: float) -> float:
        return (math.exp(x) - k) * math.exp(-0.5 * ((x - m) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))

    v, _ = quad(f, math.log(k), m + 12 * sd, epsabs=1e-13, epsrel=1e-13)
    return math.exp(-R * T) * v


def _hedge() -> float:
    h = 0.5
    d1 = (_call_quad(100 + h) - _call_quad(100 - h)) / (2 * h)
    d2 = (_call_quad(100 + h / 2) - _call_quad(100 - h / 2)) / h
    return 1000 * (4 * d2 - d1) / 3


def _arbitrage() -> float:
    c, p, s, k = 10.45, 5.00, 100.0, 100.0
    today = c - p - s + k * math.exp(-R * T)  # sell call, buy put, buy stock, borrow PV(K)
    terminal = [-max(st - k, 0) + max(k - st, 0) + st - k for st in np.linspace(1, 300, 600)]
    assert max(abs(x) for x in terminal) < 1e-9
    return today


def _kelly() -> float:
    res = minimize_scalar(
        lambda f: -(0.55 * math.log(1 + f) + 0.45 * math.log(1 - f)),
        bounds=(0.0, 0.9),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return float(res.x)


def _gm_ask() -> Fraction:
    num = den = Fraction(0)
    for value in (99, 101):
        for informed in (True, False):
            weight = Fraction(1, 2) * (Fraction(1, 5) if informed else Fraction(4, 5))
            p_buy = Fraction(int(value == 101)) if informed else Fraction(1, 2)
            num += weight * p_buy * value
            den += weight * p_buy
    return num / den


def _impact() -> Fraction:
    total = Fraction(0)
    remaining = 100_000
    for _slice in range(10):
        n = 10_000
        remaining -= n
        total += n * Fraction(n, 10**6)  # each of the n shares pays 10^-6 n
    assert remaining == 0
    return total


def _var_matrix() -> float:
    cov = np.array([[1.0, 0.3], [0.3, 1.0]]) * 20_000.0**2
    w = np.ones(2)
    return float(norm.ppf(0.99) * math.sqrt(w @ cov @ w))


def _mc_var() -> float:
    rng = np.random.default_rng(15)
    cov = np.array([[1.0, 0.3], [0.3, 1.0]]) * 20_000.0**2
    pnl = rng.multivariate_normal([0.0, 0.0], cov, size=400_000).sum(axis=1)
    return float(-np.quantile(pnl, 0.01))


EXACT: dict[str, Callable[[], Number]] = {
    "C01": _hedge,
    "C02": _arbitrage,
    "C03": _kelly,
    "C04": _gm_ask,
    "C05": _impact,
    "C06": _var_matrix,
}

TOL: dict[str, float] = {"C01": 1e-6, "C03": 1e-8}

MONTE_CARLO: dict[str, tuple[Callable[[], float], float]] = {"C06": (_mc_var, 800.0)}


if __name__ == "__main__":
    for pid, fn in EXACT.items():
        print(pid, fn())
