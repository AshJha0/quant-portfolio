"""Independent checks for the risk pack.

Each answer in ``keys/answers.md`` is recomputed here by a route that differs
from the written solution: a root-find on a distribution function instead of
a quoted quantile, a numerical integral instead of the closed-form expected
shortfall, a finite-difference bump instead of an analytic duration, exact
enumeration, or a seeded simulation. ``career-prep/tests/test_numeric_packs.py``
compares the key with ``EXACT`` and, where present, ``MONTE_CARLO``.

Run ``python career-prep/risk/verify.py`` to print the table.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
from scipy import integrate, optimize
from scipy.stats import binom, norm

Number = Fraction | float

NOTIONAL = 10_000_000.0
SIG = 0.01


def _q(alpha: float) -> float:
    """Standard normal quantile by root-finding on the CDF (not norm.ppf)."""
    return float(optimize.brentq(lambda x: norm.cdf(x) - alpha, -10, 10, xtol=1e-14))


def _es_integral(alpha: float, sd: float) -> float:
    """ES as the average of VaR_u over u in (alpha, 1), integrated numerically."""
    val, _ = integrate.quad(lambda u: sd * _q(u), alpha, 1 - 1e-12, limit=200)
    return val / (1 - alpha)


def _cov2() -> np.ndarray:
    s = np.array([5e6 * 0.01, 5e6 * 0.02])
    rho = 0.3
    return np.array([[s[0] ** 2, rho * s[0] * s[1]], [rho * s[0] * s[1], s[1] ** 2]])


def _port_var(scale2: float = 1.0) -> float:
    w = np.array([1.0, scale2])
    return float(_q(0.99) * math.sqrt(w @ _cov2() @ w))


def _component2() -> float:
    h = 1e-6
    # Euler: component = x_2 * dVaR/dx_2, by central difference in the scale of position 2
    return (_port_var(1 + h) - _port_var(1 - h)) / (2 * h)


PNL7 = [-1.2, 0.4, -3.5, 2.1, 0.9, -0.3, -2.2, 1.5, -4.1, 0.2,
        0.7, -0.8, 1.1, -1.9, 0.0, 3.0, -0.6, -2.7, 0.5, 1.8]


def _es90_hist() -> float:
    losses = sorted((-p for p in PNL7), reverse=True)
    return sum(losses[:2]) / 2


def _kupiec() -> float:
    n, x, p = 250, 6, 0.01
    return float(-2 * (binom.logpmf(x, n, p) - binom.logpmf(x, n, x / n)))


def _yellow_start() -> int:
    return next(k for k in range(251) if binom.cdf(k, 250, 0.01) >= 0.95)


def _zero_price(y: float) -> float:
    return 100 / (1 + y) ** 10


def _dv01() -> float:
    h = 1e-6
    return (_zero_price(0.05 - h) - _zero_price(0.05 + h)) / (2 * h) * 1e-4


def _dur_conv_approx() -> float:
    h = 1e-5
    p0 = _zero_price(0.05)
    d1 = (_zero_price(0.05 + h) - _zero_price(0.05 - h)) / (2 * h) / p0
    d2 = (_zero_price(0.05 + h) - 2 * p0 + _zero_price(0.05 - h)) / h**2 / p0
    dy = 0.01
    return d1 * dy + 0.5 * d2 * dy**2


def _delta_gamma() -> Fraction:
    return 100 * (Fraction(6, 10) * 2 + Fraction(1, 2) * Fraction(5, 100) * 4)


def _lognormal_var() -> float:
    return 100 * (1 - math.exp(_q(0.01) * 0.02))


def _two_loans() -> Fraction:
    p = Fraction(5, 100)
    return sum((pa * pb for a, b in itertools.product([0, 1], repeat=2) if a or b
                for pa in [p if a else 1 - p] for pb in [p if b else 1 - p]), Fraction(0))


def _vasicek() -> float:
    # P(conditional default rate <= x) = 0.999 solved for x by root-finding
    pd, rho = 0.01, 0.2

    def cond(m: float) -> float:
        return float(norm.cdf((norm.ppf(pd) - math.sqrt(rho) * m) / math.sqrt(1 - rho)))

    m999 = _q(0.001)  # bad state of the systematic factor
    return cond(m999)


def _subadd() -> int:
    p = 0.04
    dist: dict[int, float] = {}
    for a, b in itertools.product([0, 1], repeat=2):
        loss = 100 * (a + b)
        dist[loss] = dist.get(loss, 0.0) + (p if a else 1 - p) * (p if b else 1 - p)
    cum = 0.0
    for loss in sorted(dist):
        cum += dist[loss]
        if cum >= 0.95:
            return loss
    raise AssertionError


def _ewma() -> float:
    var = 0.01**2
    var = 0.94 * var + 0.06 * 0.03**2
    return math.sqrt(var)


EXACT: dict[str, Callable[[], Number]] = {
    "K01": lambda: NOTIONAL * SIG * _q(0.99),
    "K02": lambda: NOTIONAL * _es_integral(0.975, SIG),
    "K03": lambda: NOTIONAL * SIG * math.sqrt(10) * _q(0.99),
    "K04": _port_var,
    "K05": lambda: _q(0.99) * (5e4 + 1e5) - _port_var(),
    "K06": _component2,
    "K07": _es90_hist,
    "K08": _kupiec,
    "K09": lambda: Fraction(_yellow_start()),
    "K10": _dv01,
    "K11": _dur_conv_approx,
    "K12": _delta_gamma,
    "K13": _lognormal_var,
    "K14": lambda: Fraction(2, 100) * Fraction(6, 10) * 1_000_000,
    "K15": _two_loans,
    "K16": _vasicek,
    "K17": lambda: _es_integral(0.99, 1.0) / _q(0.99),
    "K18": lambda: Fraction(_subadd()),
    "K19": _ewma,
    "K20": lambda: _q(0.975) / _q(0.99),
}

N_MC = 400_000


def _mc(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _mc_var(x: np.ndarray, alpha: float) -> float:
    return float(-np.quantile(x, 1 - alpha))


def _mc_es(x: np.ndarray, alpha: float) -> float:
    v = np.quantile(x, 1 - alpha)
    return float(-x[x <= v].mean())


def _mc_port() -> float:
    z = _mc(4).multivariate_normal([0, 0], _cov2(), N_MC)
    return _mc_var(z.sum(axis=1), 0.99)


def _mc_10day() -> float:
    x = _mc(3).standard_normal((N_MC, 10)).sum(axis=1) * SIG * NOTIONAL
    return _mc_var(x, 0.99)


def _mc_vasicek() -> float:
    rng = _mc(16)
    m = rng.standard_normal(N_MC)
    rate = norm.cdf((norm.ppf(0.01) - math.sqrt(0.2) * m) / math.sqrt(0.8))
    return float(np.quantile(rate, 0.999))


MONTE_CARLO: dict[str, tuple[Callable[[], float], float]] = {
    "K01": (lambda: _mc_var(_mc(1).standard_normal(N_MC) * SIG * NOTIONAL, 0.99), 3000.0),
    "K02": (lambda: _mc_es(_mc(2).standard_normal(N_MC) * SIG * NOTIONAL, 0.975), 3000.0),
    "K03": (_mc_10day, 10000.0),
    "K04": (_mc_port, 4000.0),
    "K13": (lambda: _mc_var(100 * (np.exp(0.02 * _mc(13).standard_normal(N_MC)) - 1), 0.99),
            0.06),
    "K16": (_mc_vasicek, 0.004),
}


if __name__ == "__main__":  # pragma: no cover
    for pid, fn in EXACT.items():
        val = fn()
        mc = MONTE_CARLO.get(pid)
        extra = f"  MC {mc[0]():.6g}" if mc else ""
        print(f"{pid}  {float(val):.8g}{extra}")
