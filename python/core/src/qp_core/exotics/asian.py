"""Arithmetic Asian options: Turnbull-Wakeman and control-variate Monte Carlo.

Plain idea
----------
The average of log-normal prices is not log-normal, so the arithmetic
Asian has no closed form. Turnbull and Wakeman match the first two moments
of the average to a log-normal and price that with Black's formula. The
*geometric* average is log-normal, so its Asian has an exact price; it
moves almost one-for-one with the arithmetic average, which makes it an
excellent control variate for Monte Carlo.

Formal statement
----------------
Fixings at ``t_1..t_n``, average ``A = (1/n) sum S_{t_i}``, cost of carry
``b``. The moments are exact:

    M1 = E[A]   = (S/n) sum_i e^{b t_i}
    M2 = E[A^2] = (S^2/n^2) sum_{i,j} e^{b (t_i + t_j) + sigma^2 min(t_i, t_j)}

Turnbull and Wakeman (1991), "A quick algorithm for pricing European
average options", JFQA 26(3): treat ``A`` as log-normal with these moments,
``v^2 = ln(M2 / M1^2)``, and price ``e^{-rT} (M1 N(d1) - X N(d2))`` with
``d1 = (ln(M1/X) + v^2/2)/v``. The continuous-averaging version is Haug
(2007) s.4.20.2. The geometric Asian with discrete fixings (Kemna and Vorst
1990, J. Banking and Finance 14(1), discrete form) is exact: ``ln G`` is
normal with mean ``ln S + (b - sigma^2/2) mean(t)`` and variance
``sigma^2 / n^2 sum_{i,j} min(t_i, t_j)``.

Worked example
--------------
>>> t = [i / 12 for i in range(1, 13)]
>>> g = geometric_asian_price("call", 100, 100, t, 1.0, 0.05, 0.05, 0.2)
>>> a = turnbull_wakeman_price("call", 100, 100, t, 1.0, 0.05, 0.05, 0.2)
>>> g < a   # AM-GM: the arithmetic average dominates
True
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..normal import norm_cdf as N

__all__ = [
    "asian_mc",
    "geometric_asian_price",
    "turnbull_wakeman_continuous",
    "turnbull_wakeman_price",
]


def _check(cp: str, S: float, X: float, fixings: Sequence[float], T: float, sigma: float) -> np.ndarray:
    if cp not in ("call", "put"):
        raise ValueError(f"cp must be 'call' or 'put', got {cp!r}")
    t = np.asarray(fixings, dtype=float)
    if t.ndim != 1 or t.size == 0 or np.any(t <= 0.0) or np.any(np.diff(t) <= 0.0) or t[-1] > T + 1e-12:
        raise ValueError("fixings must be increasing, positive and not after T")
    if not (S > 0 and X > 0 and sigma > 0 and T > 0):
        raise ValueError("S, X, T and sigma must be positive")
    return t


def _black(cp: str, F: float, X: float, v: float, df: float) -> float:
    d1 = (math.log(F / X) + 0.5 * v * v) / v
    d2 = d1 - v
    if cp == "call":
        return df * (F * N(d1) - X * N(d2))
    return df * (X * N(-d2) - F * N(-d1))


def geometric_asian_price(
    cp: str, S: float, X: float, fixings: Sequence[float], T: float, r: float, b: float, sigma: float
) -> float:
    """Exact price of the discretely monitored geometric-average Asian."""
    t = _check(cp, S, X, fixings, T, sigma)
    n = t.size
    mean = math.log(S) + (b - 0.5 * sigma * sigma) * float(t.mean())
    var = sigma * sigma * float(np.minimum.outer(t, t).sum()) / (n * n)
    F = math.exp(mean + 0.5 * var)
    return _black(cp, F, X, math.sqrt(var), math.exp(-r * T))


def turnbull_wakeman_price(
    cp: str, S: float, X: float, fixings: Sequence[float], T: float, r: float, b: float, sigma: float
) -> float:
    """Turnbull-Wakeman log-normal moment match with exact discrete moments."""
    t = _check(cp, S, X, fixings, T, sigma)
    n = t.size
    e = np.exp(b * t)
    m1 = S * float(e.sum()) / n
    m2 = S * S * float((np.outer(e, e) * np.exp(sigma * sigma * np.minimum.outer(t, t))).sum()) / (n * n)
    v = math.sqrt(math.log(m2 / (m1 * m1)))
    return _black(cp, m1, X, v, math.exp(-r * T))


def turnbull_wakeman_continuous(
    cp: str, S: float, X: float, T: float, r: float, b: float, sigma: float, tau: float = 0.0
) -> float:
    """Continuous-averaging Turnbull-Wakeman (Haug 2007 s.4.20.2), averaging over ``[tau, T]``.

    Uses the closed-form moments; the ``b -> 0`` limits are handled by
    switching to the Taylor-exact forms when ``|b| < 1e-8``.
    """
    if not (S > 0 and X > 0 and sigma > 0 and T > tau >= 0.0):
        raise ValueError("need S, X, sigma > 0 and T > tau >= 0")
    if cp not in ("call", "put"):
        raise ValueError(f"cp must be 'call' or 'put', got {cp!r}")
    s2 = sigma * sigma
    L = T - tau
    if abs(b) < 1e-8:
        m1 = 1.0
        m2 = 2.0 * math.exp(s2 * T) / (s2 * s2 * L * L) - 2.0 * math.exp(s2 * tau) * (1.0 + s2 * L) / (s2 * s2 * L * L)
    else:
        m1 = (math.exp(b * T) - math.exp(b * tau)) / (b * L)
        m2 = (2.0 * math.exp((2.0 * b + s2) * T) / ((b + s2) * (2.0 * b + s2) * L * L)
              + 2.0 * math.exp((2.0 * b + s2) * tau) / (b * L * L)
              * (1.0 / (2.0 * b + s2) - math.exp(b * L) / (b + s2)))
    v = math.sqrt(math.log(m2 / (m1 * m1)))
    return _black(cp, S * m1, X, v, math.exp(-r * T))


def asian_mc(
    cp: str, S: float, X: float, fixings: Sequence[float], T: float, r: float, b: float, sigma: float,
    n_paths: int = 100_000, seed: int = 0, control_variate: bool = True,
) -> tuple[float, float]:
    """Arithmetic Asian by exact-step MC; geometric Asian as control variate.

    Returns ``(estimate, standard_error)``. The control-variate coefficient
    is estimated from the same paths (the usual regression estimator; its
    bias is O(1/n_paths), Glasserman 2004 s.4.1).
    """
    t = _check(cp, S, X, fixings, T, sigma)
    rng = np.random.default_rng(seed)
    dt = np.diff(np.concatenate([[0.0], t]))
    z = rng.standard_normal((n_paths, t.size))
    logs = math.log(S) + np.cumsum((b - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z, axis=1)
    arith = np.exp(logs).mean(axis=1)
    geo = np.exp(logs.mean(axis=1))
    df = math.exp(-r * T)
    sign = 1.0 if cp == "call" else -1.0
    y = df * np.maximum(sign * (arith - X), 0.0)
    if not control_variate:
        return float(y.mean()), float(y.std(ddof=1) / math.sqrt(n_paths))
    g = df * np.maximum(sign * (geo - X), 0.0)
    g_exact = geometric_asian_price(cp, S, X, fixings, T, r, b, sigma)
    c = np.cov(y, g, ddof=1)
    beta = c[0, 1] / c[1, 1] if c[1, 1] > 0 else 0.0
    adj = y - beta * (g - g_exact)
    return float(adj.mean()), float(adj.std(ddof=1) / math.sqrt(n_paths))
