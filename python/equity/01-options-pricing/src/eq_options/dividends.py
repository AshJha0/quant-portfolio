"""Discrete cash dividends: escrowed model, Roll-Geske-Whaley, and an American tree.

Plain idea: a known cash dividend makes the stock drop by that amount on the
ex-date. The escrowed-dividend model splits the spot into the present value
of the dividends paid before expiry (riskless) and the rest (lognormal).
Europeans are then Black-Scholes on the reduced spot. For American calls,
early exercise only pays just before an ex-date; with one dividend that
gives the Roll-Geske-Whaley closed form. For several dividends, or puts, a
tree on the risky part, with the dividends' present value added back at
each node for the exercise test, prices the same model.

Formal statement.

* Escrowed spot: ``S* = S - sum_{t_i <= T} D_i exp(-r t_i)``; European
  value ``bs(S*, K, T, r, sigma, 0)`` (Haug 2007, "The Complete Guide to
  Option Pricing Formulas", 2nd ed., section 9.1).
* Roll (1977), Geske (1979), Whaley (1981); formula as in Haug (2007),
  section 9.2.1, with the critical ex-dividend price ``I`` solving
  ``c(I, K, T - t1) = I + D - K``. Haug's worked example
  (``S = 80, K = 82, t1 = 0.25, T = 1/3, D = 4, r = 0.06, sigma = 0.30``)
  gives 4.3860, which the tests reproduce.
* Tree: Leisen-Reimer probabilities and moves on ``S*``; node spot for the
  exercise test is ``S*_node + PV_t(remaining dividends)``.

Conventions: ``r`` continuously compounded, dividends in currency units at
times in years; no continuous yield.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import multivariate_normal

from .black_scholes import OptionType, _check_rates, bs_price, validate_inputs
from .lr_tree import lr_parameters

__all__ = ["bs_price_discrete_divs", "escrowed_spot", "lr_price_discrete_divs", "rgw_call"]


def _divs_before(dividends: Sequence[tuple[float, float]], T: float):
    out = []
    for t, d in dividends:
        if not (math.isfinite(t) and math.isfinite(d)) or t < 0.0 or d < 0.0:
            raise ValueError(f"dividend (t={t!r}, D={d!r}) must be finite and >= 0")
        if t <= T:
            out.append((float(t), float(d)))
    return sorted(out)


def escrowed_spot(S: float, T: float, r: float, dividends: Sequence[tuple[float, float]]) -> float:
    """``S`` less the present value of dividends paid at or before ``T``."""
    s_star = S - sum(d * math.exp(-r * t) for t, d in _divs_before(dividends, T))
    if s_star <= 0.0:
        raise ValueError("dividends' present value exceeds the spot")
    return s_star


def bs_price_discrete_divs(
    S: float, K: float, T: float, r: float, sigma: float,
    dividends: Sequence[tuple[float, float]], option_type: OptionType = "call",
) -> float:
    """European price under the escrowed-dividend model."""
    validate_inputs(S, K, T, sigma, option_type)
    _check_rates(r=r)
    return bs_price(escrowed_spot(S, T, r, dividends), K, T, r, sigma, 0.0, option_type)


def _bvn(a: float, b: float, rho: float) -> float:
    return float(multivariate_normal.cdf([a, b], mean=[0.0, 0.0],
                                         cov=[[1.0, rho], [rho, 1.0]], abseps=1e-12, releps=1e-12))


def rgw_call(S: float, K: float, t1: float, T: float, D: float, r: float, sigma: float) -> float:
    """Roll-Geske-Whaley American call with one cash dividend ``D`` at ``t1 < T``."""
    validate_inputs(S, K, T, sigma, "call")
    _check_rates(r=r)
    if not 0.0 < t1 < T or D < 0.0 or min(S, K, sigma) <= 0.0:
        raise ValueError("need 0 < t1 < T, D >= 0 and S, K, sigma > 0")
    s_star = S - D * math.exp(-r * t1)
    if s_star <= 0.0:
        raise ValueError("dividend present value exceeds the spot")
    if D <= K * (1.0 - math.exp(-r * (T - t1))):
        return bs_price(s_star, K, T, r, sigma, 0.0, "call")  # never exercised early
    tau = T - t1

    def f(i: float) -> float:
        return bs_price(i, K, tau, r, sigma, 0.0, "call") - i - D + K

    hi = K
    while f(hi) > 0.0:
        hi *= 2.0
    crit = brentq(f, 1e-12 * K, hi, xtol=1e-14 * K, rtol=1e-15)
    st, st1 = sigma * math.sqrt(T), sigma * math.sqrt(t1)
    a1 = (math.log(s_star / K) + (r + 0.5 * sigma**2) * T) / st
    a2 = a1 - st
    b1 = (math.log(s_star / crit) + (r + 0.5 * sigma**2) * t1) / st1
    b2 = b1 - st1
    rho = -math.sqrt(t1 / T)
    return (s_star * ndtr(b1) + s_star * _bvn(a1, -b1, rho)
            - K * math.exp(-r * T) * _bvn(a2, -b2, rho)
            - (K - D) * math.exp(-r * t1) * ndtr(b2))


def lr_price_discrete_divs(
    S: float, K: float, T: float, r: float, sigma: float,
    dividends: Sequence[tuple[float, float]], option_type: OptionType = "call",
    exercise: str = "american", n_steps: int = 501,
) -> float:
    """Leisen-Reimer tree under the escrowed-dividend model (American or European)."""
    validate_inputs(S, K, T, sigma, option_type)
    _check_rates(r=r)
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")
    divs = _divs_before(dividends, T)
    s_star = escrowed_spot(S, T, r, divs)
    n, u, d, p = lr_parameters(s_star, K, T, r, sigma, 0.0, n_steps)
    dt = T / n
    disc = math.exp(-r * dt)
    sign = 1.0 if option_type == "call" else -1.0
    log_u, log_d = math.log(u), math.log(d)

    def pv_remaining(t: float) -> float:
        return sum(dv * math.exp(-r * (ti - t)) for ti, dv in divs if ti > t)

    j = np.arange(n + 1)
    values = np.maximum(sign * (s_star * np.exp(j * log_u + (n - j) * log_d) - K), 0.0)
    for i in range(n - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if exercise == "american":
            k = np.arange(i + 1)
            spots = s_star * np.exp(k * log_u + (i - k) * log_d) + pv_remaining(i * dt)
            np.maximum(values, sign * (spots - K), out=values)
    return float(values[0])
