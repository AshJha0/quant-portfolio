"""Caps, floors and European swaptions: Black-76 and Bachelier.

Market quoting conventions (Hull 2018, "Options, Futures and Other
Derivatives", 10th ed., ch. 29, eqs. 29.7-29.8 for caplets and 29.10-29.11
for swaptions; Bachelier form in Brigo and Mercurio 2006, "Interest Rate
Models - Theory and Practice", 2nd ed., Section 6.4):

    caplet  = tau * DF(T_pay) * Black(F, K, sigma, T_fix)
    payer   = A * Black(F_swap, K, sigma, T_exp),  A = sum tau_i DF(T_i)

``Black`` is the undiscounted Black-76 call on a lognormal forward;
``Bachelier`` the same on a normal forward. Floors and receivers use the
put. Cap = sum of caplets. With discount and projection on separate curves
(see ``multicurve``), ``F`` comes from the projection curve and ``DF`` and
``A`` from the discount curve.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .curve import DiscountCurve


def black76(F: float, K: float, T: float, sigma: float, call: bool = True) -> float:
    """Undiscounted Black-76 value."""
    if F <= 0.0 or K <= 0.0:
        raise ValueError("Black-76 needs F > 0 and K > 0; use Bachelier for negative rates")
    if T <= 0.0 or sigma <= 0.0:
        return max(F - K, 0.0) if call else max(K - F, 0.0)
    s = sigma * np.sqrt(T)
    d1 = np.log(F / K) / s + 0.5 * s
    d2 = d1 - s
    if call:
        return float(F * norm.cdf(d1) - K * norm.cdf(d2))
    return float(K * norm.cdf(-d2) - F * norm.cdf(-d1))


def bachelier(F: float, K: float, T: float, sigma_n: float, call: bool = True) -> float:
    """Undiscounted Bachelier (normal) value."""
    if T <= 0.0 or sigma_n <= 0.0:
        return max(F - K, 0.0) if call else max(K - F, 0.0)
    s = sigma_n * np.sqrt(T)
    d = (F - K) / s
    w = 1.0 if call else -1.0
    return float(w * (F - K) * norm.cdf(w * d) + s * norm.pdf(d))


def implied_vol(price: float, F: float, K: float, T: float, call: bool = True,
                model: str = "black") -> float:
    """Invert Black-76 (``model='black'``) or Bachelier (``'normal'``) for vol."""
    fn = black76 if model == "black" else bachelier
    if model not in ("black", "normal"):
        raise ValueError(f"model must be 'black' or 'normal', got {model!r}")
    intrinsic = max(F - K, 0.0) if call else max(K - F, 0.0)
    if price <= intrinsic:
        raise ValueError("price at or below intrinsic: no implied vol")
    hi = 5.0 if model == "black" else 1.0
    while fn(F, K, T, hi, call) < price:
        hi *= 2.0
    return float(brentq(lambda s: fn(F, K, T, s, call) - price, 1e-12, hi, xtol=1e-14))


def _cap_schedule(maturity: float, frequency: int) -> np.ndarray:
    n = maturity * frequency
    if abs(n - round(n)) > 1e-9:
        raise ValueError("maturity * frequency must be integral")
    return np.arange(0, int(round(n)) + 1, dtype=float) / frequency


def cap_price(
    discount: DiscountCurve,
    projection: DiscountCurve,
    strike: float,
    maturity: float,
    vol: float,
    frequency: int = 4,
    model: str = "black",
    floor: bool = False,
) -> float:
    """Cap (or floor) of unit notional; the first period (fixed today) is skipped."""
    t = _cap_schedule(maturity, frequency)
    tau = 1.0 / frequency
    fn = black76 if model == "black" else bachelier
    total = 0.0
    for t0, t1 in zip(t[1:-1], t[2:], strict=False):
        F = projection.simple_forward_rate(t0, t1)
        total += tau * float(discount.df(t1)) * fn(F, strike, t0, vol, call=not floor)
    return total


def swap_annuity(discount: DiscountCurve, expiry: float, tenor: float,
                 frequency: int = 1) -> tuple[np.ndarray, float]:
    n = tenor * frequency
    if abs(n - round(n)) > 1e-9:
        raise ValueError("tenor * frequency must be integral")
    t = expiry + np.arange(1, int(round(n)) + 1, dtype=float) / frequency
    return t, float(np.sum(np.asarray(discount.df(t)))) / frequency


def forward_swap_rate(discount: DiscountCurve, projection: DiscountCurve, expiry: float,
                      tenor: float, fixed_frequency: int = 1,
                      float_frequency: int = 4) -> float:
    """Forward par rate of the swap starting at ``expiry`` (multi-curve)."""
    _, A = swap_annuity(discount, expiry, tenor, fixed_frequency)
    tf = expiry + np.arange(0, int(round(tenor * float_frequency)) + 1) / float_frequency
    p = np.asarray(projection.df(tf))
    d = np.asarray(discount.df(tf[1:]))
    return float(np.sum(d * (p[:-1] / p[1:] - 1.0)) / A)


def swaption_price(
    discount: DiscountCurve,
    projection: DiscountCurve,
    expiry: float,
    tenor: float,
    strike: float,
    vol: float,
    payer: bool = True,
    model: str = "black",
    fixed_frequency: int = 1,
    float_frequency: int = 4,
) -> float:
    """European swaption of unit notional, annuity-measure formula."""
    _, A = swap_annuity(discount, expiry, tenor, fixed_frequency)
    F = forward_swap_rate(discount, projection, expiry, tenor, fixed_frequency, float_frequency)
    fn = black76 if model == "black" else bachelier
    return A * fn(F, strike, expiry, vol, call=payer)
