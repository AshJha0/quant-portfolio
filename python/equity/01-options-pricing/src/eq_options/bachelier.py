"""Bachelier (normal) model: price, Greeks and implied normal volatility.

Plain idea: if the forward moves by normal, not lognormal, increments, the
option price is the expected positive part of a normal variable, which has
a short closed form. Rates, spreads and anything that can go negative are
often quoted as normal vols in this model.

Formal statement (Bachelier 1900; in the form of Schachermayer and Teichmann
2008, "How close are the option pricing formulas of Bachelier and Black-
Merton-Scholes?", Math. Finance 18(1), 155-170). Forward ``F``, normal vol
``sigma_n`` (price units per sqrt(year)), discount factor ``df = exp(-r T)``,
``d = (F - K) / (sigma_n sqrt T)``:

``call = df [(F - K) Phi(d) + sigma_n sqrt(T) phi(d)]``,
``put = df [(K - F) Phi(-d) + sigma_n sqrt(T) phi(d)]``.

Greeks (to ``F``): delta ``df Phi(d)`` (call) or ``-df Phi(-d)`` (put);
gamma ``df phi(d) / (sigma_n sqrt T)``; vega ``df sqrt(T) phi(d)`` per unit
of normal vol; theta ``r V - df sigma_n phi(d) / (2 sqrt T)`` per year
(``-dV/dT`` with ``F`` held).

Implied normal vol: safeguarded Newton, bracketed by bisection, to a
relative tolerance of 1e-14. This is not Jaeckel's 2017 rational
approximation; it is slower but needs no tables.

Worked example: ``F = K = 100, T = 1, sigma_n = 20, r = 0``: both options
are worth ``20 / sqrt(2 pi) = 7.978846``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.special import ndtr

from .black_scholes import OptionType

__all__ = ["BachelierGreeks", "bachelier_greeks", "bachelier_implied_vol", "bachelier_price"]

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _check(F: float, K: float, T: float, sigma_n: float, r: float, option_type: str) -> None:
    for name, v in (("F", F), ("K", K), ("T", T), ("sigma_n", sigma_n), ("r", r)):
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")
    if T < 0.0 or sigma_n < 0.0:
        raise ValueError("T and sigma_n must be >= 0")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bachelier_price(F: float, K: float, T: float, sigma_n: float, r: float = 0.0,
                    option_type: OptionType = "call") -> float:
    """Bachelier price; ``F`` and ``K`` may be negative."""
    _check(F, K, T, sigma_n, r, option_type)
    df = math.exp(-r * T)
    sign = 1.0 if option_type == "call" else -1.0
    s = sigma_n * math.sqrt(T)
    if s == 0.0:
        return df * max(sign * (F - K), 0.0)
    d = (F - K) / s
    return df * (sign * (F - K) * float(ndtr(sign * d)) + s * _INV_SQRT_2PI * math.exp(-0.5 * d * d))


@dataclass(frozen=True)
class BachelierGreeks:
    """Bachelier price and Greeks (units in the module docstring)."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float


def bachelier_greeks(F: float, K: float, T: float, sigma_n: float, r: float = 0.0,
                     option_type: OptionType = "call") -> BachelierGreeks:
    """Analytic Bachelier Greeks; needs ``T > 0`` and ``sigma_n > 0``."""
    _check(F, K, T, sigma_n, r, option_type)
    if T <= 0.0 or sigma_n <= 0.0:
        raise ValueError("Greeks need T > 0 and sigma_n > 0")
    df = math.exp(-r * T)
    sq = math.sqrt(T)
    s = sigma_n * sq
    d = (F - K) / s
    pdf = _INV_SQRT_2PI * math.exp(-0.5 * d * d)
    price = bachelier_price(F, K, T, sigma_n, r, option_type)
    delta = df * float(ndtr(d)) if option_type == "call" else -df * float(ndtr(-d))
    return BachelierGreeks(
        price=price, delta=delta, gamma=df * pdf / s, vega=df * sq * pdf,
        theta=r * price - df * sigma_n * pdf / (2.0 * sq),
    )


def bachelier_implied_vol(price: float, F: float, K: float, T: float, r: float = 0.0,
                          option_type: OptionType = "call") -> float:
    """Normal implied vol; raises ``ValueError`` at or below intrinsic."""
    _check(F, K, T, 0.0, r, option_type)
    if not math.isfinite(price) or T <= 0.0:
        raise ValueError("need a finite price and T > 0")
    df = math.exp(-r * T)
    sign = 1.0 if option_type == "call" else -1.0
    if not price > df * max(sign * (F - K), 0.0):
        raise ValueError("price at or below intrinsic: no implied vol")
    if sign * (F - K) > 0.0:
        # Invert the out-of-the-money option (parity), which carries no
        # intrinsic value and so keeps every digit of the time value.
        price -= df * sign * (F - K)
        sign = -sign
        option_type = "call" if sign > 0.0 else "put"
    lo, hi = 0.0, max(abs(F - K), 1.0) / math.sqrt(T)
    while bachelier_price(F, K, T, hi, r, option_type) < price:
        hi *= 2.0
        if hi > 1e300:
            raise ValueError("price above any finite normal vol")
    sig = min(price / df * math.sqrt(2.0 * math.pi / T), hi)  # ATM inversion
    for _ in range(200):
        diff = bachelier_price(F, K, T, sig, r, option_type) - price
        if abs(diff) <= 4.0 * 2.0**-52 * price:
            return sig
        if diff > 0.0:
            hi = sig
        else:
            lo = sig
        d = (F - K) / (sig * math.sqrt(T)) if sig > 0.0 else math.inf
        vega = df * math.sqrt(T) * _INV_SQRT_2PI * math.exp(-0.5 * d * d)
        step = sig - diff / vega if vega > 0.0 else math.nan
        new = step if lo <= step <= hi else 0.5 * (lo + hi)
        if abs(new - sig) <= 1e-14 * new:
            return new
        sig = new
    return sig
