"""Garman-Kohlhagen pricer, FX delta conventions, and robust implied vol.

Conventions (see CONVENTIONS.md at the portfolio root):

* Pairs are quoted BASE/QUOTE (EURUSD = USD per 1 EUR).  The *domestic*
  rate ``r_d`` is the QUOTE-currency rate, the *foreign* rate ``r_f`` is
  the BASE-currency rate.  Garman-Kohlhagen is Black-Scholes with a
  continuous dividend yield ``q = r_f``.
* All rates are continuously compounded, annualised (ACT/365F year
  fractions ``T``).  Vols are annualised log-return vols in decimals
  (0.10 = 10%).  ``cp = +1`` for calls (on the base currency),
  ``cp = -1`` for puts.
* Prices are in domestic (quote) currency per one unit of base currency.

Delta conventions (first-class, all four market standards):

===============  ==========================================================
``spot``         unadjusted spot delta  ``w * exp(-r_f T) * N(w d1)``
``forward``      unadjusted forward delta  ``w * N(w d1)``
``spot_pa``      premium-adjusted spot delta ``w * exp(-r_f T) (K/F) N(w d2)``
``forward_pa``   premium-adjusted forward delta ``w * (K/F) * N(w d2)``
===============  ==========================================================

Premium-adjusted deltas arise when the option premium is paid in the
*base* currency (market standard for USDJPY and most USD-base pairs):
the premium itself is a position in the underlying and is subtracted
from the hedge, ``delta_pa = delta - V/S``.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm

__all__ = [
    "DELTA_CONVENTIONS",
    "gk_price",
    "gk_forward",
    "gk_delta",
    "gk_vega",
    "gk_gamma",
    "gk_vanna",
    "gk_volga",
    "gk_rho_domestic",
    "gk_rho_foreign",
    "gk_theta",
    "gk_digital",
    "implied_vol",
]

DELTA_CONVENTIONS = ("spot", "forward", "spot_pa", "forward_pa")

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _phi(x: float) -> float:
    """Standard normal pdf."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _validate(S: float, K: float, T: float, sigma: float | None = None) -> None:
    """Validate pricer inputs.

    Note the explicit ``isfinite`` guards: a bare ``S <= 0`` test is
    silently passed by ``NaN`` (every comparison with NaN is False), so a
    NaN spot or vol would otherwise propagate into a NaN premium — a
    price that looks like a number to every downstream consumer.
    """
    if not math.isfinite(S) or S <= 0.0:
        raise ValueError(f"spot must be finite and positive, got S={S}")
    if not math.isfinite(K) or K <= 0.0:
        raise ValueError(f"strike must be finite and positive, got K={K}")
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError(f"time to expiry must be finite and positive, got T={T}")
    if sigma is not None and (not math.isfinite(sigma) or sigma <= 0.0):
        raise ValueError(f"volatility must be finite and positive, got sigma={sigma}")


def _check_cp(cp: int) -> None:
    if cp not in (+1, -1):
        raise ValueError(f"cp must be +1 (call) or -1 (put), got {cp}")


def _check_rates(r_d: float, r_f: float) -> None:
    """Reject non-finite rates (NaN slips past every inequality test)."""
    if not math.isfinite(r_d):
        raise ValueError(f"domestic (quote-ccy) rate must be finite, got r_d={r_d}")
    if not math.isfinite(r_f):
        raise ValueError(f"foreign (base-ccy) rate must be finite, got r_f={r_f}")


def _d1_d2(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float):
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / (sigma * sqT)
    return d1, d1 - sigma * sqT


def gk_forward(S: float, T: float, r_d: float, r_f: float) -> float:
    """FX forward ``F = S * exp((r_d - r_f) T)`` (covered interest parity)."""
    _check_rates(r_d, r_f)
    if not math.isfinite(S) or S <= 0.0:
        raise ValueError(f"spot must be finite and positive, got S={S}")
    if not math.isfinite(T):
        raise ValueError(f"T must be finite, got T={T}")
    return S * math.exp((r_d - r_f) * T)


def gk_price(
    S: float, K: float, T: float, r_d: float, r_f: float, sigma: float, cp: int = 1
) -> float:
    """Garman-Kohlhagen price of a European FX vanilla.

    Parameters
    ----------
    S : float
        Spot, domestic units per unit of foreign (base) currency.
    K : float
        Strike, same units as spot.
    T : float
        Time to expiry in years (ACT/365F), ``T > 0``.
    r_d, r_f : float
        Domestic (quote-ccy) and foreign (base-ccy) continuously
        compounded zero rates.
    sigma : float
        Annualised lognormal volatility, decimal.
    cp : int
        +1 call on base currency, -1 put.

    Returns
    -------
    float
        Premium in domestic currency per unit foreign notional.
    """
    _validate(S, K, T, sigma)
    _check_cp(cp)
    _check_rates(r_d, r_f)
    d1, d2 = _d1_d2(S, K, T, r_d, r_f, sigma)
    return cp * (
        S * math.exp(-r_f * T) * ndtr(cp * d1) - K * math.exp(-r_d * T) * ndtr(cp * d2)
    )


def gk_delta(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    sigma: float,
    cp: int = 1,
    convention: str = "spot",
) -> float:
    """FX delta under one of the four market conventions.

    ``spot``/``forward`` are the unadjusted Black deltas; ``spot_pa`` and
    ``forward_pa`` are premium-adjusted (premium paid in base currency),
    ``delta_pa = delta - V/S`` which collapses to
    ``w * DF * (K/F) * N(w d2)``.

    Returns the *signed* delta (calls positive, puts negative).
    """
    _validate(S, K, T, sigma)
    _check_cp(cp)
    _check_rates(r_d, r_f)
    if convention not in DELTA_CONVENTIONS:
        raise ValueError(
            f"unknown delta convention {convention!r}; expected one of {DELTA_CONVENTIONS}"
        )
    d1, d2 = _d1_d2(S, K, T, r_d, r_f, sigma)
    F = gk_forward(S, T, r_d, r_f)
    if convention == "spot":
        return cp * math.exp(-r_f * T) * ndtr(cp * d1)
    if convention == "forward":
        return cp * ndtr(cp * d1)
    if convention == "spot_pa":
        return cp * math.exp(-r_f * T) * (K / F) * ndtr(cp * d2)
    return cp * (K / F) * ndtr(cp * d2)  # forward_pa


def gk_vega(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float) -> float:
    """dV/dsigma (per unit vol, i.e. per 1.00 = 100 vol points)."""
    _validate(S, K, T, sigma)
    _check_rates(r_d, r_f)
    d1, _ = _d1_d2(S, K, T, r_d, r_f, sigma)
    return S * math.exp(-r_f * T) * math.sqrt(T) * _phi(d1)


def gk_gamma(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float) -> float:
    """d2V/dS2 (spot gamma)."""
    _validate(S, K, T, sigma)
    _check_rates(r_d, r_f)
    d1, _ = _d1_d2(S, K, T, r_d, r_f, sigma)
    return math.exp(-r_f * T) * _phi(d1) / (S * sigma * math.sqrt(T))


def gk_vanna(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float) -> float:
    """d2V/dS dsigma.  Positive for high strikes (d2 < 0), negative for
    strikes well below the forward (d2 > 0)."""
    _validate(S, K, T, sigma)
    _check_rates(r_d, r_f)
    d1, d2 = _d1_d2(S, K, T, r_d, r_f, sigma)
    return -math.exp(-r_f * T) * _phi(d1) * d2 / sigma


def gk_volga(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float) -> float:
    """d2V/dsigma2 (vomma).  Positive away from ATM, ~0 at the DNS strike."""
    _validate(S, K, T, sigma)
    _check_rates(r_d, r_f)
    d1, d2 = _d1_d2(S, K, T, r_d, r_f, sigma)
    return gk_vega(S, K, T, r_d, r_f, sigma) * d1 * d2 / sigma


def gk_rho_domestic(
    S: float, K: float, T: float, r_d: float, r_f: float, sigma: float, cp: int = 1
) -> float:
    """dV/dr_d.  Positive for calls (forward rises with r_d)."""
    _validate(S, K, T, sigma)
    _check_cp(cp)
    _check_rates(r_d, r_f)
    _, d2 = _d1_d2(S, K, T, r_d, r_f, sigma)
    return cp * K * T * math.exp(-r_d * T) * ndtr(cp * d2)


def gk_rho_foreign(
    S: float, K: float, T: float, r_d: float, r_f: float, sigma: float, cp: int = 1
) -> float:
    """dV/dr_f.  Negative for calls (forward falls with r_f)."""
    _validate(S, K, T, sigma)
    _check_cp(cp)
    _check_rates(r_d, r_f)
    d1, _ = _d1_d2(S, K, T, r_d, r_f, sigma)
    return -cp * S * T * math.exp(-r_f * T) * ndtr(cp * d1)


def gk_theta(
    S: float, K: float, T: float, r_d: float, r_f: float, sigma: float, cp: int = 1
) -> float:
    """Calendar theta dV/dt = -dV/dT, per year."""
    _validate(S, K, T, sigma)
    _check_cp(cp)
    _check_rates(r_d, r_f)
    d1, d2 = _d1_d2(S, K, T, r_d, r_f, sigma)
    term = -S * math.exp(-r_f * T) * _phi(d1) * sigma / (2.0 * math.sqrt(T))
    term += cp * (
        r_f * S * math.exp(-r_f * T) * ndtr(cp * d1)
        - r_d * K * math.exp(-r_d * T) * ndtr(cp * d2)
    )
    return term


def gk_digital(
    S: float, K: float, T: float, r_d: float, r_f: float, sigma: float, cp: int = 1
) -> float:
    """Domestic cash-or-nothing digital (pays 1 domestic unit if ITM).

    Flat-vol (no smile) value ``exp(-r_d T) N(cp * d2)``.  A real desk
    adds the skew correction ``-vega_digital * dsigma/dK``; see
    :func:`fx_surface.smile.smile_digital` for the smile-consistent value.
    """
    _validate(S, K, T, sigma)
    _check_cp(cp)
    _check_rates(r_d, r_f)
    _, d2 = _d1_d2(S, K, T, r_d, r_f, sigma)
    return math.exp(-r_d * T) * ndtr(cp * d2)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    cp: int = 1,
    lo: float = 1e-6,
    hi: float = 5.0,
    tol: float = 1e-12,
    on_fail: str = "raise",
) -> float:
    """Robust Garman-Kohlhagen implied volatility (Brent).

    Checks the static no-arbitrage bounds first, expands the upper
    bracket geometrically up to ``hi``, then runs Brent to ``tol``.

    Parameters
    ----------
    on_fail : {"raise", "nan"}
        Behaviour when the price violates no-arbitrage bounds (up to a
        1e-12 absolute slack) or exceeds the vol bracket.

    Returns
    -------
    float
        Implied vol in decimals, or NaN when ``on_fail='nan'``.
    """
    _validate(S, K, T)
    _check_cp(cp)
    df_d = math.exp(-r_d * T)
    df_f = math.exp(-r_f * T)
    lower = max(cp * (S * df_f - K * df_d), 0.0)
    upper = S * df_f if cp == 1 else K * df_d
    slack = 1e-12
    if price < lower - slack or price > upper + slack:
        if on_fail == "nan":
            return math.nan
        raise ValueError(
            f"price {price} violates no-arbitrage bounds [{lower}, {upper}] "
            f"for {'call' if cp == 1 else 'put'} K={K}, T={T}"
        )
    # Clip prices inside the slack band onto the bounds.
    price = min(max(price, lower), upper)
    if price - lower < 1e-14:  # effectively intrinsic -> vol ~ 0
        return lo

    def objective(sigma: float) -> float:
        return gk_price(S, K, T, r_d, r_f, sigma, cp) - price

    f_lo = objective(lo)
    if f_lo > 0.0:  # price below the sigma=lo price: vol under the bracket
        return lo
    b = 0.5
    while objective(b) < 0.0:
        b *= 2.0
        if b > hi:
            if on_fail == "nan":
                return math.nan
            raise ValueError(
                f"implied vol above bracket {hi} for price={price}, K={K}, T={T}"
            )
    return brentq(objective, lo, b, xtol=tol, rtol=8.9e-16, maxiter=200)
