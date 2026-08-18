"""Minimal Black-Scholes pricer, vega and a robust implied-volatility solver.

Conventions (see CONVENTIONS.md):

* Rates ``r`` and dividend yields ``q`` are continuously compounded, annualised,
  ACT/365F.
* ``sigma`` is the annualised volatility of log-returns.
* ``T`` is the time to expiry in years.

The implied-vol solver brackets the root with Brent's method and polishes it
with a few Newton steps.  Prices below discounted intrinsic value, above the
upper no-arbitrage bound, or so deep in the wings that vega underflows are
*rejected*: the solver returns ``nan`` and emits an :class:`ImpliedVolWarning`
rather than producing a garbage number.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "ImpliedVolWarning",
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "implied_vol",
    "implied_vol_vector",
]

OptionKind = Literal["call", "put"]

# Solver bracket for annualised volatility.
_SIGMA_LO = 1e-6
_SIGMA_HI = 5.0
# Vega below this (per unit vol, i.e. dPrice/dSigma) is treated as numerically
# dead: the price carries no usable vol information in double precision.
_MIN_VEGA = 1e-12


class ImpliedVolWarning(UserWarning):
    """Warns when an implied volatility cannot be recovered reliably."""


def _validate_market(S: float, K: float, T: float) -> None:
    if S <= 0.0:
        raise ValueError(f"spot must be positive, got S={S}")
    if K <= 0.0:
        raise ValueError(f"strike must be positive, got K={K}")
    if T < 0.0:
        raise ValueError(f"time to expiry must be non-negative, got T={T}")


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    kind: OptionKind = "call",
) -> float:
    """Black-Scholes price of a European option.

    Parameters
    ----------
    S : float
        Spot price (>0).
    K : float
        Strike (>0).
    T : float
        Time to expiry in years (>=0).  ``T == 0`` returns intrinsic value.
    r : float
        Continuously compounded risk-free rate (annualised).
    q : float
        Continuously compounded dividend yield (annualised).
    sigma : float
        Annualised log-return volatility (>=0).  ``sigma == 0`` returns the
        discounted-intrinsic (deterministic-forward) value.
    kind : {"call", "put"}
        Option type.

    Returns
    -------
    float
        Present value of the option.
    """
    _validate_market(S, K, T)
    if sigma < 0.0:
        raise ValueError(f"volatility must be non-negative, got sigma={sigma}")
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")

    if T == 0.0:
        intrinsic = S - K if kind == "call" else K - S
        return max(intrinsic, 0.0)

    F = S * np.exp((r - q) * T)
    df = np.exp(-r * T)
    if sigma == 0.0:
        intrinsic = F - K if kind == "call" else K - F
        return df * max(intrinsic, 0.0)

    srt = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * srt * srt) / srt
    d2 = d1 - srt
    if kind == "call":
        return float(df * (F * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(df * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))


def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes vega, dPrice/dSigma (identical for calls and puts).

    Returns
    -------
    float
        Sensitivity of the option price to a unit (1.00 = 100 vol points)
        change in annualised volatility.
    """
    _validate_market(S, K, T)
    if sigma <= 0.0 or T == 0.0:
        return 0.0
    F = S * np.exp((r - q) * T)
    srt = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * srt * srt) / srt
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def bs_delta(
    S: float, K: float, T: float, r: float, q: float, sigma: float, kind: OptionKind = "call"
) -> float:
    """Black-Scholes spot delta, dPrice/dS."""
    _validate_market(S, K, T)
    if T == 0.0 or sigma <= 0.0:
        F = S * np.exp((r - q) * T)
        itm = F > K if kind == "call" else F < K
        sign = 1.0 if kind == "call" else -1.0
        return float(sign * np.exp(-q * T)) if itm else 0.0
    srt = sigma * np.sqrt(T)
    F = S * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * srt * srt) / srt
    if kind == "call":
        return float(np.exp(-q * T) * norm.cdf(d1))
    return float(-np.exp(-q * T) * norm.cdf(-d1))


def bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes gamma, d2Price/dS2 (identical for calls and puts)."""
    _validate_market(S, K, T)
    if T == 0.0 or sigma <= 0.0:
        return 0.0
    srt = sigma * np.sqrt(T)
    F = S * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * srt * srt) / srt
    return float(np.exp(-q * T) * norm.pdf(d1) / (S * srt))


def _price_bounds(
    S: float, K: float, T: float, r: float, q: float, kind: OptionKind
) -> tuple[float, float]:
    """No-arbitrage lower/upper price bounds for a European option."""
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)
    if kind == "call":
        lower = max(S * df_q - K * df_r, 0.0)
        upper = S * df_q
    else:
        lower = max(K * df_r - S * df_q, 0.0)
        upper = K * df_r
    return float(lower), float(upper)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    kind: OptionKind = "call",
    tol: float = 1e-12,
) -> float:
    """Robust Black-Scholes implied volatility.

    Strategy: verify no-arbitrage price bounds, bracket the root on
    ``[1e-6, 5.0]`` with Brent's method, then polish with Newton iterations
    (safeguarded to stay inside the bracket).

    Failure behaviour -- returns ``nan`` and emits :class:`ImpliedVolWarning`
    (never a garbage number) when:

    * the price is below discounted intrinsic (sub-intrinsic) or above the
      upper bound ``S e^{-qT}`` / ``K e^{-rT}``;
    * the price sits so deep in a wing that vega is numerically zero and no
      volatility in the bracket reproduces it (price indistinguishable from
      the intrinsic/zero limit in double precision);
    * ``T == 0`` (no time value to invert).

    Parameters
    ----------
    price : float
        Observed option present value.
    S, K, T, r, q : float
        Market inputs as in :func:`bs_price`.
    kind : {"call", "put"}
        Option type.
    tol : float
        Absolute tolerance on the recovered volatility.

    Returns
    -------
    float
        Implied volatility, or ``nan`` on failure (with a warning).
    """
    _validate_market(S, K, T)
    if not np.isfinite(price):
        warnings.warn(
            f"implied_vol: non-finite price {price}; returning nan", ImpliedVolWarning
        )
        return np.nan
    if T == 0.0:
        warnings.warn("implied_vol: T=0 option has no time value; returning nan", ImpliedVolWarning)
        return np.nan

    lower, upper = _price_bounds(S, K, T, r, q, kind)
    # Tolerance scaled to price magnitude: below this the price carries no vol info.
    eps = 1e-12 * max(S, K)
    if price < lower - eps:
        warnings.warn(
            f"implied_vol: price {price:.6g} below discounted intrinsic {lower:.6g} "
            "(sub-intrinsic, arbitrageable quote); returning nan",
            ImpliedVolWarning,
        )
        return np.nan
    if price > upper + eps:
        warnings.warn(
            f"implied_vol: price {price:.6g} above upper bound {upper:.6g}; returning nan",
            ImpliedVolWarning,
        )
        return np.nan
    if price < eps:
        warnings.warn(
            f"implied_vol: price {price:.6g} is numerically zero at this scale "
            "(deep OTM wing); vol is unidentifiable, returning nan",
            ImpliedVolWarning,
        )
        return np.nan
    if lower > 0.0 and price - lower < eps:
        warnings.warn(
            "implied_vol: price carries no measurable time value above intrinsic "
            "(deep ITM wing, vega ~ 0); vol is unidentifiable, returning nan",
            ImpliedVolWarning,
        )
        return np.nan

    def objective(sig: float) -> float:
        return bs_price(S, K, T, r, q, sig, kind) - price

    f_lo = objective(_SIGMA_LO)
    f_hi = objective(_SIGMA_HI)
    if f_lo > 0.0 or f_hi < 0.0:
        # Price is numerically indistinguishable from the sigma->0 limit
        # (deep wing, zero vega) or exceeds the sigma->5 price.
        if abs(f_lo) <= eps:
            warnings.warn(
                "implied_vol: price equals the zero-vol limit to machine precision "
                "(deep wing, vega ~ 0); vol is unidentifiable, returning nan",
                ImpliedVolWarning,
            )
        else:
            warnings.warn(
                "implied_vol: no volatility in [1e-6, 5.0] reproduces the price "
                "(deep wing or extreme quote); returning nan",
                ImpliedVolWarning,
            )
        return np.nan

    sigma = brentq(objective, _SIGMA_LO, _SIGMA_HI, xtol=max(tol, 1e-14), rtol=8.9e-16)

    # Newton polish: quadratic convergence near the root, guarded by the bracket.
    for _ in range(3):
        vega = bs_vega(S, K, T, r, q, sigma)
        if vega < _MIN_VEGA:
            break
        step = objective(sigma) / vega
        new = sigma - step
        if not (_SIGMA_LO <= new <= _SIGMA_HI):
            break
        sigma = new
        if abs(step) < tol:
            break

    # Final safety: the recovered vol must actually reproduce the price.
    if abs(objective(sigma)) > max(1e-6 * price, eps):
        warnings.warn(
            "implied_vol: solver could not reproduce the price to tolerance "
            "(near-zero vega region); returning nan",
            ImpliedVolWarning,
        )
        return np.nan
    return float(sigma)


def implied_vol_vector(
    prices: np.ndarray,
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    q: float,
    kind: OptionKind = "call",
) -> np.ndarray:
    """Vectorised wrapper over :func:`implied_vol` for one expiry.

    Returns an array of implied vols with ``nan`` where inversion fails.
    """
    prices = np.asarray(prices, dtype=float)
    K = np.asarray(K, dtype=float)
    if prices.shape != K.shape:
        raise ValueError("prices and strikes must have the same shape")
    out = np.empty_like(prices)
    for i in range(prices.size):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ImpliedVolWarning)
            out.flat[i] = implied_vol(prices.flat[i], S, K.flat[i], T, r, q, kind)
    return out
