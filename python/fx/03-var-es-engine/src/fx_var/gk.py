"""Minimal Garman-Kohlhagen pricer and Greeks for FX options.

Implemented internally (self-contained, vectorised) so this project does not
import the options-pricing project.  Garman-Kohlhagen is Black-Scholes with
the foreign (base-currency) rate playing the role of a continuous dividend
yield: for pair BASE/QUOTE with spot X (QUOTE per 1 BASE),

    F = X * exp((r_d - r_f) * T),
    price_quote = e^{-r_d T} * (F N(d1) - K N(d2))      (call)

with r_d = quote-currency cc rate, r_f = base-currency cc rate, both
annualised ACT/365, sigma the annualised lognormal vol of X, T in years.
Prices are in QUOTE currency per 1 unit of BASE notional.

Only the pieces needed by the VaR engine are provided: price, delta (spot),
gamma, vega.  All functions broadcast over NumPy arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["gk_price", "gk_delta", "gk_gamma", "gk_vega", "gk_d1_d2"]

_EPS_T = 1e-12
_EPS_V = 1e-12


def _validate(strike, expiry, vol) -> None:
    # Non-finite inputs must be refused rather than absorbed: the degenerate
    # sigma*sqrt(T) branch below treats a NaN as the *zero-vol* case and would
    # otherwise return forward intrinsic for a NaN vol -- a silently wrong
    # price rather than an error.
    for name, value in (("strike", strike), ("expiry", expiry), ("vol", vol)):
        if not np.all(np.isfinite(np.asarray(value, dtype=float))):
            raise ValueError(
                f"{name} must be finite, got {value!r} "
                "(NaN policy: refuse, never impute)"
            )
    if np.any(np.asarray(strike) <= 0):
        raise ValueError("strike must be > 0")
    if np.any(np.asarray(expiry) < 0):
        raise ValueError("expiry must be >= 0")
    if np.any(np.asarray(vol) < 0):
        raise ValueError("vol must be >= 0")


def gk_d1_d2(spot, strike, expiry, r_d, r_f, vol):
    """Return (d1, d2); degenerate T=0 / vol=0 handled via +/-inf limits."""
    spot = np.asarray(spot, dtype=float)
    sig_sqrt_t = np.maximum(np.sqrt(np.maximum(expiry, 0.0)) * vol, 0.0)
    safe = np.where(sig_sqrt_t > _EPS_V, sig_sqrt_t, 1.0)
    d1 = (np.log(spot / strike) + (r_d - r_f + 0.5 * vol**2) * expiry) / safe
    d2 = d1 - sig_sqrt_t
    # deterministic limit: sign of ln(F/K)
    fwd_ratio = np.log(spot / strike) + (r_d - r_f) * expiry
    lim = np.where(fwd_ratio >= 0, np.inf, -np.inf)
    d1 = np.where(sig_sqrt_t > _EPS_V, d1, lim)
    d2 = np.where(sig_sqrt_t > _EPS_V, d2, lim)
    return d1, d2


def gk_price(spot, strike, expiry, r_d, r_f, vol, kind: str = "call"):
    """Garman-Kohlhagen price in QUOTE ccy per unit BASE notional.

    Parameters
    ----------
    spot : array_like
        Spot X, QUOTE per 1 BASE.
    strike : array_like
        Strike K in the same quotation.
    expiry : array_like
        Time to expiry in years (ACT/365).
    r_d, r_f : array_like
        Quote-ccy (domestic) and base-ccy (foreign) cc rates, annualised.
    vol : array_like
        Annualised lognormal vol of X.
    kind : {"call", "put"}
        Call = right to buy BASE (receive BASE, pay QUOTE at K).
    """
    _validate(strike, expiry, vol)
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    spot = np.asarray(spot, dtype=float)
    d1, d2 = gk_d1_d2(spot, strike, expiry, r_d, r_f, vol)
    df_d = np.exp(-np.asarray(r_d, dtype=float) * expiry)
    df_f = np.exp(-np.asarray(r_f, dtype=float) * expiry)
    if kind == "call":
        return spot * df_f * norm.cdf(d1) - strike * df_d * norm.cdf(d2)
    return strike * df_d * norm.cdf(-d2) - spot * df_f * norm.cdf(-d1)


def gk_delta(spot, strike, expiry, r_d, r_f, vol, kind: str = "call"):
    """Spot delta dV/dX (unit BASE notional, price in QUOTE ccy)."""
    d1, _ = gk_d1_d2(spot, strike, expiry, r_d, r_f, vol)
    df_f = np.exp(-np.asarray(r_f, dtype=float) * expiry)
    if kind == "call":
        return df_f * norm.cdf(d1)
    if kind == "put":
        return df_f * (norm.cdf(d1) - 1.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def gk_gamma(spot, strike, expiry, r_d, r_f, vol):
    """Spot gamma d2V/dX2 (same for calls and puts)."""
    spot = np.asarray(spot, dtype=float)
    d1, _ = gk_d1_d2(spot, strike, expiry, r_d, r_f, vol)
    sig_sqrt_t = np.sqrt(np.maximum(expiry, 0.0)) * np.asarray(vol, dtype=float)
    df_f = np.exp(-np.asarray(r_f, dtype=float) * expiry)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = df_f * norm.pdf(d1) / (spot * sig_sqrt_t)
    return np.where(sig_sqrt_t > _EPS_V, g, 0.0)


def gk_vega(spot, strike, expiry, r_d, r_f, vol):
    """dV/dsigma per 1.00 (=100 vol points) of annualised vol."""
    spot = np.asarray(spot, dtype=float)
    d1, _ = gk_d1_d2(spot, strike, expiry, r_d, r_f, vol)
    df_f = np.exp(-np.asarray(r_f, dtype=float) * expiry)
    sqrt_t = np.sqrt(np.maximum(expiry, 0.0))
    v = spot * df_f * norm.pdf(d1) * sqrt_t
    return np.where(np.isfinite(d1), v, 0.0)
