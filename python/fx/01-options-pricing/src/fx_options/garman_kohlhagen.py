"""Garman-Kohlhagen pricing for European FX options.

Garman-Kohlhagen (1983) is Black-Scholes with the continuous dividend yield
replaced by the foreign interest rate: holding the foreign currency pays the
foreign risk-free rate, exactly as a dividend-paying stock pays its yield.

Conventions: pair BASE/QUOTE (EURUSD = USD per EUR); ``S`` and prices in
domestic (quote) currency per unit foreign (base) notional; ``r_d`` = quote
currency rate, ``r_f`` = base currency rate; rates continuously compounded,
annualised, ACT/365F.

Formulae
--------
    d1 = [ln(S/K) + (r_d - r_f + sigma^2/2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    call = S e^{-r_f T} N(d1) - K e^{-r_d T} N(d2)
    put  = K e^{-r_d T} N(-d2) - S e^{-r_f T} N(-d1)

Limits handled explicitly: T = 0 returns intrinsic value; sigma = 0 (or
sigma*sqrt(T) = 0) returns the discounted intrinsic on the forward,
``e^{-r_d T} max(phi (F - K), 0)``.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm

from ._common import validate_inputs, validate_option_type

__all__ = ["d1", "d2", "gk_price", "gk_call", "gk_put", "implied_vol"]

_MIN_VOL = 1e-12


def d1(S: float, K: float, T: float, r_d: float, r_f: float,
       sigma: float) -> float:
    """Garman-Kohlhagen d1.

    Parameters
    ----------
    S, K : float
        Spot (domestic per foreign) and strike, both > 0.
    T : float
        Time to expiry in years, > 0.
    r_d, r_f : float
        Domestic / foreign continuously compounded rates.
    sigma : float
        Annualised volatility, > 0.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If inputs are invalid or ``sigma * sqrt(T)`` is zero.
    """
    validate_inputs(S, K, T, r_d, r_f, sigma)
    vol_sqrt_t = sigma * math.sqrt(T)
    if vol_sqrt_t <= _MIN_VOL:
        raise ValueError(
            f"d1 undefined for sigma*sqrt(T)={vol_sqrt_t}; need sigma>0, T>0"
        )
    return (math.log(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / vol_sqrt_t


def d2(S: float, K: float, T: float, r_d: float, r_f: float,
       sigma: float) -> float:
    """Garman-Kohlhagen d2 = d1 - sigma*sqrt(T).  See :func:`d1`."""
    return d1(S, K, T, r_d, r_f, sigma) - sigma * math.sqrt(T)


def gk_price(S: float, K: float, T: float, r_d: float, r_f: float,
             sigma: float, option_type: str) -> float:
    """Garman-Kohlhagen price of a European FX option.

    Price is in domestic (quote) currency per unit of foreign (base)
    notional, e.g. USD per EUR for EURUSD.

    Parameters
    ----------
    S : float
        Spot FX rate, domestic per unit foreign, > 0.
    K : float
        Strike in the same quotation, > 0.
    T : float
        Time to expiry in years, >= 0 (T = 0 returns intrinsic).
    r_d : float
        Domestic (quote-currency) continuously compounded rate.
    r_f : float
        Foreign (base-currency) continuously compounded rate.
    sigma : float
        Annualised volatility, >= 0 (sigma = 0 returns discounted
        forward intrinsic).
    option_type : str
        ``"call"`` (call on the base currency) or ``"put"``.

    Returns
    -------
    float
        Option premium in domestic currency per unit foreign notional.

    Raises
    ------
    ValueError
        On invalid inputs (see :func:`fx_options._common.validate_inputs`).
    """
    phi = validate_option_type(option_type)
    validate_inputs(S, K, T, r_d, r_f, sigma)
    if T == 0.0:
        return max(phi * (S - K), 0.0)
    if sigma * math.sqrt(T) <= _MIN_VOL:
        forward = S * math.exp((r_d - r_f) * T)
        return math.exp(-r_d * T) * max(phi * (forward - K), 0.0)
    _d1 = d1(S, K, T, r_d, r_f, sigma)
    _d2 = _d1 - sigma * math.sqrt(T)
    return phi * (
        S * math.exp(-r_f * T) * norm.cdf(phi * _d1)
        - K * math.exp(-r_d * T) * norm.cdf(phi * _d2)
    )


def gk_call(S: float, K: float, T: float, r_d: float, r_f: float,
            sigma: float) -> float:
    """Convenience wrapper: ``gk_price(..., "call")``."""
    return gk_price(S, K, T, r_d, r_f, sigma, "call")


def gk_put(S: float, K: float, T: float, r_d: float, r_f: float,
           sigma: float) -> float:
    """Convenience wrapper: ``gk_price(..., "put")``."""
    return gk_price(S, K, T, r_d, r_f, sigma, "put")


def _vega(S: float, K: float, T: float, r_d: float, r_f: float,
          sigma: float) -> float:
    """GK vega (dV/dsigma), same for calls and puts."""
    _d1 = d1(S, K, T, r_d, r_f, sigma)
    return S * math.exp(-r_f * T) * norm.pdf(_d1) * math.sqrt(T)


def implied_vol(price: float, S: float, K: float, T: float, r_d: float,
                r_f: float, option_type: str, tol: float = 1e-12,
                max_iter: int = 100) -> float:
    """Implied Garman-Kohlhagen volatility from a domestic-currency premium.

    Strategy: Newton-Raphson from an initial guess (fast quadratic
    convergence when vega is healthy), falling back to bracketed Brent
    (guaranteed convergence) if Newton stalls or wanders outside the
    no-arbitrage bracket.

    Parameters
    ----------
    price : float
        Observed premium, domestic ccy per unit foreign notional.
    S, K, T, r_d, r_f : float
        As in :func:`gk_price`.  Requires T > 0.
    option_type : str
        ``"call"`` or ``"put"``.
    tol : float
        Absolute tolerance on the vol root.
    max_iter : int
        Newton iteration budget before Brent fallback.

    Returns
    -------
    float
        Implied volatility (annualised).

    Raises
    ------
    ValueError
        If the price violates the no-arbitrage bounds
        ``[discounted intrinsic on the forward, discounted forward
        bound]``, if T = 0, or if the price sits in the "flat plateau"
        near the sigma -> infinity bound where deep ITM + long-dated +
        high vol drives ``N(d1)``/``N(d2)`` to saturate to 0/1 in double
        precision (vol is genuinely unrecoverable there, not just hard --
        see docs/VALIDATION.md, failure mode 4).
    """
    phi = validate_option_type(option_type)
    validate_inputs(S, K, T, r_d, r_f, 0.0)
    if T <= 0.0:
        raise ValueError("implied_vol requires T > 0")
    if not math.isfinite(price):
        raise ValueError(f"price must be finite, got {price!r}")

    df_d = math.exp(-r_d * T)
    df_f = math.exp(-r_f * T)
    forward = S * df_f / df_d
    lower = df_d * max(phi * (forward - K), 0.0)  # sigma -> 0 limit
    upper = S * df_f if phi > 0 else K * df_d     # sigma -> inf limit
    if price < lower - 1e-14 or price > upper + 1e-14:
        raise ValueError(
            f"price {price} outside no-arbitrage bounds [{lower}, {upper}]"
        )
    if price - lower <= 1e-16 * max(1.0, lower):
        # Time value below double-precision resolution: vol unrecoverable,
        # return the sigma -> 0 limit (documented in docs/VALIDATION.md).
        return 0.0

    def objective(sig: float) -> float:
        return gk_price(S, K, T, r_d, r_f, sig, option_type) - price

    # Newton with a moneyness-aware start (Brenner-Subrahmanyam flavoured).
    sigma = max(0.05, math.sqrt(2.0 * abs(math.log(forward / K)) / T))
    lo, hi = 1e-10, 10.0
    for _ in range(max_iter):
        diff = objective(sigma)
        if abs(diff) < 1e-14:
            return sigma
        vega = _vega(S, K, T, r_d, r_f, sigma)
        if vega < 1e-12:
            break  # flat objective; Newton unreliable -> Brent
        step = diff / vega
        new_sigma = sigma - step
        if not (lo < new_sigma < hi):
            break
        if abs(new_sigma - sigma) < tol:
            return new_sigma
        sigma = new_sigma

    # Brent fallback on an expanding bracket.
    hi = 1.0
    f_hi = objective(hi)
    while f_hi < 0.0 and hi < 50.0:
        hi *= 2.0
        f_hi = objective(hi)
    if f_hi < 0.0:
        raise ValueError(
            f"implied vol > {hi}: price {price} unattainably high"
        )
    if f_hi == 0.0:
        # objective(hi) landed exactly on zero without ever going strictly
        # positive during expansion: deep ITM + long-dated + high vol drives
        # |d1|, |d2| large enough that N(d1)/N(d2) saturate to 0 or 1 in
        # double precision, so gk_price(sigma) is bit-identical to the
        # sigma -> inf bound for every sigma from the true root up to `hi`
        # (and beyond -- this is not a bracket, it is a flat plateau). Any
        # point in that plateau is an equally "valid" root of the floating
        # point objective, so accepting `hi` (an arbitrary artifact of the
        # doubling schedule) would silently return a vol that can be wrong
        # by whole vol points or more with no signal to the caller. This is
        # the upper-bound mirror of the near-`lower` short-circuit above;
        # unlike that case, there is no finite limiting sigma to fall back
        # to (sigma -> infinity is not representable), so the honest
        # answer is that the vol is unrecoverable at this precision.
        raise ValueError(
            f"price {price} is within double-precision resolution of the "
            f"sigma->inf bound {upper}; implied volatility is unrecoverably "
            "large (vega has underflowed to zero in this regime -- see "
            "docs/VALIDATION.md)"
        )
    return float(brentq(objective, lo, hi, xtol=tol, maxiter=200))
