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
import numbers
import warnings

from scipy.stats import norm

from ._common import finite_result, ncdf, safe_exp, validate_inputs, validate_option_type
from ._iv_core import black_implied_vol

_EPS = 2.220446049250313e-16

__all__ = ["d1", "d2", "gk_call", "gk_price", "gk_put", "implied_vol"]

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
    disc_s = S * safe_exp(-r_f * T, "gk_price")
    disc_k = K * safe_exp(-r_d * T, "gk_price")
    if sigma * math.sqrt(T) <= _MIN_VOL:
        # e^{-r_d T} max(phi (F - K), 0) as S e^{-r_f T} - K e^{-r_d T}:
        # the forward itself can overflow when (r_d - r_f) T > 709.
        return finite_result(max(phi * (disc_s - disc_k), 0.0), "gk_price")
    _d1 = d1(S, K, T, r_d, r_f, sigma)
    _d2 = _d1 - sigma * math.sqrt(T)
    return finite_result(
        phi * (disc_s * ncdf(phi * _d1) - disc_k * ncdf(phi * _d2)), "gk_price")


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
                r_f: float, option_type: str, tol: float | None = None,
                max_iter: int | None = None) -> float:
    """Implied Garman-Kohlhagen volatility from a domestic-currency premium.

    The quote is mapped to normalised Black form and solved on the
    out-of-the-money side (an in-the-money quote is converted by parity)
    with safeguarded Halley iteration and a relative stopping rule. The
    algorithm is shared with the equity engine and the C++ and Rust
    engines (``_iv_core.py``): at most 10 evaluations, accuracy
    ``max(eps * price / vega, 1e-12 * sigma)``.

    Bounds use scale-aware tolerances: ``tau_lo = 4 eps df max(F, K)`` (zero when out of the money) and
    ``tau = 4 eps upper``. A price within ``tau_lo`` of the sigma -> 0
    bound returns 0.0 (the time value is below the bound's own rounding).
    A price within ``tau`` of the sigma -> inf bound raises, because no
    finite vol is identifiable there.

    Parameters
    ----------
    price : float
        Observed premium, domestic ccy per unit foreign notional.
    S, K, T, r_d, r_f : float
        As in :func:`gk_price`.  Requires T > 0.
    option_type : str
        ``"call"`` or ``"put"``.
    tol, max_iter
        Deprecated and ignored. The solver has no tuning parameters.

    Returns
    -------
    float
        Implied volatility (annualised).

    Raises
    ------
    ValueError
        If the price lies outside ``(lower - tau_lo, upper - tau)``, if T = 0,
        if an input is invalid, or if the solver does not converge.
    """
    for name, value in (("tol", tol), ("max_iter", max_iter)):
        if value is not None:
            warnings.warn(
                f"implied_vol({name}=...) is deprecated and ignored; "
                "the solver has no tuning parameters",
                DeprecationWarning, stacklevel=2)
    phi = validate_option_type(option_type)
    validate_inputs(S, K, T, r_d, r_f, 0.0)
    if T <= 0.0:
        raise ValueError("implied_vol requires T > 0")
    if not isinstance(price, numbers.Real) or not math.isfinite(price):
        raise ValueError(f"price must be finite, got {price!r}")

    df_d = safe_exp(-r_d * T, "implied_vol")
    df_f = safe_exp(-r_f * T, "implied_vol")
    forward = finite_result(S * df_f / df_d, "forward")
    lower = df_d * max(phi * (forward - K), 0.0)  # sigma -> 0 limit
    upper = S * df_f if phi > 0 else K * df_d     # sigma -> inf limit
    # Rounding of the sigma -> 0 bound df (F - K): a few eps of df max(F, K)
    # when the option is in the money; exactly zero otherwise.
    tau_lo = 4.0 * _EPS * df_d * max(forward, K) if lower > 0.0 else 0.0
    tau = 4.0 * _EPS * upper      # rounding of the sigma -> inf bound
    if price < lower - tau_lo or price > upper + tau:
        raise ValueError(
            f"price {price} outside no-arbitrage bounds [{lower}, {upper}]"
        )
    if price - lower <= tau_lo:
        # Time value below double-precision resolution: return the
        # sigma -> 0 limit (documented in docs/VALIDATION.md).
        return 0.0
    if upper - price <= tau:
        raise ValueError(
            f"price {price} is within {tau:.3g} of the sigma->inf bound "
            f"{upper}; no finite implied volatility is identifiable"
        )
    return black_implied_vol(float(price), forward, K, T, df_d, phi > 0)
