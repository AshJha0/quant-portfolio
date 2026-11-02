"""Black-Scholes-Merton pricing for European equity options.

Conventions
-----------
* Rates ``r`` and dividend yields ``q`` are continuously compounded,
  annualised (ACT/365F).
* ``T`` is the time to expiry in years; ``sigma`` is the annualised
  volatility of log-returns.
* Prices are in the same currency units as ``S`` and ``K``.
* ``option_type`` is ``"call"`` or ``"put"``.

Edge-case policy (documented + unit tested)
-------------------------------------------
* ``T == 0``   -> intrinsic value ``max(S - K, 0)`` / ``max(K - S, 0)``.
* ``sigma == 0`` -> discounted intrinsic on the forward,
  ``exp(-r T) * max(F - K, 0)`` with ``F = S exp((r - q) T)``.
* ``K == 0``   -> call is a forward on the stock, ``S exp(-q T)``; put is 0.
* ``S == 0``   -> call is 0; put is ``K exp(-r T)``.
* Negative, NaN or infinite ``S``, ``K``, ``T`` or ``sigma`` raise
  ``ValueError``. Negative ``r`` and ``q`` are fully supported.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

from ._iv_core import black_implied_vol

OptionType = Literal["call", "put"]

__all__ = [
    "bs_price",
    "d1_d2",
    "forward_price",
    "implied_vol",
    "intrinsic_value",
    "validate_inputs",
]

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)


def _ncdf(x: float) -> float:
    """Standard normal CDF via ``erfc``: full relative precision in the lower tail."""
    return 0.5 * math.erfc(-x / _SQRT2)


def _check_rates(**rates: float) -> None:
    """Raise ``ValueError`` unless every rate is a finite real number."""
    for name, value in rates.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")


def _exp(x: float, what: str) -> float:
    """``math.exp`` that raises ``ValueError`` (not ``OverflowError``) on overflow."""
    try:
        return math.exp(x)
    except OverflowError:
        raise ValueError(f"{what}: exp({x!r}) overflows a double") from None


def _finite(value: float, what: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{what} is not representable as a finite double")
    return value


def validate_inputs(
    S: float, K: float, T: float, sigma: float, option_type: str = "call"
) -> None:
    """Validate common Black-Scholes inputs, raising ``ValueError`` on error.

    Parameters
    ----------
    S : float
        Spot price, must satisfy ``S >= 0``.
    K : float
        Strike price, must satisfy ``K >= 0``.
    T : float
        Time to expiry in years (ACT/365F), must satisfy ``T >= 0``.
    sigma : float
        Annualised log-return volatility, must satisfy ``sigma >= 0``.
    option_type : str
        ``"call"`` or ``"put"``.

    Raises
    ------
    ValueError
        If any of the constraints above is violated or an input is NaN
        or infinite.
    """
    for name, value in (("S", S), ("K", K), ("T", T), ("sigma", sigma)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value!r}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def intrinsic_value(S: float, K: float, option_type: OptionType = "call") -> float:
    """Intrinsic (exercise-now) value ``max(S - K, 0)`` or ``max(K - S, 0)``.

    Parameters
    ----------
    S, K : float
        Spot and strike, in currency units.
    option_type : {"call", "put"}
        Payoff direction.

    Returns
    -------
    float
        Intrinsic value in currency units.
    """
    if option_type == "call":
        return max(S - K, 0.0)
    return max(K - S, 0.0)


def forward_price(S: float, T: float, r: float, q: float = 0.0) -> float:
    """Equity forward ``F = S * exp((r - q) * T)``.

    Parameters
    ----------
    S : float
        Spot price.
    T : float
        Time to delivery in years (ACT/365F).
    r, q : float
        Continuously compounded annualised risk-free rate and dividend yield.

    Returns
    -------
    float
        Forward price in currency units.
    """
    return S * math.exp((r - q) * T)


def d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0
) -> tuple[float, float]:
    """Black-Scholes ``d1`` and ``d2`` terms.

    ``d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))`` and
    ``d2 = d1 - sigma sqrt(T)``.

    Parameters
    ----------
    S, K : float
        Spot and strike, strictly positive for a finite result.
    T : float
        Time to expiry in years, strictly positive.
    r : float
        Continuously compounded annualised risk-free rate.
    sigma : float
        Annualised volatility, strictly positive.
    q : float
        Continuously compounded annualised dividend yield.

    Returns
    -------
    tuple of float
        ``(d1, d2)``.

    Raises
    ------
    ValueError
        If ``S``, ``K``, ``T`` or ``sigma`` is not strictly positive.
    """
    validate_inputs(S, K, T, sigma)
    _check_rates(r=r, q=q)
    if S <= 0.0 or K <= 0.0 or T <= 0.0 or sigma <= 0.0:
        raise ValueError(
            "d1/d2 require strictly positive S, K, T and sigma; "
            f"got S={S}, K={K}, T={T}, sigma={sigma}"
        )
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Black-Scholes-Merton price of a European option with dividend yield.

    Parameters
    ----------
    S : float
        Spot price (currency units), ``S >= 0``.
    K : float
        Strike price (currency units), ``K >= 0``.
    T : float
        Time to expiry in years (ACT/365F), ``T >= 0``.
    r : float
        Continuously compounded annualised risk-free rate (e.g. 0.05).
        Negative rates are supported.
    sigma : float
        Annualised log-return volatility (e.g. 0.2), ``sigma >= 0``.
    q : float
        Continuously compounded annualised dividend yield.
    option_type : {"call", "put"}
        Option payoff direction.

    Returns
    -------
    float
        Present value of the option in currency units.

    Raises
    ------
    ValueError
        If ``S``, ``K``, ``T`` or ``sigma`` is negative or NaN, or
        ``option_type`` is invalid.

    Notes
    -----
    ``T == 0`` returns intrinsic value; ``sigma == 0`` returns the
    discounted intrinsic on the forward, ``exp(-rT) max(±(F - K), 0)``.
    """
    validate_inputs(S, K, T, sigma, option_type)
    _check_rates(r=r, q=q)
    if T == 0.0:
        return intrinsic_value(S, K, option_type)
    if K == 0.0:
        # Zero-strike call is a (dividend-adjusted) forward on the stock.
        return _finite(S * _exp(-q * T, "bs_price"), "bs_price") if option_type == "call" else 0.0
    if S == 0.0:
        return 0.0 if option_type == "call" else _finite(K * _exp(-r * T, "bs_price"), "bs_price")
    disc_s = S * _exp(-q * T, "bs_price")
    disc_k = K * _exp(-r * T, "bs_price")
    if sigma == 0.0:
        # exp(-rT) max(+-(F - K), 0) written as S e^{-qT} - K e^{-rT}: the
        # forward itself can overflow when (r - q) T > 709.
        sign = 1.0 if option_type == "call" else -1.0
        return _finite(max(sign * (disc_s - disc_k), 0.0), "bs_price")

    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        value = disc_s * _ncdf(d1) - disc_k * _ncdf(d2)
    else:
        value = disc_k * _ncdf(-d2) - disc_s * _ncdf(-d1)
    return _finite(value, "bs_price")


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """Analytic vega dV/dsigma (per unit of vol, i.e. per 100 vol points)."""
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * math.exp(-0.5 * d1 * d1) / _SQRT_2PI * math.sqrt(T)


_IV_DEPRECATED_DEFAULTS = {"tol": 1e-12, "max_iter": 100, "sigma_lo": 1e-9, "sigma_hi": 10.0}


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    tol: float | None = None,
    max_iter: int | None = None,
    sigma_lo: float | None = None,
    sigma_hi: float | None = None,
) -> float:
    """Implied Black-Scholes volatility, solved on the out-of-the-money side.

    The quote is mapped to normalised Black form
    ``b = price / (e^{-rT} sqrt(F K))`` with ``x = -|ln(F/K)|``. An
    in-the-money quote is converted to its out-of-the-money counterpart
    by subtracting the discounted forward intrinsic, so the solver always
    works on time value. The equation ``b(x, s) = beta`` in the total vol
    ``s = sigma sqrt(T)`` is solved by safeguarded Halley iteration on one
    of three transformed objectives (``1/ln b`` below the inflection point
    ``s_c = sqrt(2|x|)``, ``ln b`` or ``ln(b_max - b)`` above it), from an
    asymptotic initial guess, with a relative stopping rule. The same
    algorithm runs in the C++ and Rust engines (``_iv_core.py``).

    Accuracy: the recovered vol is within ``max(eps * price / vega,
    1e-12 * sigma)`` of the true root over T in [1e-7, 50], sigma in
    [1e-4, 5] and |ln(F/K)| up to 5; at most 10 evaluations, typically 2-4.
    For a deep in-the-money quote the price itself carries the
    information limit: with S=100, K=90, T=0.02, sigma=10% one ulp of the
    call price (1.8e-15) moves the vol by about 5e-4, so the returned vol
    reprices the quote to within a few ulps but cannot be closer to 10%
    than that. Quote the out-of-the-money side when the vol matters.

    Parameters
    ----------
    price : float
        Observed option premium, currency units. Must lie strictly between
        the no-arbitrage bounds (discounted intrinsic on the forward, and
        ``S exp(-qT)`` for calls / ``K exp(-rT)`` for puts).
    S, K, T, r, q : float
        As in :func:`bs_price`. ``S``, ``K`` and ``T`` must be strictly positive.
    option_type : {"call", "put"}
        Option payoff direction.
    tol, max_iter, sigma_lo, sigma_hi
        Deprecated and ignored. The solver has no tuning parameters.

    Returns
    -------
    float
        Annualised implied volatility.

    Raises
    ------
    ValueError
        If ``price`` is at or below the ``sigma -> 0`` bound or at or above
        the ``sigma -> inf`` bound, if ``T == 0``, if any input is invalid,
        or if the solver fails to converge.
    """
    for name, value in (("tol", tol), ("max_iter", max_iter),
                        ("sigma_lo", sigma_lo), ("sigma_hi", sigma_hi)):
        if value is not None:
            warnings.warn(
                f"implied_vol({name}=...) is deprecated and ignored; "
                "the solver has no tuning parameters",
                DeprecationWarning,
                stacklevel=2,
            )
    validate_inputs(S, K, T, 0.0, option_type)
    _check_rates(r=r, q=q)
    if not math.isfinite(price):
        raise ValueError(f"price must be finite, got {price!r}")
    if T <= 0.0:
        raise ValueError("implied_vol requires T > 0 (option already expired)")
    if S <= 0.0 or K <= 0.0:
        raise ValueError("implied_vol requires S > 0 and K > 0")

    df = _exp(-r * T, "implied_vol")
    forward = _finite(S * _exp((r - q) * T, "implied_vol"), "forward")
    lower = bs_price(S, K, T, r, 0.0, q, option_type)  # discounted fwd intrinsic
    upper = S * _exp(-q * T, "implied_vol") if option_type == "call" else K * df
    if price <= lower:
        raise ValueError(
            f"price {price!r} is at or below the sigma->0 arbitrage bound "
            f"{lower:.10g}; implied vol is undefined"
        )
    if price >= upper:
        raise ValueError(
            f"price {price!r} is at or above the sigma->inf bound {upper:.10g}; "
            "implied vol is undefined"
        )
    return black_implied_vol(price, forward, K, T, df, option_type == "call")
