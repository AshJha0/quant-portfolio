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
from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qp_core.black_scholes import carry_d1_d2, carry_discounts, carry_value, finite, safe_exp
from qp_core.iv_core import black_implied_vol
from qp_core.normal import norm_cdf

OptionType = Literal["call", "put"]

__all__ = [
    "bs_price",
    "d1_d2",
    "forward_price",
    "implied_vol",
    "intrinsic_value",
    "validate_inputs",
]

# Standard normal CDF via ``erfc``: full relative precision in the lower tail.
_ncdf = norm_cdf


def _check_rates(**rates: float) -> None:
    """Raise ``ValueError`` unless every rate is a finite real number."""
    for name, value in rates.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")


# Overflow-to-ValueError exp and the finite-result guard live in qp_core.
_exp = safe_exp
_finite = finite


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
    return carry_d1_d2(S, K, T, r, q, sigma)


def _any_array(*args: object) -> bool:
    """True when any argument is an array or a list/tuple (the array path)."""
    return any(isinstance(a, (np.ndarray, list, tuple)) for a in args)


@overload
def bs_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
             option_type: OptionType = "call") -> float: ...
@overload
def bs_price(S: NDArray[Any] | Sequence[float],
             K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike,
             q: ArrayLike = 0.0, option_type: str | ArrayLike = "call") -> NDArray[np.float64]: ...
def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str | ArrayLike = "call",
) -> float | NDArray[np.float64]:
    """Black-Scholes-Merton price of a European option with dividend yield.

    Scalars in, a float out. If any argument is a NumPy array, list or
    tuple, the call goes to :func:`eq_options.arrays.bs_price_array`
    (broadcasting, the same domain and boundary rules) and returns an
    array. Per element the two agree within 1e-15 * max(1, S, K)
    (``tests/test_bs_array_tolerant.py``; measured 5.2e-16 on 20,000 random
    contracts). For a type checker the array overload needs ``S`` to be an
    array or a sequence; for other mixes call ``bs_price_array`` directly.

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
    if _any_array(S, K, T, r, sigma, q, option_type):
        from .arrays import bs_price_array

        return bs_price_array(S, K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
    return _bs_price_scalar(S, K, T, r, sigma, q, option_type)  # type: ignore[arg-type]


def _bs_price_scalar(S: float, K: float, T: float, r: float, sigma: float, q: float,
                     option_type: OptionType) -> float:
    validate_inputs(S, K, T, sigma, option_type)
    _check_rates(r=r, q=q)
    if T == 0.0:
        return intrinsic_value(S, K, option_type)
    if K == 0.0:
        # Zero-strike call is a (dividend-adjusted) forward on the stock.
        return _finite(S * _exp(-q * T, "bs_price"), "bs_price") if option_type == "call" else 0.0
    if S == 0.0:
        return 0.0 if option_type == "call" else _finite(K * _exp(-r * T, "bs_price"), "bs_price")
    disc_s, disc_k = carry_discounts(S, K, T, r, q, "bs_price")
    if sigma == 0.0:
        # exp(-rT) max(+-(F - K), 0) written as S e^{-qT} - K e^{-rT}: the
        # forward itself can overflow when (r - q) T > 709.
        sign = 1.0 if option_type == "call" else -1.0
        return _finite(max(sign * (disc_s - disc_k), 0.0), "bs_price")

    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    phi = 1.0 if option_type == "call" else -1.0
    # + 0.0 maps the -0.0 that phi = -1 gives for a put worth exactly zero to
    # +0.0, the value this function has always returned; nothing else moves.
    return _finite(carry_value(disc_s, disc_k, d1, d2, phi) + 0.0, "bs_price")


_IV_DEPRECATED_DEFAULTS = {"tol": 1e-12, "max_iter": 100, "sigma_lo": 1e-9, "sigma_hi": 10.0}


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
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
    algorithm runs in the C++ and Rust engines (``qp_core.iv_core``).

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
