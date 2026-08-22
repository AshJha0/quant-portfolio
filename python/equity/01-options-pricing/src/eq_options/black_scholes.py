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
from typing import Literal

from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]

__all__ = [
    "bs_price",
    "d1_d2",
    "implied_vol",
    "validate_inputs",
    "intrinsic_value",
    "forward_price",
]

_SQRT_2PI = math.sqrt(2.0 * math.pi)


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
    if T == 0.0:
        return intrinsic_value(S, K, option_type)
    if K == 0.0:
        # Zero-strike call is a (dividend-adjusted) forward on the stock.
        return S * math.exp(-q * T) if option_type == "call" else 0.0
    if S == 0.0:
        return 0.0 if option_type == "call" else K * math.exp(-r * T)
    if sigma == 0.0:
        forward = forward_price(S, T, r, q)
        sign = 1.0 if option_type == "call" else -1.0
        return math.exp(-r * T) * max(sign * (forward - K), 0.0)

    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_s = S * math.exp(-q * T)
    disc_k = K * math.exp(-r * T)
    if option_type == "call":
        return disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
    return disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """Analytic vega dV/dsigma (per unit of vol, i.e. per 100 vol points)."""
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * math.exp(-0.5 * d1 * d1) / _SQRT_2PI * math.sqrt(T)


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    tol: float = 1e-12,
    max_iter: int = 100,
    sigma_lo: float = 1e-9,
    sigma_hi: float = 10.0,
) -> float:
    """Implied Black-Scholes volatility via bracketed Newton with bisection fallback.

    Newton iterations use analytic vega; every step is kept inside a
    maintained bracket, and if Newton stalls (tiny vega deep ITM/OTM, or a
    step outside the bracket) the algorithm falls back to bisection /
    Brent's method, so it is robust across moneyness 0.5x-2.0x and
    expiries from days to years. The Newton loop always finishes with a
    Brent refinement on its final bracket rather than returning as soon as
    the price residual is within ``tol`` -- see the "flat vega" note below.

    Known hard regime: very long-dated *and* very high vol (e.g. T > 10y
    with sigma > 200%) pushes ``|d1|``, ``|d2|`` large enough that vega
    ``~ S sqrt(T) phi(d1)`` underflows towards zero and the price sits
    within double-precision noise of the ``sigma -> inf`` arbitrage bound.
    There the price-to-vol map is genuinely ill-conditioned (a tiny price
    residual corresponds to a large sigma residual): recovered vol is only
    accurate to the 1e-4-1e-3 level in that corner rather than the 1e-8
    achieved elsewhere, no matter how the residual tolerance is set. This
    is a property of the inverse problem itself, not a fixable solver bug.

    Parameters
    ----------
    price : float
        Observed option premium, currency units. Must lie strictly between
        the no-arbitrage bounds (discounted intrinsic on the forward, and
        ``S exp(-qT)`` for calls / ``K exp(-rT)`` for puts).
    S, K, T, r, q : float
        As in :func:`bs_price`. ``T`` must be strictly positive.
    option_type : {"call", "put"}
        Option payoff direction.
    tol : float
        Absolute price tolerance for convergence.
    max_iter : int
        Maximum Newton iterations before switching to Brent.
    sigma_lo, sigma_hi : float
        Initial volatility bracket, annualised.

    Returns
    -------
    float
        Annualised implied volatility.

    Raises
    ------
    ValueError
        If ``price`` is at or below the ``sigma -> 0`` lower bound
        (discounted forward intrinsic), above the upper bound, if ``T == 0``,
        or if any underlying input is invalid.
    """
    validate_inputs(S, K, T, 0.0, option_type)
    if not math.isfinite(price):
        raise ValueError(f"price must be finite, got {price!r}")
    if T <= 0.0:
        raise ValueError("implied_vol requires T > 0 (option already expired)")
    if S <= 0.0 or K <= 0.0:
        raise ValueError("implied_vol requires S > 0 and K > 0")

    lower = bs_price(S, K, T, r, 0.0, q, option_type)  # discounted fwd intrinsic
    upper = S * math.exp(-q * T) if option_type == "call" else K * math.exp(-r * T)
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

    def objective(sig: float) -> float:
        return bs_price(S, K, T, r, sig, q, option_type) - price

    lo, hi = sigma_lo, sigma_hi
    f_lo, f_hi = objective(lo), objective(hi)
    # Expand the top of the bracket if needed (extremely high premiums).
    while f_hi < 0.0 and hi < 1e3:
        hi *= 2.0
        f_hi = objective(hi)
    if f_lo > 0.0 or f_hi < 0.0:
        raise ValueError("failed to bracket implied volatility in [1e-9, 1e3]")

    # Bracketed Newton: start from the midpoint, never leave [lo, hi].
    sigma = 0.5 * (lo + hi)
    for _ in range(max_iter):
        diff = objective(sigma)
        if abs(diff) < tol:
            break
        if diff > 0.0:
            hi = sigma
        else:
            lo = sigma
        vega = _bs_vega(S, K, T, r, sigma, q)
        if vega > 1e-14:
            step = diff / vega
            candidate = sigma - step
        else:
            candidate = math.nan
        if not (lo < candidate < hi):
            candidate = 0.5 * (lo + hi)  # bisection fallback
        if abs(candidate - sigma) < 1e-16:
            break
        sigma = candidate

    # Always finish with Brent on the maintained bracket, even when Newton
    # exited on the `tol` price-residual check above. That check alone is
    # not a reliable stopping rule: in a flat-vega region (e.g. very
    # long-dated + very high vol, where d1/d2 blow up and vega ~ exp(-d1^2/2)
    # underflows towards the bracket's arbitrage bound) the price can sit
    # within `tol` of the target while sigma is still off by whole vol
    # points, because a tiny price residual maps through a tiny vega to a
    # large sigma residual. Brent on the bracket costs at most a few extra
    # evaluations in the well-conditioned case and recovers full bracket
    # precision in the ill-conditioned one.
    return float(brentq(objective, lo, hi, xtol=1e-16, rtol=8.9e-16, maxiter=200))
