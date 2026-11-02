"""Black-Scholes-Merton pricing, built from scratch.

This module depends on nothing but the standard library ``math`` module
(``math.erfc`` for the standard normal CDF). No ``scipy.stats.norm``,
no ``numpy``. The point of this project is to re-derive the model from
theory, so the normal CDF is written out here instead of imported.
``math.erfc`` comes from the platform C library. The CDF is written as
``0.5 * erfc(-x / sqrt(2))`` so the left tail keeps full relative
accuracy: the form ``0.5 * (1 + erf(x / sqrt(2)))`` cancels to zero
below about x = -8.3.

This module also intentionally works with **scalar Python floats**, not
NumPy arrays. That is the right trade-off *here*: readability and a
direct, one-line-per-formula mapping to the maths, at the cost of being
slow if you wanted to price an entire chain in a hot loop. A vectorised,
faster sibling of this exact model (batched arrays, analytic
Greeks including vanna/volga, CRR/Black-76/Monte Carlo cross-validation,
C++/Rust performance twins) lives at
``python/equity/01-options-pricing``; use that for anything
performance-sensitive or production-facing. This project is the
from-scratch reference the other one can be checked against in spirit,
not a competing implementation.

Model assumptions (each one fails in practice -- see
``docs/METHODOLOGY.md`` for the full assumptions register with
"what breaks if violated" for each):

  1. The underlying follows geometric Brownian motion with CONSTANT
     volatility sigma:  dS = mu*S*dt + sigma*S*dW.
  2. Constant risk-free rate r, continuous compounding.
  3. Frictionless markets: no transaction costs, continuous trading,
     unlimited borrowing/shorting.
  4. No arbitrage; European exercise; (here) no dividends.

Under these assumptions the option payoff can be replicated by a
continuously rebalanced portfolio of stock and cash, so its price is
the discounted risk-neutral expectation of the payoff::

    C = S0*N(d1) - K*exp(-rT)*N(d2)
    d1 = [ln(S0/K) + (r + sigma^2/2)T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

Interpretation worth knowing: N(d2) is the risk-neutral probability the
option finishes in the money; S0*N(d1) is the discounted expected value
of receiving the stock given exercise.

Conventions: ``r`` is continuously compounded and annualised; ``T`` is
time to expiry in years (ACT/365F); ``sigma`` is annualised volatility
of log-returns; no dividends (q=0 throughout -- see METHODOLOGY.md for
the one-line extension that would add a continuous yield).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "Greeks",
    "call_greeks",
    "call_price",
    "implied_volatility",
    "put_greeks",
    "put_price",
]


def _norm_cdf(x: float) -> float:
    """Standard normal CDF N(x), computed via ``math.erfc``.

    Parameters
    ----------
    x : float
        Evaluation point.

    Returns
    -------
    float
        P(Z <= x) for Z ~ N(0, 1).

    Notes
    -----
    N(x) = 0.5 * erfc(-x / sqrt(2)). This keeps full relative accuracy
    in the left tail. The algebraically equal 0.5 * (1 + erf(x / sqrt(2)))
    loses all relative accuracy below x = -8 and returns 0 below -8.3.
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF phi(x) = exp(-x^2/2) / sqrt(2*pi).

    Parameters
    ----------
    x : float
        Evaluation point.

    Returns
    -------
    float
        Density of the standard normal at ``x``.
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _discount_factor(r: float, T: float) -> float:
    """``exp(-r*T)``, with an informative error instead of ``OverflowError``.

    A sufficiently negative ``r*T`` (e.g. ``r=-10`` over ``T=100``) makes
    ``exp(-r*T)`` exceed the largest representable double, and ``math.exp``
    raises a bare ``OverflowError: math range error`` that says nothing
    about which input caused it. Such inputs are economically absurd
    rather than merely extreme, so the right response is a clear refusal.

    Parameters
    ----------
    r : float
        Continuously compounded, annualised rate (may be negative).
    T : float
        Time to expiry in years.

    Returns
    -------
    float
        The discount factor ``exp(-r*T)``.

    Raises
    ------
    ValueError
        If ``-r*T`` is large enough to overflow a double (about 709.78).
    """
    exponent = -r * T
    if exponent > 709.0:
        raise ValueError(
            f"discount factor exp({exponent:.1f}) overflows double precision "
            f"(r={r}, T={T}); a rate/maturity combination this extreme is "
            "outside the representable range, not merely an unusual market"
        )
    return math.exp(exponent)


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    """Compute the Black-Scholes ``d1`` and ``d2`` terms.

    Parameters
    ----------
    S : float
        Spot price of the underlying (must be > 0).
    K : float
        Strike price (must be > 0).
    r : float
        Continuously compounded, annualised risk-free rate. May be zero
        or negative -- the closed form does not require r > 0.
    sigma : float
        Annualised volatility of log-returns. Must be > 0.
    T : float
        Time to expiry in years. Must be > 0.

    Returns
    -------
    tuple[float, float]
        ``(d1, d2)``.

    Raises
    ------
    ValueError
        If any input is NaN/infinite, or if ``sigma <= 0`` or ``T <= 0``.
        The latter two are boundary cases with
        well-defined economic limits (intrinsic value) that must be
        handled by the caller explicitly rather than silently by
        dividing by zero here -- see ``docs/VALIDATION.md`` edge-case
        section.
    """
    # Finiteness first: every comparison with NaN is False, so the sign
    # guards below would silently accept a NaN spot / vol and return NaN.
    for name, value in (("S", S), ("K", K), ("r", r), ("sigma", sigma), ("T", T)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if sigma <= 0 or T <= 0:
        raise ValueError(
            f"sigma and T must be strictly positive (got sigma={sigma}, T={T}); "
            "the T->0 and sigma->0 limits are intrinsic value, not a division "
            "by zero -- take the limit explicitly at the call site."
        )
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be strictly positive (got S={S}, K={K})")
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """European call price under Black-Scholes-Merton.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    r : float
        Continuously compounded, annualised risk-free rate (may be
        zero or negative).
    sigma : float
        Annualised volatility of log-returns (must be > 0).
    T : float
        Time to expiry in years (must be > 0).

    Returns
    -------
    float
        Call price ``C = S*N(d1) - K*exp(-rT)*N(d2)``.

    Raises
    ------
    ValueError
        If ``sigma <= 0``, ``T <= 0``, or ``S``/``K`` are non-positive.
    """
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return S * _norm_cdf(d1) - K * _discount_factor(r, T) * _norm_cdf(d2)


def put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """European put price under Black-Scholes-Merton.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    r : float
        Continuously compounded, annualised risk-free rate (may be
        zero or negative).
    sigma : float
        Annualised volatility of log-returns (must be > 0).
    T : float
        Time to expiry in years (must be > 0).

    Returns
    -------
    float
        Put price ``P = K*exp(-rT)*N(-d2) - S*N(-d1)``. By construction
        this satisfies put-call parity ``C - P = S - K*exp(-rT)``
        exactly (to floating-point precision), since both formulas
        share the same ``d1``/``d2``.

    Raises
    ------
    ValueError
        If ``sigma <= 0``, ``T <= 0``, or ``S``/``K`` are non-positive.
    """
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return K * _discount_factor(r, T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


@dataclass
class Greeks:
    """Analytic Black-Scholes Greeks for a single option.

    Attributes
    ----------
    delta : float
        dV/dS -- hedge ratio (shares of underlying per option).
    gamma : float
        d2V/dS2 -- convexity of delta; identical for calls and puts at
        the same strike/expiry (put-call parity has zero second
        derivative in S).
    vega : float
        dV/dsigma, per 1.00 (100 vol points) of volatility. Divide by
        100 for the "per 1 vol point" convention some desks quote.
        Identical for calls and puts (parity has no sigma dependence).
    theta : float
        dV/dt, per YEAR, where t is calendar time (so as time passes
        and T = expiry - t shrinks, theta is the rate of value decay).
        Divide by 365 for a per-calendar-day figure. Negative for a
        long call in the usual (r >= 0) regime; puts can have positive
        theta deep ITM under high r.
    rho : float
        dV/dr, per 1.00 (100%) of the risk-free rate.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def call_greeks(S: float, K: float, r: float, sigma: float, T: float) -> Greeks:
    """Analytic Greeks for a European call.

    Parameters
    ----------
    S, K, r, sigma, T : float
        See :func:`call_price`.

    Returns
    -------
    Greeks
        delta, gamma, vega, theta, rho for the call. See
        ``docs/VALIDATION.md`` for finite-difference cross-checks.
    """
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    sqrtT = math.sqrt(T)
    disc = _discount_factor(r, T)
    return Greeks(
        delta=_norm_cdf(d1),
        gamma=_norm_pdf(d1) / (S * sigma * sqrtT),
        vega=S * _norm_pdf(d1) * sqrtT,
        theta=(-S * _norm_pdf(d1) * sigma / (2 * sqrtT)
               - r * K * disc * _norm_cdf(d2)),
        rho=K * T * disc * _norm_cdf(d2),
    )


def put_greeks(S: float, K: float, r: float, sigma: float, T: float) -> Greeks:
    """Analytic Greeks for a European put, derived via put-call parity.

    Rather than re-deriving each put Greek from the put pricing formula
    from scratch, this function differentiates the model-free parity
    identity ``C - P = S - K*exp(-rT)`` with respect to each input and
    solves for the put Greek in terms of the call Greek. This is both
    less code to get wrong and a second, independent-in-spirit way of
    arriving at the put formulas (parity, rather than direct
    differentiation of ``put_price``) -- if the two ever disagreed it
    would indicate a bug, which is exactly the kind of cross-check this
    project is built around (see ``tests/test_greeks.py``).

    Parity relations used (each obtained by differentiating
    ``C - P = S - K*exp(-rT)`` with respect to the named variable and
    rearranging for the put term):

    - ``delta_put = delta_call - 1``
      (d/dS of both sides: dC/dS - dP/dS = 1)
    - ``gamma_put = gamma_call``
      (d2/dS2 of both sides: the RHS is linear in S, so its second
      derivative is zero -- gamma is identical for calls and puts)
    - ``vega_put = vega_call``
      (d/dsigma of both sides: the RHS does not depend on sigma)
    - ``theta_put = theta_call + r*K*exp(-rT)``
      (d/dT of both sides, then flip sign for calendar-time theta:
      d/dT[K*exp(-rT)] = -r*K*exp(-rT), so dP/dT = dC/dT - r*K*exp(-rT),
      and theta = -dV/dT)
    - ``rho_put = rho_call - K*T*exp(-rT)``
      (d/dr of both sides: d/dr[K*exp(-rT)] = -T*K*exp(-rT), so
      dP/dr = dC/dr - T*K*exp(-rT))

    Parameters
    ----------
    S, K, r, sigma, T : float
        See :func:`call_price`.

    Returns
    -------
    Greeks
        delta, gamma, vega, theta, rho for the put.
    """
    c = call_greeks(S, K, r, sigma, T)
    disc_K = K * _discount_factor(r, T)
    return Greeks(
        delta=c.delta - 1.0,
        gamma=c.gamma,
        vega=c.vega,
        theta=c.theta + r * disc_K,
        rho=c.rho - T * disc_K,
    )


def implied_volatility(price: float, S: float, K: float, r: float, T: float,
                       tol: float = 1e-8, max_iter: int = 100,
                       option_type: str = "call") -> float:
    """Implied volatility of a European call or put.

    The quote is first mapped to the out-of-the-money side by put-call
    parity (``P = C - S + K exp(-rT)``). An in-the-money option's price is
    mostly intrinsic value, so inverting it directly gives a vol that
    matches the price to tolerance but carries no information. The OTM
    price holds the same time value without the intrinsic part.

    The OTM price is then inverted by Newton-Raphson on vega, with a
    bisection fallback whose upper bracket doubles from 5.0 until it
    spans the target.

    If the time value is below what double precision can resolve
    (``64 * eps * max(S, K)``), the vol is not determined by the quote.
    The function then warns (``RuntimeWarning``) and returns ``nan``.

    Parameters
    ----------
    price : float
        Observed option price.
    S, K, r, T : float
        See :func:`call_price`.
    tol : float, default 1e-8
        Absolute cap on the OTM price residual of the Newton stage. The
        stage also requires the residual to be below
        ``1e-10 * OTM price + 64 eps max(S, K)``, so a small OTM price is
        still matched to about ten significant figures.
    max_iter : int, default 100
        Newton iterations before falling back to bisection.
    option_type : {"call", "put"}, default "call"
        Which option ``price`` refers to.

    Returns
    -------
    float
        Implied annualised volatility, or ``nan`` (with a warning) when
        the quote has no resolvable time value.

    Raises
    ------
    ValueError
        For non-finite inputs, non-positive ``S``, ``K`` or ``T``, an
        unknown ``option_type``, or a price outside the no-arbitrage
        bounds (call: ``max(S - K e^{-rT}, 0) <= C <= S``; put:
        ``max(K e^{-rT} - S, 0) <= P <= K e^{-rT}``).
    """
    import warnings

    for name, value in (("price", price), ("S", S), ("K", K), ("r", r), ("T", T)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be strictly positive (got S={S}, K={K})")
    if T <= 0:
        raise ValueError(
            f"T must be strictly positive (got T={T}); at expiry the price is "
            "intrinsic value and carries no volatility information at all"
        )
    kdf = K * _discount_factor(r, T)
    if option_type == "call":
        lower, upper = max(S - kdf, 0.0), S
    else:
        lower, upper = max(kdf - S, 0.0), kdf
    if price < lower - 1e-12 or price > upper + 1e-12:
        raise ValueError(
            f"price {price} violates no-arbitrage bounds "
            f"[{lower}, {upper}] for {option_type} S={S}, K={K}, r={r}, T={T}"
        )

    # Map to the OTM side by parity.
    use_put = kdf < S  # call is ITM when S > K e^{-rT}
    if option_type == "call":
        otm_price = price - S + kdf if use_put else price
    else:
        otm_price = price if use_put else price + S - kdf
    pricer = put_price if use_put else call_price
    otm_upper = kdf if use_put else S

    resolution = 64.0 * 2.220446049250313e-16 * max(S, K)
    if otm_price <= resolution:
        warnings.warn(
            f"time value {otm_price:.3e} is below numerical resolution "
            f"{resolution:.1e}; implied vol is not determined, returning nan",
            RuntimeWarning, stacklevel=2,
        )
        return math.nan
    if otm_price >= otm_upper - resolution:
        warnings.warn(
            "price is at the upper no-arbitrage bound; implied vol is "
            "unbounded, returning nan",
            RuntimeWarning, stacklevel=2,
        )
        return math.nan

    sigma = 0.2
    for _ in range(max_iter):
        diff = pricer(S, K, r, sigma, T) - otm_price
        if abs(diff) <= min(tol, 1e-10 * otm_price + resolution):
            return sigma
        vega = call_greeks(S, K, r, sigma, T).vega  # same for put and call
        if vega < 1e-10:
            break
        sigma = max(1e-6, sigma - diff / vega)

    lo, hi = 1e-6, 5.0
    for _ in range(20):
        if pricer(S, K, r, hi, T) >= otm_price:
            break
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if pricer(S, K, r, mid, T) > otm_price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
