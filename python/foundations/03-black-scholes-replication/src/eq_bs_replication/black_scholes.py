"""Black-Scholes-Merton pricing, built from scratch.

This module deliberately depends on nothing but the standard library
``math`` module (specifically ``math.erf`` for the standard normal CDF).
No ``scipy.stats.norm``, no ``numpy``. That is a documented methodology
decision (see ``docs/METHODOLOGY.md``), not an oversight: the point of
this project is to *prove out* the model by re-deriving it from theory,
and importing ``scipy.stats.norm.cdf`` would let the normal-CDF piece of
the derivation go unexamined. ``math.erf`` is itself a from-scratch
numerical primitive (Abramowitz-Stegun-style rational approximation
under the hood in CPython), so building on it -- rather than on a
pre-packaged option-pricing routine -- keeps the whole chain from "why
does the price come out this way" down to a single well-understood
special function.

This module also intentionally works with **scalar Python floats**, not
NumPy arrays. That is the right trade-off *here*: readability and a
direct, one-line-per-formula mapping to the maths, at the cost of being
slow if you wanted to price an entire chain in a hot loop. A vectorised,
production-grade sibling of this exact model (batched arrays, analytic
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
    "call_price",
    "put_price",
    "call_greeks",
    "put_greeks",
    "implied_volatility",
]


def _norm_cdf(x: float) -> float:
    """Standard normal CDF N(x), computed via ``math.erf``.

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
    N(x) = 0.5 * (1 + erf(x / sqrt(2))). No scipy needed for pricing --
    this is the entire dependency-minimalism design decision in one
    line.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


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
        If ``sigma <= 0`` or ``T <= 0``. Both are boundary cases with
        well-defined economic limits (intrinsic value) that must be
        handled by the caller explicitly rather than silently by
        dividing by zero here -- see ``docs/VALIDATION.md`` edge-case
        section.
    """
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
                       tol: float = 1e-8, max_iter: int = 100) -> float:
    """Invert the call formula for sigma via Newton-Raphson.

    Vega is the derivative of price w.r.t. sigma, so the Newton update
    is ``sigma -= (model_price - market_price) / vega``.

    Falls back to bisection when vega is tiny (deep ITM/OTM) or when
    Newton fails to converge in ``max_iter`` steps. The bisection
    bracket starts at ``[1e-6, 5.0]`` (0.0001% to 500% volatility) and
    is **doubled upward** until it contains the target price, so a quote
    implying a volatility above 500% is still inverted rather than
    silently pinned to the bracket's edge.

    Conditioning -- the part that matters more than the algorithm
    ------------------------------------------------------------------
    Implied volatility is only as well-determined as vega is large.
    Near either no-arbitrage bound, vega collapses toward zero and a
    whole range of volatilities reprices the option to within any
    reasonable tolerance:

    - **At (or just above) intrinsic value**, the true implied vol is 0,
      but ``|dPrice/dsigma|`` is so small that the returned sigma can be
      of order 1e-2 while still matching the input price to ``1e-8``.
      The number is *correct to tolerance* and *meaningless as a vol*.
    - **At the upper bound ``C = S``**, the true implied vol is infinite;
      the routine returns whatever large finite value first matches to
      tolerance.

    Both are documented in ``docs/VALIDATION.md`` and unit-tested. A desk
    would not quote either: the practical rule is that implied vol from a
    quote whose vega is below some floor (a few cents per vol point) is
    not information, and such strikes are dropped from surface fits
    rather than fitted with a wide error bar.

    Parameters
    ----------
    price : float
        Observed (market) call price to invert.
    S, K, r, T : float
        See :func:`call_price`.
    tol : float, default 1e-8
        Convergence tolerance on the **price residual** (not on sigma)
        for the Newton-Raphson stage. See the conditioning note above:
        a tight price tolerance does not imply a tight vol tolerance.
    max_iter : int, default 100
        Maximum Newton-Raphson iterations before falling back to
        bisection.

    Returns
    -------
    float
        Implied annualised volatility sigma such that
        ``call_price(S, K, r, sigma, T) == price`` (to tolerance).

    Raises
    ------
    ValueError
        If ``price`` violates the model-free no-arbitrage bounds for a
        call, ``max(S - K*exp(-rT), 0) <= C <= S``. A price outside
        these bounds cannot correspond to any volatility, so refusing
        to invent one is intentional (see ``docs/VALIDATION.md``
        numerical-limits section). Also raised for non-positive ``S``,
        ``K`` or ``T``, and for rate/maturity combinations whose
        discount factor overflows.
    """
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be strictly positive (got S={S}, K={K})")
    if T <= 0:
        raise ValueError(
            f"T must be strictly positive (got T={T}); at expiry the price is "
            "intrinsic value and carries no volatility information at all"
        )
    # No-arbitrage bounds for a call: max(S - K e^{-rT}, 0) <= C <= S
    intrinsic = max(S - K * _discount_factor(r, T), 0.0)
    if price < intrinsic - 1e-12 or price > S + 1e-12:
        raise ValueError(
            f"price {price} violates no-arbitrage bounds "
            f"[{intrinsic}, {S}] for S={S}, K={K}, r={r}, T={T}"
        )

    sigma = 0.2  # standard starting guess
    for _ in range(max_iter):
        diff = call_price(S, K, r, sigma, T) - price
        if abs(diff) < tol:
            return sigma
        vega = call_greeks(S, K, r, sigma, T).vega
        if vega < 1e-10:
            break  # Newton unreliable -> bisection below
        sigma = max(1e-6, sigma - diff / vega)

    # Bisection fallback. Expand the upper bracket until it spans the
    # target price, so an extreme quote is inverted rather than clamped
    # to an arbitrary 500% ceiling. The call price is increasing in sigma
    # and bounded above by S, so this terminates unless price == S (which
    # the bound check above has already narrowed to the degenerate case).
    lo, hi = 1e-6, 5.0
    for _ in range(20):
        if call_price(S, K, r, hi, T) >= price:
            break
        lo, hi = hi, hi * 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if call_price(S, K, r, mid, T) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
