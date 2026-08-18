"""
Black-Scholes-Merton pricing, built from scratch.

Model assumptions (each one fails in practice -- see README):
  1. The underlying follows geometric Brownian motion with CONSTANT
     volatility sigma:  dS = mu*S*dt + sigma*S*dW.
  2. Constant risk-free rate r, continuous compounding.
  3. Frictionless markets: no transaction costs, continuous trading,
     unlimited borrowing/shorting.
  4. No arbitrage; European exercise; (here) no dividends.

Under these assumptions the option payoff can be replicated by a
continuously rebalanced portfolio of stock and cash, so its price is
the discounted risk-neutral expectation of the payoff:

  C = S0*N(d1) - K*exp(-rT)*N(d2)
  d1 = [ln(S0/K) + (r + sigma^2/2)T] / (sigma*sqrt(T))
  d2 = d1 - sigma*sqrt(T)

Interpretation worth knowing: N(d2) is the risk-neutral probability
the option finishes in the money; S0*N(d1) is the discounted expected
value of receiving the stock given exercise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf -- no scipy needed for pricing."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float):
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive")
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


@dataclass
class Greeks:
    delta: float   # dV/dS       -- hedge ratio
    gamma: float   # d2V/dS2     -- convexity, hedge instability
    vega: float    # dV/dsigma   -- per 1.00 of vol (divide by 100 for 1%)
    theta: float   # dV/dt       -- per YEAR (divide by 365 for per-day)
    rho: float     # dV/dr       -- per 1.00 of rate


def call_greeks(S: float, K: float, r: float, sigma: float, T: float) -> Greeks:
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    sqrtT = math.sqrt(T)
    return Greeks(
        delta=_norm_cdf(d1),
        gamma=_norm_pdf(d1) / (S * sigma * sqrtT),
        vega=S * _norm_pdf(d1) * sqrtT,
        theta=(-S * _norm_pdf(d1) * sigma / (2 * sqrtT)
               - r * K * math.exp(-r * T) * _norm_cdf(d2)),
        rho=K * T * math.exp(-r * T) * _norm_cdf(d2),
    )


def implied_volatility(price: float, S: float, K: float, r: float, T: float,
                       tol: float = 1e-8, max_iter: int = 100) -> float:
    """Invert the call formula for sigma via Newton-Raphson.

    Vega is the derivative of price w.r.t. sigma, so the Newton update
    is sigma -= (model_price - market_price) / vega.

    Falls back to bisection when vega is tiny (deep ITM/OTM), where
    Newton becomes unstable.
    """
    # No-arbitrage bounds for a call: max(S - K e^{-rT}, 0) <= C <= S
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    if price < intrinsic - 1e-12 or price > S + 1e-12:
        raise ValueError("price violates no-arbitrage bounds")

    sigma = 0.2  # standard starting guess
    for _ in range(max_iter):
        diff = call_price(S, K, r, sigma, T) - price
        if abs(diff) < tol:
            return sigma
        vega = call_greeks(S, K, r, sigma, T).vega
        if vega < 1e-10:
            break  # Newton unreliable -> bisection below
        sigma = max(1e-6, sigma - diff / vega)

    lo, hi = 1e-6, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if call_price(S, K, r, mid, T) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
