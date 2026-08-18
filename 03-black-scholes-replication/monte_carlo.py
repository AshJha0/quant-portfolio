"""
Monte Carlo pricing under the SAME model assumptions as the closed form.

This is a validation tool, not an alternative model: if we simulate
GBM under the risk-neutral measure and discount the average payoff,
the estimate must converge to the Black-Scholes price at rate
O(1/sqrt(n)). Agreement between two independent implementations of
the same model is strong evidence both are coded correctly.

The terminal price under risk-neutral GBM has a closed form,

  S_T = S0 * exp( (r - sigma^2/2) T + sigma sqrt(T) Z ),  Z ~ N(0,1)

so no path discretisation (and no discretisation error) is needed
for a European payoff.

Antithetic variates: for each Z we also use -Z. The two estimates are
negatively correlated, which reduces variance at no extra model risk.
"""
from __future__ import annotations

import math

import numpy as np


def mc_call_price(S: float, K: float, r: float, sigma: float, T: float,
                  n_paths: int = 100_000, seed: int | None = 0,
                  antithetic: bool = True) -> tuple[float, float]:
    """Returns (price_estimate, standard_error)."""
    rng = np.random.default_rng(seed)
    n = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal(n)
    if antithetic:
        z = np.concatenate([z, -z])

    drift = (r - 0.5 * sigma**2) * T
    st = S * np.exp(drift + sigma * math.sqrt(T) * z)
    payoff = np.maximum(st - K, 0.0)
    disc = math.exp(-r * T)

    price = disc * payoff.mean()
    stderr = disc * payoff.std(ddof=1) / math.sqrt(len(payoff))
    return price, stderr
