"""Monte Carlo pricing under the SAME model assumptions as the closed form.

This is a validation tool, not an alternative model: if we simulate GBM
under the risk-neutral measure and discount the average payoff, the
estimate must converge to the Black-Scholes price at rate
O(1/sqrt(n)). Agreement between two independent implementations of the
same model (one closed-form, one simulated, sharing no code) is strong
evidence both are coded correctly -- see ``docs/VALIDATION.md`` for the
measured convergence rate.

The terminal price under risk-neutral GBM has a closed form,

    S_T = S0 * exp( (r - sigma^2/2) T + sigma sqrt(T) Z ),  Z ~ N(0,1)

so no path discretisation (and no discretisation error) is needed for a
European payoff -- this is an *exact* one-step simulation of the
terminal distribution, not an Euler-Maruyama path scheme, so all of the
estimator's error is statistical (reported as a standard error) with no
discretisation bias to separate out.

Antithetic variates: for each Z we also use -Z. The two estimates are
negatively correlated (the payoff is monotone in Z for a call), which
reduces variance at no extra model risk and no extra model evaluations
beyond the RNG draws.

This module is the one dependency exception in an otherwise
scipy/numpy-free project: ``numpy`` is used here for vectorised random
sampling and array reductions, which is a genuine and justified
dependency (batch-generating and reducing a million-path array without
it would be both slow and much harder to read). ``math`` is used for
the scalar drift/discount terms to keep them exact and dependency-free.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["mc_call_price"]


def mc_call_price(S: float, K: float, r: float, sigma: float, T: float,
                  n_paths: int = 100_000, seed: int | None = 0,
                  antithetic: bool = True) -> tuple[float, float]:
    """Monte Carlo price of a European call under risk-neutral GBM.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    r : float
        Continuously compounded, annualised risk-free rate.
    sigma : float
        Annualised volatility of log-returns.
    T : float
        Time to expiry in years.
    n_paths : int, default 100_000
        Number of simulated terminal prices. When ``antithetic=True``
        this is rounded down to an even number of independent draws
        (``n_paths // 2`` draws, each mirrored), so the realised sample
        size may be one less than requested for odd ``n_paths``.
    seed : int or None, default 0
        Seed for ``numpy.random.default_rng``. Every stochastic routine
        in this project takes an explicit seed so results are
        reproducible; pass ``None`` for non-deterministic draws.
    antithetic : bool, default True
        If True, pair each standard normal draw ``Z`` with ``-Z`` to
        reduce variance via antithetic sampling.

    Returns
    -------
    tuple[float, float]
        ``(price_estimate, standard_error)``. The standard error is the
        sample standard deviation of the discounted payoffs divided by
        ``sqrt(n)``, so it already reflects any variance reduction from
        antithetic sampling.
    """
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
