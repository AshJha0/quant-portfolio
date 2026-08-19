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

That negative correlation is also why the standard error of an
antithetic estimator must **not** be computed by treating the ``2m``
mirrored payoffs as ``2m`` independent observations: they are not
independent, and doing so materially misstates the error bar (on a
100k-path ATM call it overstates it by about a third, i.e. it reports a
less accurate estimate than it actually delivered). The estimator is
really a mean of ``m`` independent *pair averages*, so its standard
error is the sample standard deviation of those pair averages divided by
``sqrt(m)``, which is what this module computes.

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


def _validate(S: float, K: float, sigma: float, T: float, n_paths: int,
              antithetic: bool) -> None:
    """Validate Monte Carlo inputs, mirroring the closed form's contract.

    The simulation would happily run with ``sigma = 0`` or ``T = 0``
    (the terminal price is then deterministic), but the closed form
    deliberately refuses those cases so the caller takes the intrinsic
    -value limit explicitly. Accepting them here would make the two
    implementations disagree about what a valid input is, which is
    exactly the kind of silent divergence this project exists to catch.

    Parameters
    ----------
    S, K, sigma, T : float
        See :func:`mc_call_price`.
    n_paths : int
        Requested number of simulated terminal prices.
    antithetic : bool
        Whether antithetic pairing is enabled.

    Raises
    ------
    ValueError
        If ``S``/``K``/``sigma``/``T`` are not strictly positive, if
        ``n_paths`` is not a positive integer, or if ``antithetic`` is
        requested with ``n_paths < 2`` (a mirrored pair is the smallest
        unit an antithetic estimator can produce; ``n_paths=1`` used to
        yield ``n_paths // 2 == 0`` draws and a silent ``NaN`` price).
    """
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be strictly positive (got S={S}, K={K})")
    if sigma <= 0 or T <= 0:
        raise ValueError(
            f"sigma and T must be strictly positive (got sigma={sigma}, T={T}); "
            "the T->0 and sigma->0 limits are intrinsic value, not a simulation "
            "-- take the limit explicitly at the call site, as the closed form "
            "also requires"
        )
    if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)):
        raise ValueError(f"n_paths must be an int, got {n_paths!r}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if antithetic and n_paths < 2:
        raise ValueError(
            "antithetic sampling needs n_paths >= 2 (one mirrored pair); "
            f"got n_paths={n_paths}. Pass antithetic=False to run a single "
            "unpaired draw."
        )


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
        Number of simulated terminal prices, >= 1 (>= 2 when
        ``antithetic=True``). With ``antithetic=True`` this is rounded
        **down** to an even number: ``n_paths // 2`` independent draws,
        each mirrored, so an odd ``n_paths`` simulates one path fewer
        than requested (e.g. ``n_paths=101`` simulates 100). The
        returned standard error always refers to the realised sample.
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
        ``(price_estimate, standard_error)``.

        Without antithetic sampling the standard error is the usual
        ``std(discounted payoffs, ddof=1) / sqrt(n)``.

        **With** antithetic sampling the estimator is a mean over ``m =
        n_paths // 2`` independent *pairs*, and the standard error is
        computed from the pair averages: ``std(pair means, ddof=1) /
        sqrt(m)``. Treating the ``2m`` mirrored payoffs as independent
        (as an earlier version of this function did) ignores their
        negative correlation and overstates the error by roughly a third
        for an ATM call -- which would make every "within 3 standard
        errors" check in the test suite quietly weaker than it claims.

        ``NaN`` standard error when the realised sample has fewer than 2
        independent units (``ddof=1`` is undefined), i.e. ``n_paths < 2``
        without antithetic sampling, or ``n_paths in (2, 3)`` with it --
        one pair is a point estimate with no measurable spread. The
        price itself is still returned.

    Raises
    ------
    ValueError
        See :func:`_validate` -- non-positive ``S``/``K``/``sigma``/``T``,
        or ``n_paths`` below 1 (below 2 when ``antithetic=True``).
    """
    _validate(S, K, sigma, T, n_paths, antithetic)
    rng = np.random.default_rng(seed)
    m = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal(m)
    if antithetic:
        z = np.concatenate([z, -z])

    drift = (r - 0.5 * sigma**2) * T
    st = S * np.exp(drift + sigma * math.sqrt(T) * z)
    payoff = np.maximum(st - K, 0.0)
    disc = math.exp(-r * T)

    price = disc * float(payoff.mean())
    if antithetic:
        # Independent units are the m PAIRS, not the 2m mirrored draws.
        units = 0.5 * (payoff[:m] + payoff[m:])
    else:
        units = payoff
    if len(units) < 2:
        return price, float("nan")
    stderr = disc * float(units.std(ddof=1)) / math.sqrt(len(units))
    return price, stderr
