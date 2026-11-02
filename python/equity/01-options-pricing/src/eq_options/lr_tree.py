"""Leisen-Reimer binomial tree for European and American options.

Plain idea: CRR puts the tree's up/down moves on a fixed grid and lets the
probabilities fall where they may, so the strike lands anywhere between two
nodes and the error oscillates. Leisen-Reimer turns this round. It picks the
probabilities first, from a normal-to-binomial inversion of ``d1`` and ``d2``,
and then solves for the moves so the tree's terminal distribution is
centred on the strike. The European error then falls like ``1/n**2``, not
``1/n``, with no odd/even wobble.

Formal statement (Leisen and Reimer 1996, "Binomial models for option
valuation - examining and improving convergence", Applied Mathematical
Finance 3(4), 319-346, section 3; Peizer-Pratt method 2 inversion, eq. (6)):

* ``n`` is forced odd (``n -> n + 1`` if even).
* ``p' = h(d1)``, ``p = h(d2)``, with
  ``h(z) = 1/2 + sign(z)/2 * sqrt(1 - exp(-(z / (n + 1/3 + 0.1/(n + 1)))**2 * (n + 1/6)))``.
* ``u = exp((r - q) dt) p' / p``, ``d = (exp((r - q) dt) - p u) / (1 - p)``.

Worked example: ``S = K = 100, T = 1, r = 0.05, q = 0, sigma = 0.2``, call,
``n = 101`` gives 10.45055, against Black-Scholes 10.45058 (error 3.4e-5,
and 8.7e-6 at ``n = 201``: the error falls 4x per doubling); CRR at
``n = 101`` is off by 1.7e-2.

Conventions are those of :mod:`eq_options.black_scholes`.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .black_scholes import OptionType, _check_rates, d1_d2, validate_inputs

ExerciseStyle = Literal["european", "american"]

__all__ = ["lr_parameters", "lr_price", "peizer_pratt"]


def peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt method 2 inversion: the binomial probability matching ``N(z)``.

    Leisen and Reimer (1996), eq. (6). ``n`` must be odd.
    """
    if n < 1 or n % 2 == 0:
        raise ValueError(f"n must be a positive odd integer, got {n!r}")
    a = z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))
    root = math.sqrt(-math.expm1(-a * a * (n + 1.0 / 6.0)))
    return 0.5 + math.copysign(0.5, z) * root


def lr_parameters(
    S: float, K: float, T: float, r: float, sigma: float, q: float, n_steps: int
) -> tuple[int, float, float, float]:
    """Return ``(n, u, d, p)`` for the Leisen-Reimer tree (``n`` made odd)."""
    n = n_steps if n_steps % 2 == 1 else n_steps + 1
    dt = T / n
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    p_prime = peizer_pratt(d1, n)
    p = peizer_pratt(d2, n)
    growth = math.exp((r - q) * dt)
    u = growth * p_prime / p
    d = (growth - p * u) / (1.0 - p)
    return n, u, d, p


def lr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    exercise: ExerciseStyle = "european",
    n_steps: int = 201,
) -> float:
    """Leisen-Reimer binomial price of a European or American option.

    Parameters
    ----------
    S, K : float
        Spot and strike (currency units), strictly positive.
    T : float
        Time to expiry in years (ACT/365F), strictly positive.
    r, q : float
        Continuously compounded annualised rate and dividend yield.
    sigma : float
        Annualised log-return volatility, strictly positive.
    option_type : {"call", "put"}
    exercise : {"european", "american"}
    n_steps : int
        Number of steps, ``>= 1``; an even value is raised by one.

    Returns
    -------
    float
        Present value in currency units.

    Raises
    ------
    ValueError
        On invalid inputs, a zero ``S``, ``K``, ``T`` or ``sigma`` (the
        inversion needs a finite ``d1``), or a tree probability outside (0, 1).
    """
    validate_inputs(S, K, T, sigma, option_type)
    _check_rates(r=r, q=q)
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")
    if min(S, K, T, sigma) <= 0.0:
        raise ValueError("lr_price needs S, K, T and sigma strictly positive")

    n, u, d, p = lr_parameters(S, K, T, r, sigma, q, n_steps)
    if not 0.0 < p < 1.0 or not d < u:
        raise ValueError(f"Leisen-Reimer tree invalid: p={p!r}, u={u!r}, d={d!r}")
    disc = math.exp(-r * T / n)
    pu, pd = disc * p, disc * (1.0 - p)
    sign = 1.0 if option_type == "call" else -1.0
    american = exercise == "american"

    j = np.arange(n + 1)
    log_u, log_d = math.log(u), math.log(d)
    values = np.maximum(sign * (S * np.exp(j * log_u + (n - j) * log_d) - K), 0.0)
    for i in range(n - 1, -1, -1):
        values = pu * values[1:] + pd * values[:-1]
        if american:
            k = np.arange(i + 1)
            spots = S * np.exp(k * log_u + (i - k) * log_d)
            np.maximum(values, sign * (spots - K), out=values)
    return float(values[0])
