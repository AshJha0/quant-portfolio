"""Cox-Ross-Rubinstein binomial tree for European and American options.

Conventions match :mod:`eq_options.black_scholes`: continuously compounded
annualised ``r`` and ``q`` (ACT/365F), ``T`` in years, ``sigma`` annualised.

The backward induction is vectorised over tree nodes with NumPy — the only
Python-level loop is over the ``n_steps`` time slices.

Edge-case policy
----------------
* ``T == 0`` -> intrinsic value.
* ``sigma == 0`` -> the world is deterministic: the European price is the
  discounted forward intrinsic (identical to Black-Scholes), and the
  American price is the maximum over the time grid of the discounted
  intrinsic along the deterministic path ``S exp((r - q) t)``.
* Negative ``S``, ``K``, ``T``, ``sigma`` raise ``ValueError``;
  ``n_steps < 1`` raises ``ValueError``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .black_scholes import OptionType, bs_price, validate_inputs

ExerciseStyle = Literal["european", "american"]

__all__ = ["crr_price", "early_exercise_premium"]


def _deterministic_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: OptionType,
    exercise: ExerciseStyle,
    n_steps: int,
) -> float:
    """Price under ``sigma == 0``: the stock grows deterministically at r - q."""
    if exercise == "european":
        return bs_price(S, K, T, r, 0.0, q, option_type)
    times = np.linspace(0.0, T, n_steps + 1)
    path = S * np.exp((r - q) * times)
    sign = 1.0 if option_type == "call" else -1.0
    payoffs = np.maximum(sign * (path - K), 0.0)
    return float(np.max(np.exp(-r * times) * payoffs))


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    exercise: ExerciseStyle = "european",
    n_steps: int = 500,
) -> float:
    """Cox-Ross-Rubinstein binomial price of a European or American option.

    Uses ``u = exp(sigma sqrt(dt))``, ``d = 1/u`` and risk-neutral
    probability ``p = (exp((r - q) dt) - d) / (u - d)``. Backward induction
    is vectorised over nodes; American options compare continuation value
    with intrinsic at every node.

    Parameters
    ----------
    S : float
        Spot price (currency units), ``S >= 0``.
    K : float
        Strike price (currency units), ``K >= 0``.
    T : float
        Time to expiry in years (ACT/365F), ``T >= 0``.
    r : float
        Continuously compounded annualised risk-free rate. Negative
        rates are supported.
    sigma : float
        Annualised log-return volatility, ``sigma >= 0``.
    q : float
        Continuously compounded annualised dividend yield.
    option_type : {"call", "put"}
        Option payoff direction.
    exercise : {"european", "american"}
        Exercise style.
    n_steps : int
        Number of time steps, ``>= 1``. Convergence to Black-Scholes for
        European options is O(1/n) with an oscillating (odd/even) term.

    Returns
    -------
    float
        Present value in currency units.

    Raises
    ------
    ValueError
        On negative/NaN inputs, invalid ``option_type``/``exercise``, or if
        the risk-neutral probability falls outside (0, 1) — a sign that
        ``dt`` is too large for the given ``r - q`` and ``sigma``.
    """
    validate_inputs(S, K, T, sigma, option_type)
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', got {exercise!r}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")

    sign = 1.0 if option_type == "call" else -1.0
    if T == 0.0:
        return max(sign * (S - K), 0.0)
    if S == 0.0:
        if option_type == "call":
            return 0.0
        return K if exercise == "american" else K * math.exp(-r * T)
    if K == 0.0:
        if option_type == "put":
            return 0.0
        # Zero-strike call: American early exercise is optimal iff q > 0
        # in continuous time; on the tree, exercising at expiry gives the
        # dividend-adjusted forward. American value >= S when q > 0.
        if exercise == "american" and q > 0.0:
            return S
        return S * math.exp(-q * T)
    if sigma == 0.0:
        return _deterministic_price(S, K, T, r, q, option_type, exercise, n_steps)

    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk-neutral probability p={p:.6f} outside (0, 1); "
            "increase n_steps or check r, q, sigma"
        )
    disc = math.exp(-r * dt)
    pu, pd = disc * p, disc * (1.0 - p)

    # Terminal stock prices S * u^j * d^(n-j), j = 0..n, in log space for
    # numerical stability at large n.
    j = np.arange(n_steps + 1)
    log_s = math.log(S) + (2.0 * j - n_steps) * (sigma * math.sqrt(dt))
    values = np.maximum(sign * (np.exp(log_s) - K), 0.0)

    american = exercise == "american"
    log_u = sigma * math.sqrt(dt)
    for i in range(n_steps - 1, -1, -1):
        values = pu * values[1:] + pd * values[:-1]
        if american:
            nodes = np.exp(math.log(S) + (2.0 * np.arange(i + 1) - i) * log_u)
            np.maximum(values, sign * (nodes - K), out=values)
    return float(values[0])


def early_exercise_premium(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "put",
    n_steps: int = 500,
) -> float:
    """American-minus-European value on the same CRR tree.

    Using the *same* tree for both legs cancels the O(1/n) discretisation
    error, so the premium is accurate to much better than either price.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
        As in :func:`crr_price`.
    option_type : {"call", "put"}
        Option payoff direction (puts carry the premium when ``q < r``).
    n_steps : int
        Tree steps shared by both legs.

    Returns
    -------
    float
        Early-exercise premium in currency units; floored at 0 to remove
        residual floating-point noise.
    """
    amer = crr_price(S, K, T, r, sigma, q, option_type, "american", n_steps)
    euro = crr_price(S, K, T, r, sigma, q, option_type, "european", n_steps)
    return max(amer - euro, 0.0)
