"""Cox-Ross-Rubinstein binomial lattice: the one shared backward induction.

Plain idea
----------
Cut ``[0, T]`` into ``n`` steps. In each step the spot moves up by ``u`` or
down by ``d = 1/u``; the up-probability ``p`` is set so the expected
growth matches the risk-neutral drift. Start from the payoff at expiry and
discount expected values back one step at a time; for American exercise,
take the larger of that value and immediate exercise at every node.

Formal statement
----------------
With ``dt = T/n``, ``u = exp(sigma sqrt(dt))``, ``d = 1/u`` and
``p = (exp((r - q) dt) - d) / (u - d)`` (Cox, Ross and Rubinstein 1979,
"Option pricing: a simplified approach", J. Financial Economics 7,
eqs. 2-3, with Merton's continuous yield ``q``), the node values are
``V_i,k = e^{-r dt} (p V_{i+1,k+1} + (1 - p) V_{i+1,k})``, maxed with the
intrinsic value for American exercise. Node ``k`` of step ``i`` sits at
spot ``S u^(2k - i)``; the spots ``S u^m``, ``m = -n..n``, are computed once
as ``exp(ln S + m sigma sqrt(dt))``, so the induction does no
transcendental work. FX: pass ``r = r_d`` and ``q = r_f``.

The lattice converges to Black-Scholes at ``O(1/n)`` with an odd/even
oscillation (Leisen and Reimer 1996, "Binomial models for option
valuation", Applied Mathematical Finance 3(4)).

Worked example
--------------
>>> round(crr_lattice(100.0, 100.0, 1.0, 0.05, 0.02, 0.2, 1.0, False, 1), 6)
11.073541

One step: ``u = e^0.2``, ``p = (e^0.03 - e^-0.2) / (e^0.2 - e^-0.2)
= 0.5250...``, so the call is ``e^-0.05 p (100 e^0.2 - 100) = 11.0735``.

This module does no input validation beyond the probability check: the
callers (``eq_options.binomial``, ``fx_options.binomial``) validate inputs
and handle the degenerate cases (``T = 0``, ``sigma = 0``, ``S = 0``,
``K = 0``) under their own documented policies.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["crr_lattice"]


def crr_lattice(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    phi: float,
    american: bool,
    n_steps: int,
) -> float:
    """CRR lattice value of a vanilla option.

    Parameters
    ----------
    S, K : float
        Spot and strike, both ``> 0``.
    T : float
        Time to expiry in years, ``> 0``.
    r, q : float
        Continuously compounded discount rate and yield (``r_d``, ``r_f``
        for FX).
    sigma : float
        Annualised volatility, ``> 0``.
    phi : float
        ``+1.0`` for a call, ``-1.0`` for a put.
    american : bool
        Early exercise at every node when true.
    n_steps : int
        Number of time steps, ``>= 1``.

    Returns
    -------
    float
        Present value in the units of ``S``.

    Raises
    ------
    ValueError
        If the risk-neutral probability is outside ``(0, 1)``: ``dt`` is too
        large for the given ``r - q`` and ``sigma``.
    """
    dt = T / n_steps
    log_u = sigma * math.sqrt(dt)
    u = math.exp(log_u)
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(
            f"risk-neutral probability p={p:.6f} outside (0, 1); "
            "increase n_steps or check r, q, sigma"
        )
    disc = math.exp(-r * dt)
    pu, pd = disc * p, disc * (1.0 - p)
    m = np.arange(-n_steps, n_steps + 1)
    spots = np.exp(math.log(S) + m * log_u)
    exercise_value = phi * (spots - K)
    values = np.maximum(exercise_value[0::2], 0.0)  # m = -n, -n+2, ..., n
    for i in range(n_steps - 1, -1, -1):
        values = pu * values[1:] + pd * values[:-1]
        if american:
            np.maximum(values, exercise_value[n_steps - i:n_steps + i + 1:2], out=values)
    return float(values[0])
