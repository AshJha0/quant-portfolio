"""Cox-Ross-Rubinstein binomial tree for FX options.

The foreign interest rate enters exactly like a continuous dividend yield:
risk-neutral drift of the spot under the domestic measure is ``r_d - r_f``,
so the up-move probability is

    p = (e^{(r_d - r_f) dt} - d) / (u - d),   u = e^{sigma sqrt(dt)},  d = 1/u.

Supports European and American exercise.  American FX options trade OTC;
the economically interesting case is an American *call* on a high-yielding
foreign currency (``r_f > r_d``): the foreign carry lost by holding the
option instead of the currency makes early exercise optimal, giving the
American call a strictly positive premium over European — mirroring the
dividend-yield story for equities.
"""

from __future__ import annotations

import math

import numpy as np

from ._common import validate_inputs, validate_option_type
from .garman_kohlhagen import gk_price

__all__ = ["binomial_price", "binomial_convergence"]


def binomial_price(S: float, K: float, T: float, r_d: float, r_f: float,
                   sigma: float, option_type: str, steps: int = 500,
                   exercise: str = "european") -> float:
    """CRR binomial price of an FX option.

    Parameters
    ----------
    S, K, T, r_d, r_f, sigma : float
        As in :func:`fx_options.garman_kohlhagen.gk_price`.
    option_type : str
        ``"call"`` or ``"put"``.
    steps : int
        Number of tree steps, >= 1.
    exercise : str
        ``"european"`` or ``"american"``.

    Returns
    -------
    float
        Price in domestic currency per unit foreign notional.

    Raises
    ------
    ValueError
        On invalid inputs, non-positive ``steps``, unknown ``exercise``,
        or if the tree probability falls outside [0, 1] (time step too
        coarse for the drift/vol combination).
    """
    phi = validate_option_type(option_type)
    validate_inputs(S, K, T, r_d, r_f, sigma)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    if exercise not in ("european", "american"):
        raise ValueError(
            f"exercise must be 'european' or 'american', got {exercise!r}"
        )
    if T == 0.0:
        return max(phi * (S - K), 0.0)
    if sigma == 0.0:
        # Degenerate tree; defer to the analytic limit (European) or
        # deterministic exercise optimisation (American on a drifting spot).
        if exercise == "european":
            return gk_price(S, K, T, r_d, r_f, sigma, option_type)
        drift = r_d - r_f
        times = np.linspace(0.0, T, steps + 1)
        values = np.exp(-r_d * times) * np.maximum(
            phi * (S * np.exp(drift * times) - K), 0.0)
        return float(values.max())

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((r_d - r_f) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"risk-neutral probability {p:.6f} outside [0, 1]; "
            "increase steps (dt too large for |r_d - r_f| vs sigma)"
        )
    disc = math.exp(-r_d * dt)

    j = np.arange(steps + 1)
    spot = S * u ** (2.0 * j - steps)          # terminal nodes, low -> high
    values = np.maximum(phi * (spot - K), 0.0)

    for step in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if exercise == "american":
            spot = S * u ** (2.0 * np.arange(step + 1) - step)
            values = np.maximum(values, phi * (spot - K))
    return float(values[0])


def binomial_convergence(S: float, K: float, T: float, r_d: float,
                         r_f: float, sigma: float, option_type: str,
                         step_grid: tuple[int, ...] = (10, 25, 50, 100,
                                                       200, 400, 800),
                         ) -> list[dict[str, float]]:
    """Convergence table of the European CRR tree towards Garman-Kohlhagen.

    Returns
    -------
    list of dict
        One row per step count: ``{"steps", "tree_price", "gk_price",
        "abs_error"}``.
    """
    analytic = gk_price(S, K, T, r_d, r_f, sigma, option_type)
    rows = []
    for n in step_grid:
        tree = binomial_price(S, K, T, r_d, r_f, sigma, option_type,
                              steps=n, exercise="european")
        rows.append({"steps": float(n), "tree_price": tree,
                     "gk_price": analytic, "abs_error": abs(tree - analytic)})
    return rows
