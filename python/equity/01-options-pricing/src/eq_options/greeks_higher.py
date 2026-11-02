"""Third-order and cross Black-Scholes-Merton Greeks: charm, speed, colour, zomma, veta.

Plain idea: a desk hedging delta and gamma also wants to know how those
hedges drift. Charm is how delta moves as a day passes, colour how gamma
moves as a day passes, speed how gamma moves with spot, zomma how gamma
moves with vol, and veta how vega moves as a day passes.

Formal statement (Haug 2007, "The Complete Guide to Option Pricing
Formulas", 2nd ed., sections 2.2-2.4; with continuous yield ``q``).
``phi = phi(d1)``, ``st = sigma sqrt(T)``, and every time derivative is
``d/dt = -d/dT`` per year, matching ``theta`` in :mod:`eq_options.greeks`:

* charm ``dDelta/dt``: call ``q e^{-qT} Phi(d1) - e^{-qT} phi (2(r-q)T - d2 st) / (2 T st)``;
  put ``-q e^{-qT} Phi(-d1) - e^{-qT} phi (2(r-q)T - d2 st) / (2 T st)``.
* speed ``dGamma/dS = -Gamma / S (d1 / st + 1)``.
* colour ``dGamma/dt = e^{-qT} phi / (2 S T st) [2 q T + 1 + (2(r-q)T - d2 st) d1 / st]``.
* zomma ``dGamma/dsigma = Gamma (d1 d2 - 1) / sigma``.
* veta ``dVega/dt = S e^{-qT} phi sqrt(T) [q + (r-q) d1 / st - (1 + d1 d2) / (2T)]``.

Units: per unit of spot, per unit of vol, per year. The tests check every
formula against ``mpmath.diff`` of the price at 30 digits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .black_scholes import OptionType, _ncdf, d1_d2

__all__ = ["HigherGreeks", "higher_greeks"]


@dataclass(frozen=True)
class HigherGreeks:
    """Charm, speed, colour, zomma and veta (units in the module docstring)."""

    charm: float
    speed: float
    colour: float
    zomma: float
    veta: float


def higher_greeks(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
                  option_type: OptionType = "call") -> HigherGreeks:
    """Analytic higher-order Greeks; ``S, K, T, sigma`` strictly positive."""
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    sqrt_t = math.sqrt(T)
    st = sigma * sqrt_t
    dq = math.exp(-q * T)
    phi = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    gamma = dq * phi / (S * st)
    drift = (2.0 * (r - q) * T - d2 * st) / (2.0 * T * st)
    if option_type == "call":
        charm = q * dq * _ncdf(d1) - dq * phi * drift
    else:
        charm = -q * dq * _ncdf(-d1) - dq * phi * drift
    speed = -gamma / S * (d1 / st + 1.0)
    colour = dq * phi / (2.0 * S * T * st) * (
        2.0 * q * T + 1.0 + (2.0 * (r - q) * T - d2 * st) * d1 / st)
    zomma = gamma * (d1 * d2 - 1.0) / sigma
    veta = S * dq * phi * sqrt_t * (q + (r - q) * d1 / st - (1.0 + d1 * d2) / (2.0 * T))
    return HigherGreeks(charm=charm, speed=speed, colour=colour, zomma=zomma, veta=veta)
