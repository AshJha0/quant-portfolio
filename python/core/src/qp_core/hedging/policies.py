"""Delta-hedging rules with proportional transaction costs.

Plain idea
----------
A dealer who sold an option holds ``delta`` shares to stay flat. Trading
costs money, so rebalancing at every step is too expensive and never
rebalancing leaves the book open. Three rules sit between those extremes:

* **time**: rebalance to the Black-Scholes delta on a fixed calendar grid;
* **band** (Whalley and Wilmott 1997): rebalance only when the hedge leaves
  a no-trade band around delta, and then only to the nearest band edge;
* **Leland** (Leland 1985): rebalance on the grid, but to the delta computed
  with a volatility raised to pay for the expected trading cost.

Formal statement
----------------
With proportional cost ``k`` (a fraction of traded value paid on each side,
so a round trip costs ``2k``), rebalancing interval ``dt`` and volatility
``sigma``, Leland's adjusted variance for a short option is

    sigma_L^2 = sigma^2 * (1 + sqrt(2/pi) * 2k / (sigma * sqrt(dt))),

Leland (1985) eq. 9, whose ``k`` is the round-trip cost. The asymptotic
Whalley-Wilmott half-width for exponential utility with risk aversion
``lambda`` is

    H = (3/2 * exp(-r (T-t)) * k * S * Gamma^2 / lambda)^(1/3),

Whalley and Wilmott (1997) eq. 4.6 (their ``k`` is the one-side cost).

Worked example
--------------
>>> round(leland_sigma(0.20, 0.001, 1.0 / 52.0), 6)
0.205673
>>> round(whalley_wilmott_halfwidth(100.0, 0.02, 0.001, 1.0, 0.0, 1.0), 4)
0.0391
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from qp_core.errors import InputError

__all__ = [
    "HedgeRule",
    "HedgeRuleSpec",
    "leland_number",
    "leland_sigma",
    "whalley_wilmott_halfwidth",
]


class HedgeRule(StrEnum):
    """Which rebalancing rule a simulation uses."""

    TIME = "time"
    BAND = "band"
    LELAND = "leland"


@dataclass(frozen=True)
class HedgeRuleSpec:
    """A rule plus its parameters.

    ``cost`` is the one-side proportional cost ``k`` (0.0005 = 5 bp of
    traded value). ``risk_aversion`` is the Whalley-Wilmott ``lambda`` and is
    only read by the band rule; ``every`` is the rebalance spacing in grid
    steps for the time and Leland rules (the band rule checks every step).
    """

    rule: HedgeRule
    cost: float = 0.0
    risk_aversion: float = 1.0
    every: int = 1

    def __post_init__(self) -> None:
        if not (self.cost >= 0.0 and math.isfinite(self.cost)):
            raise InputError(f"cost must be finite and >= 0, got {self.cost}")
        if not (self.risk_aversion > 0.0 and math.isfinite(self.risk_aversion)):
            raise InputError(f"risk_aversion must be finite and > 0, got {self.risk_aversion}")
        if self.every < 1:
            raise InputError(f"every must be >= 1, got {self.every}")


def leland_number(sigma: float, cost: float, dt: float) -> float:
    """Leland's ``Le = sqrt(2/pi) * 2k / (sigma sqrt(dt))`` for one-side cost ``k``."""
    if not (sigma > 0.0 and dt > 0.0 and cost >= 0.0):
        raise InputError("leland_number needs sigma > 0, dt > 0, cost >= 0")
    return math.sqrt(2.0 / math.pi) * 2.0 * cost / (sigma * math.sqrt(dt))


def leland_sigma(sigma: float, cost: float, dt: float) -> float:
    """Adjusted volatility for a short option position: ``sigma * sqrt(1 + Le)``."""
    return sigma * math.sqrt(1.0 + leland_number(sigma, cost, dt))


def whalley_wilmott_halfwidth(
    spot: float, gamma: float, cost: float, risk_aversion: float, rate: float, tau: float
) -> float:
    """No-trade band half-width in shares around the Black-Scholes delta."""
    if not (spot > 0.0 and risk_aversion > 0.0 and cost >= 0.0 and tau >= 0.0):
        raise InputError("whalley_wilmott_halfwidth needs spot, risk_aversion > 0; cost, tau >= 0")
    x = 1.5 * math.exp(-rate * tau) * cost * spot * gamma * gamma / risk_aversion
    return x ** (1.0 / 3.0) if x > 0.0 else 0.0
