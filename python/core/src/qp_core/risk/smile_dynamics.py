"""Smile dynamics: what the implied-vol smile does when spot moves.

The plain idea
--------------
An option's delta is not one number until you say what happens to its
implied vol when spot moves. If the smile stays put in strike (sticky
strike), a 1% spot move leaves the option's vol unchanged. If the smile moves
with spot (sticky moneyness), the option now sits at a different point on
the smile and its vol changes too. On a skewed smile the two deltas differ by
roughly ``vega x skew``, which for a 25-delta risk reversal is a large share
of the hedge. A risk call that does not state its dynamic is therefore
ambiguous, so every function here takes ``smile_dynamics`` as a required
keyword.

The formal statement
--------------------
Let ``vol0(K)`` be the smile observed at reference spot ``S0`` for one expiry
``T``, with forward ``F = S e^{(r-q)T}``. The three regimes (Derman 1999,
"Regimes of volatility"; Bergomi 2016 ch. 2)::

    sticky strike     sigma(K; S) = vol0(K)
    sticky moneyness  sigma(K; S) = vol0(K S0 / S)
    sticky delta      sigma(K; S) = vol0(K F0 / F)

Sticky delta keeps the vol a fixed function of Black-Scholes delta. Black
delta depends on ``(F/K, sigma)`` only, so holding the vol fixed as a function
of delta is the same as holding it fixed as a function of forward moneyness
``K/F``; that is the third line, and it needs no fixed-point iteration. With
deterministic carry (``r`` and ``q`` unchanged by the spot move) ``F/S`` is
constant and sticky delta equals sticky moneyness exactly. They differ when a
carry shock is applied with the spot move, which is why both names exist.

The smile-consistent delta is the total derivative::

    Delta_dyn = dV/dS|_sigma + vega * d sigma / dS
    d sigma / dS = 0                                   (sticky strike)
                 = -vol0'(K S0/S) K S0 / S^2           (sticky moneyness)
                 = -vol0'(K F0/F) K F0 / (F S)         (sticky delta)

Premium-adjusted FX delta (Clark 2011, *Foreign Exchange Option Pricing*,
ch. 3): when the premium is paid in the foreign (base) currency the hedge is
reduced by the premium itself, ``Delta_pa = Delta_spot - V/S``. For a call
this equals ``(K/S) e^{-r_d T} N(d2)``, which the tests use as the
independent check.

Worked example
--------------
With ``S0 = K = 100``, ``T = 1``, ``r = q = 0`` and a linear smile
``vol0(K) = 0.20 - 0.001 (K - 100)``, the call's sticky-strike delta is
``N(d1) = 0.5398``. Under sticky moneyness ``d sigma/dS = +0.001`` (the smile
slides right with spot), vega is ``39.67``, so the delta rises by
``0.0397`` to ``0.5795``. ``tests/test_risk_smile.py`` checks both against
finite differences of the surface shifted the matching way.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ..black_scholes import carry_greeks

__all__ = [
    "Smile",
    "SmileDynamics",
    "premium_adjusted_delta",
    "rank_smile_dynamics",
    "smile_delta",
    "smile_gamma",
    "smile_price",
]


class SmileDynamics(StrEnum):
    """How the smile moves with spot (see the module docstring)."""

    STICKY_STRIKE = "sticky_strike"
    STICKY_MONEYNESS = "sticky_moneyness"
    STICKY_DELTA = "sticky_delta"


def _dyn(value: SmileDynamics | str) -> SmileDynamics:
    try:
        return SmileDynamics(value)
    except ValueError:
        raise ValueError(
            f"smile_dynamics must be one of {[d.value for d in SmileDynamics]}, got {value!r}"
        ) from None


@dataclass(frozen=True)
class Smile:
    """One expiry's smile observed at a reference market.

    Parameters
    ----------
    spot0 : reference spot ``S0`` at which ``vol_of_strike`` was observed.
    expiry : ``T`` in years.
    r, q : discount rate and carry yield (``r_d``, ``r_f`` for FX), cont. comp.
    vol_of_strike : ``vol0(K)``, positive on the strikes used.
    """

    spot0: float
    expiry: float
    r: float
    q: float
    vol_of_strike: Callable[[float], float]

    def __post_init__(self) -> None:
        for name in ("spot0", "expiry"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and > 0, got {v}")
        for name in ("r", "q"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

    def forward(self, spot: float, r: float | None = None, q: float | None = None) -> float:
        rr = self.r if r is None else r
        qq = self.q if q is None else q
        return spot * math.exp((rr - qq) * self.expiry)

    def _shifted_strike(
        self, strike: float, spot: float, dynamics: SmileDynamics,
        r: float | None, q: float | None,
    ) -> float:
        if dynamics is SmileDynamics.STICKY_STRIKE:
            return strike
        if dynamics is SmileDynamics.STICKY_MONEYNESS:
            return strike * self.spot0 / spot
        return strike * self.forward(self.spot0) / self.forward(spot, r, q)

    def vol(
        self, strike: float, spot: float, *, smile_dynamics: SmileDynamics | str,
        r: float | None = None, q: float | None = None,
    ) -> float:
        """Implied vol of ``strike`` when spot is ``spot`` under the dynamic.

        ``r`` and ``q`` override the reference carry (only sticky delta
        depends on them).
        """
        d = _dyn(smile_dynamics)
        v = float(self.vol_of_strike(self._shifted_strike(strike, spot, d, r, q)))
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"smile returned a non-positive vol {v} at strike {strike}")
        return v

    def dvol_dspot(
        self, strike: float, spot: float, *, smile_dynamics: SmileDynamics | str,
        rel_step: float = 1e-5,
    ) -> float:
        """``d sigma / dS`` under the dynamic (chain rule; ``vol0'`` by a
        central difference of relative step ``rel_step`` in strike)."""
        d = _dyn(smile_dynamics)
        if d is SmileDynamics.STICKY_STRIKE:
            return 0.0
        k_eff = self._shifted_strike(strike, spot, d, None, None)
        hk = rel_step * k_eff
        slope = (self.vol_of_strike(k_eff + hk) - self.vol_of_strike(k_eff - hk)) / (2.0 * hk)
        # k_eff = strike * c / spot for both moneyness forms (c fixed), so
        # d k_eff / dS = -k_eff / S.
        return float(slope * (-k_eff / spot))


def smile_price(
    smile: Smile, strike: float, spot: float, is_call: bool, *,
    smile_dynamics: SmileDynamics | str,
) -> float:
    """Black-Scholes price at ``spot`` with the vol the dynamic implies."""
    sigma = smile.vol(strike, spot, smile_dynamics=smile_dynamics)
    return carry_greeks(spot, strike, smile.expiry, smile.r, smile.q, sigma, is_call).price


def smile_delta(
    smile: Smile, strike: float, spot: float, is_call: bool, *,
    smile_dynamics: SmileDynamics | str,
) -> float:
    """Smile-consistent spot delta ``dV/dS|_sigma + vega d sigma/dS``."""
    sigma = smile.vol(strike, spot, smile_dynamics=smile_dynamics)
    g = carry_greeks(spot, strike, smile.expiry, smile.r, smile.q, sigma, is_call)
    return g.delta + g.vega * smile.dvol_dspot(strike, spot, smile_dynamics=smile_dynamics)


def smile_gamma(
    smile: Smile, strike: float, spot: float, is_call: bool, *,
    smile_dynamics: SmileDynamics | str, rel_step: float = 1e-4,
) -> float:
    """Smile-consistent gamma: central difference of :func:`smile_delta`."""
    h = rel_step * spot
    up = smile_delta(smile, strike, spot + h, is_call, smile_dynamics=smile_dynamics)
    dn = smile_delta(smile, strike, spot - h, is_call, smile_dynamics=smile_dynamics)
    return (up - dn) / (2.0 * h)


def premium_adjusted_delta(
    spot: float, strike: float, expiry: float, r_d: float, r_f: float, sigma: float,
    is_call: bool,
) -> float:
    """Premium-adjusted spot delta ``Delta_spot - V/S`` (premium in the base
    currency; Clark 2011 ch. 3), in units of base currency per unit notional.
    """
    g = carry_greeks(spot, strike, expiry, r_d, r_f, sigma, is_call)
    return g.delta - g.price / spot


def rank_smile_dynamics(
    spots: Sequence[float],
    prices: Sequence[float],
    times: Sequence[float],
    smile: Smile,
    strike: float,
    is_call: bool,
) -> dict[SmileDynamics, float]:
    """Which dynamic does a realised path support?

    For each step ``i -> i+1`` the risk-based P&L under dynamic ``d`` is
    ``Delta_d dS + 0.5 Gamma_d dS^2 + Theta dt`` evaluated at step ``i``,
    with theta (``dV/dt``, calendar time) from the vol the dynamic implies.
    The smile ``vol0(K)`` is held fixed over the path, which is the right
    approximation for a path short against the expiry; the unexplained P&L is the
    realised price change minus that. Returns the root-mean-square
    unexplained P&L per dynamic; the smallest is the dynamic the path
    supports. ``times`` are years from the smile's reference date (the
    option's remaining life is ``smile.expiry - t``).
    """
    s = np.asarray(spots, dtype=float)
    p = np.asarray(prices, dtype=float)
    t = np.asarray(times, dtype=float)
    if not (s.shape == p.shape == t.shape) or s.ndim != 1 or s.size < 2:
        raise ValueError("spots, prices and times must be 1-D of equal length >= 2")
    out: dict[SmileDynamics, float] = {}
    for d in SmileDynamics:
        resid = []
        for i in range(s.size - 1):
            tau = smile.expiry - t[i]
            local_t = Smile(smile.spot0, tau, smile.r, smile.q, smile.vol_of_strike)
            sigma = local_t.vol(strike, s[i], smile_dynamics=d)
            g = carry_greeks(s[i], strike, tau, smile.r, smile.q, sigma, is_call)
            delta = g.delta + g.vega * local_t.dvol_dspot(strike, s[i], smile_dynamics=d)
            gamma = smile_gamma(local_t, strike, s[i], is_call, smile_dynamics=d)
            ds = s[i + 1] - s[i]
            explained = delta * ds + 0.5 * gamma * ds * ds + g.theta * (t[i + 1] - t[i])
            resid.append((p[i + 1] - p[i]) - explained)
        out[d] = float(np.sqrt(np.mean(np.square(resid))))
    return out
