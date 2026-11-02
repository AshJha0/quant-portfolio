"""Three-point FX smile from ATM, 25-delta risk reversal and butterfly.

Plain idea: the FX market quotes a smile as three numbers. ATM is the vol
at the delta-neutral straddle strike. The 25-delta risk reversal is the
call vol minus the put vol at 25 delta: the skew. The 25-delta butterfly is
the average of those two vols minus ATM: the curvature. From those we get
three (strike, vol) points, draw a smooth curve through them, and then
answer "which strike has a 25-delta (or 10-delta) under this smile?". That
last question is circular, because the vol depends on the strike, and is
solved by root finding on ``delta(K, sigma(K))``.

Formal statement (Reiswich and Wystup 2010, "FX volatility smile
construction", Wilmott Journal 2(2), 58-69 (published 2012), sections 2-3;
Clark 2011, "Foreign Exchange Option Pricing: A Practitioner's Guide",
Wiley, ch. 3). With the "smile strangle" reading of the butterfly:

``sigma_25C = ATM + BF + RR / 2``, ``sigma_25P = ATM + BF - RR / 2``.

Pillar strikes: ``K_ATM`` = delta-neutral straddle strike at the ATM vol;
``K_25C``, ``K_25P`` = strike from +/-0.25 delta at their own vols, all
under the pair's delta convention (:mod:`fx_options.deltas`). Between the
pillars, vol is the quadratic in ``x = ln(K / F)`` through the three points;
beyond them it is flat at the wing vol, so it never turns negative.

Not covered: the broker ("market") strangle, whose quoted butterfly must
first be converted into the smile strangle by solving for the strangle
price (Reiswich and Wystup, section 3.2).

Worked example: ``S = 1.10, T = 0.5, r_d = 0.04, r_f = 0.02``, ATM 8 %,
RR -1 %, BF 0.3 %: the 25-delta call and put vols are 7.8 % and 8.8 %.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy.optimize import brentq

from .deltas import _validate_convention, atm_dns_strike, delta, strike_from_delta

__all__ = ["Smile3", "smile_from_quotes"]


@dataclass(frozen=True)
class Smile3:
    """A three-point smile; build with :func:`smile_from_quotes`."""

    S: float
    T: float
    r_d: float
    r_f: float
    convention: str
    strikes: tuple[float, float, float]  # 25P, ATM, 25C
    vols: tuple[float, float, float]
    _coef: tuple[float, float, float] = field(repr=False)

    @property
    def forward(self) -> float:
        return self.S * math.exp((self.r_d - self.r_f) * self.T)

    def vol(self, K: float) -> float:
        """Smile vol at strike ``K`` (quadratic in log-moneyness, flat wings)."""
        if not (math.isfinite(K) and K > 0.0):
            raise ValueError(f"K must be positive and finite, got {K!r}")
        if K <= self.strikes[0]:
            return self.vols[0]
        if K >= self.strikes[2]:
            return self.vols[2]
        x = math.log(K / self.forward)
        a, b, c = self._coef
        return a + b * x + c * x * x

    def delta(self, K: float, option_type: str) -> float:
        """Delta at ``K`` using the smile vol at ``K``."""
        return delta(self.S, K, self.T, self.r_d, self.r_f, self.vol(K), option_type,
                     self.convention)

    def strike_from_delta(self, target_delta: float, option_type: str) -> float:
        """Smile-consistent strike: solves ``delta(K, sigma(K)) = target_delta``.

        The root is taken on the branch away from ATM (above ATM for calls
        with a delta below the ATM delta, below it for puts likewise), which
        for premium-adjusted calls is the market's decreasing branch.
        """
        if not math.isfinite(target_delta):
            raise ValueError("target_delta must be finite")
        k_atm = self.strikes[1]
        spread = max(self.vols) * math.sqrt(self.T)
        lo, hi = self.forward * math.exp(-12.0 * spread), self.forward * math.exp(12.0 * spread)
        f = lambda k: self.delta(k, option_type) - target_delta  # noqa: E731
        at_atm = f(k_atm)
        if at_atm == 0.0:
            return k_atm
        a, b = (k_atm, hi) if at_atm > 0.0 else (lo, k_atm)
        if f(a) * f(b) > 0.0:
            raise ValueError(f"delta {target_delta!r} not attainable under this smile")
        return brentq(f, a, b, xtol=1e-14 * k_atm, rtol=1e-15, maxiter=200)


def smile_from_quotes(S: float, T: float, r_d: float, r_f: float, atm: float, rr25: float,
                      bf25: float, convention: str = "spot") -> Smile3:
    """Build the smile from ATM (DNS), 25-delta RR and smile-strangle BF."""
    conv = _validate_convention(convention)
    v_call, v_put = atm + bf25 + 0.5 * rr25, atm + bf25 - 0.5 * rr25
    if min(atm, v_call, v_put) <= 0.0:
        raise ValueError("quotes imply a non-positive vol")
    k_atm = atm_dns_strike(S, T, r_d, r_f, atm, conv)
    k_c = strike_from_delta(0.25, S, T, r_d, r_f, v_call, "call", conv)
    k_p = strike_from_delta(-0.25, S, T, r_d, r_f, v_put, "put", conv)
    if not k_p < k_atm < k_c:
        raise ValueError("pillar strikes not ordered 25P < ATM < 25C")
    F = S * math.exp((r_d - r_f) * T)
    xs = [math.log(k / F) for k in (k_p, k_atm, k_c)]
    vs = [v_put, atm, v_call]
    # Lagrange quadratic through the three points, as a + b x + c x^2.
    a = b = c = 0.0
    for i in range(3):
        j, k = [m for m in range(3) if m != i]
        den = (xs[i] - xs[j]) * (xs[i] - xs[k])
        c += vs[i] / den
        b += -vs[i] * (xs[j] + xs[k]) / den
        a += vs[i] * xs[j] * xs[k] / den
    return Smile3(S, T, r_d, r_f, conv, (k_p, k_atm, k_c), (v_put, atm, v_call), (a, b, c))
