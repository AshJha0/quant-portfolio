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

Broker ("market") strangle: brokers often quote the butterfly as a single
vol ``ATM + BF_mkt`` at which a 25-delta strangle is priced, with both
strikes found at that one vol. That number is not the smile strangle. The
conversion (Reiswich and Wystup, section 3.2, the "market strangle"
equations; Clark 2011, section 3.6) finds the smile-strangle ``BF_s`` such
that the smile built from ``(ATM, RR, BF_s)`` prices the broker strangle
strikes at the broker strangle premium:

``C(K_c^ms, sigma(K_c^ms)) + P(K_p^ms, sigma(K_p^ms)) = C(K_c^ms, v_ms) + P(K_p^ms, v_ms)``

with ``v_ms = ATM + BF_mkt`` and ``K^ms`` the +/-25-delta strikes at
``v_ms``. :func:`smile_strangle_from_broker` solves it for ``BF_s`` and
:func:`broker_strangle_from_smile` inverts it. With ``RR = 0`` the two are
equal, because the broker strikes then coincide with the smile pillars.

Worked example: ``S = 1.10, T = 0.5, r_d = 0.04, r_f = 0.02``, ATM 8 %,
RR -1 %, BF 0.3 %: the 25-delta call and put vols are 7.8 % and 8.8 %.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy.optimize import brentq

from .deltas import _validate_convention, atm_dns_strike, delta, strike_from_delta
from .garman_kohlhagen import gk_price

__all__ = ["Smile3", "broker_strangle_from_smile", "broker_strangle_premium",
           "smile_from_quotes", "smile_strangle_from_broker"]


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


def _broker_strikes(S: float, T: float, r_d: float, r_f: float, v_ms: float,
                    conv: str) -> tuple[float, float]:
    k_c = strike_from_delta(0.25, S, T, r_d, r_f, v_ms, "call", conv)
    k_p = strike_from_delta(-0.25, S, T, r_d, r_f, v_ms, "put", conv)
    return k_c, k_p


def broker_strangle_premium(S: float, T: float, r_d: float, r_f: float, atm: float,
                            bf_mkt: float, convention: str = "spot") -> float:
    """Premium (domestic pips) of the broker 25-delta strangle at ``ATM + BF_mkt``."""
    conv = _validate_convention(convention)
    v_ms = atm + bf_mkt
    if not (math.isfinite(v_ms) and v_ms > 0.0):
        raise ValueError("ATM + BF_mkt must be a positive vol")
    k_c, k_p = _broker_strikes(S, T, r_d, r_f, v_ms, conv)
    return (gk_price(S, k_c, T, r_d, r_f, v_ms, "call")
            + gk_price(S, k_p, T, r_d, r_f, v_ms, "put"))


def _smile_price_at(sm: Smile3, k_c: float, k_p: float) -> float:
    return (gk_price(sm.S, k_c, sm.T, sm.r_d, sm.r_f, sm.vol(k_c), "call")
            + gk_price(sm.S, k_p, sm.T, sm.r_d, sm.r_f, sm.vol(k_p), "put"))


def _bracket_root(f, lo: float, hi: float) -> float:
    """Brent root of ``f`` on ``[lo, hi]``; raises if no sign change."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError("no strangle solution in the search range")
    return brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=300)


def smile_strangle_from_broker(S: float, T: float, r_d: float, r_f: float, atm: float,
                               rr25: float, bf_mkt: float, convention: str = "spot") -> float:
    """Convert a broker (market) 25-delta butterfly into the smile butterfly.

    Solves the Reiswich-Wystup (2010, section 3.2) market-strangle condition
    for ``BF_s``: the smile from ``(atm, rr25, BF_s)`` must price the broker
    strangle strikes at the broker premium. Returns ``BF_s``; feed it to
    :func:`smile_from_quotes`.
    """
    conv = _validate_convention(convention)
    v_ms = atm + bf_mkt
    target = broker_strangle_premium(S, T, r_d, r_f, atm, bf_mkt, conv)
    k_c, k_p = _broker_strikes(S, T, r_d, r_f, v_ms, conv)
    floor = -(atm - 0.5 * abs(rr25)) + 1e-6  # keeps both pillar vols positive
    lo, hi = max(bf_mkt - 0.25 * atm - abs(rr25), floor), bf_mkt + 0.25 * atm + abs(rr25)

    def f(bf_s: float) -> float:
        try:
            sm = smile_from_quotes(S, T, r_d, r_f, atm, rr25, bf_s, conv)
        except ValueError:
            return -target  # an unbuildable (too low) smile counts as too cheap
        return _smile_price_at(sm, k_c, k_p) - target

    return _bracket_root(f, lo, hi)


def broker_strangle_from_smile(S: float, T: float, r_d: float, r_f: float, atm: float,
                               rr25: float, bf_smile: float, convention: str = "spot") -> float:
    """Inverse of :func:`smile_strangle_from_broker`: the broker butterfly.

    Finds ``BF_mkt`` such that the strangle struck at the 25-delta strikes
    of the single vol ``atm + BF_mkt`` costs the same at that vol as under
    the smile ``(atm, rr25, bf_smile)``.
    """
    conv = _validate_convention(convention)
    sm = smile_from_quotes(S, T, r_d, r_f, atm, rr25, bf_smile, conv)

    def f(bf_mkt: float) -> float:
        v_ms = atm + bf_mkt
        k_c, k_p = _broker_strikes(S, T, r_d, r_f, v_ms, conv)
        return broker_strangle_premium(S, T, r_d, r_f, atm, bf_mkt, conv) -             _smile_price_at(sm, k_c, k_p)

    lo = max(bf_smile - 0.25 * atm - abs(rr25), -atm + 1e-6)
    hi = bf_smile + 0.25 * atm + abs(rr25)
    return _bracket_root(f, lo, hi)
