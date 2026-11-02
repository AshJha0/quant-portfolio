"""CDS pricing and hazard-rate bootstrapping.

Model
-----
Piecewise-constant hazard lambda(t) on the CDS maturity grid, a flat
continuously-compounded discount rate r and recovery R. Survival is
Q(t) = exp(-int_0^t lambda). For a contract of maturity T with premium dates
t_1 < ... < t_n = T (accrual fractions Delta_i = t_i - t_{i-1}):

    protection = (1 - R) int_0^T D(u) dQ(u)   (computed exactly per segment)
    premium    = s * sum_i Delta_i D(t_i) Q(t_i)
                 + s * sum_i (Delta_i / 2) D(t_i) (Q(t_{i-1}) - Q(t_i))

The second premium term is accrued premium paid on default, with the
half-period approximation of O'Kane and Turnbull (2003), "Valuation of
Credit Default Swaps", Lehman Brothers Fixed Income Quantitative Research,
eqs. (5)-(6). On a segment with constant lambda and r the protection leg
integral is closed form:

    (1 - R) * lambda / (lambda + r) * D(a) Q(a) * (1 - exp(-(lambda + r)(b - a))).

Bootstrap
---------
Par spreads s_1 < ... are given for maturities T_1 < ...; the hazard on
(T_{k-1}, T_k] is solved so the k-th contract prices at par, holding the
earlier hazards fixed (Hull, "Options, Futures and Other Derivatives", 10th
ed., section 25.4; O'Kane 2008, "Modelling Single-name and Multi-name
Credit Derivatives", section 7.3).

Oracle: with continuous premium payment the par spread equals
lambda (1 - R) exactly (the "credit triangle"), so a fine premium grid with
a flat hazard converges to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class HazardCurve:
    """Piecewise-constant hazard: ``hazards[k]`` applies on (times[k-1], times[k]]."""

    times: np.ndarray
    hazards: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        h = np.asarray(self.hazards, dtype=float)
        if t.ndim != 1 or t.shape != h.shape or t.size == 0:
            raise ValueError("times and hazards must be equal-length 1-D arrays")
        if np.any(np.diff(t) <= 0.0) or t[0] <= 0.0:
            raise ValueError("times must be positive and strictly increasing")
        if np.any(h < 0.0):
            raise ValueError("hazards must be non-negative")

    def cumulative(self, t: np.ndarray | float) -> np.ndarray:
        """int_0^t lambda(u) du; flat extrapolation of the last hazard."""
        x = np.atleast_1d(np.asarray(t, dtype=float))
        knots = np.concatenate([[0.0], self.times])
        out = np.zeros_like(x)
        for k, lam in enumerate(self.hazards):
            a, b = knots[k], knots[k + 1] if k + 1 < len(self.hazards) else np.inf
            out += lam * np.clip(np.minimum(x, b) - a, 0.0, None)
        return out

    def survival(self, t: np.ndarray | float) -> np.ndarray:
        return np.exp(-self.cumulative(t))


def _schedule(maturity: float, freq: int) -> np.ndarray:
    n = max(int(round(maturity * freq)), 1)
    return np.linspace(maturity / n, maturity, n)


def _hazard_at(curve: HazardCurve, t: float) -> float:
    idx = int(np.searchsorted(curve.times, t, side="left"))
    return float(curve.hazards[min(idx, len(curve.hazards) - 1)])


def cds_legs(
    curve: HazardCurve, maturity: float, rate: float, recovery: float, freq: int = 4
) -> tuple[float, float]:
    """(protection leg PV, risky annuity incl. accrual) per unit notional."""
    if not 0.0 <= recovery < 1.0:
        raise ValueError("recovery must lie in [0, 1)")
    if maturity <= 0.0 or freq < 1:
        raise ValueError("maturity and freq must be positive")
    pay = _schedule(maturity, freq)
    prev = np.concatenate([[0.0], pay[:-1]])
    q_pay = curve.survival(pay)
    q_prev = curve.survival(prev)
    d_pay = np.exp(-rate * pay)
    delta = pay - prev
    annuity = float(np.sum(delta * d_pay * q_pay) + np.sum(0.5 * delta * d_pay * (q_prev - q_pay)))
    # Protection: integrate exactly over sub-segments split at hazard knots.
    knots = np.unique(np.concatenate([[0.0], curve.times[curve.times < maturity], [maturity]]))
    prot = 0.0
    for a, b in zip(knots[:-1], knots[1:], strict=True):
        lam = _hazard_at(curve, 0.5 * (a + b))
        qa = float(curve.survival(a)[0])
        k = lam + rate
        if k == 0.0:
            continue
        prot += lam / k * np.exp(-rate * a) * qa * (1.0 - np.exp(-k * (b - a)))
    return (1.0 - recovery) * prot, annuity


def par_spread(curve: HazardCurve, maturity: float, rate: float, recovery: float, freq: int = 4) -> float:
    prot, ann = cds_legs(curve, maturity, rate, recovery, freq)
    return prot / ann


def bootstrap_hazards(
    maturities: np.ndarray, spreads: np.ndarray, rate: float, recovery: float, freq: int = 4
) -> HazardCurve:
    """Bootstrap a piecewise-constant hazard curve that reprices every par spread."""
    t = np.asarray(maturities, dtype=float)
    s = np.asarray(spreads, dtype=float)
    if t.shape != s.shape or t.ndim != 1 or t.size == 0:
        raise ValueError("maturities and spreads must be equal-length 1-D arrays")
    if np.any(np.diff(t) <= 0.0) or t[0] <= 0.0:
        raise ValueError("maturities must be positive and strictly increasing")
    if np.any(s <= 0.0):
        raise ValueError("spreads must be positive")
    hz: list[float] = []
    for k in range(t.size):

        def gap(lam: float, k: int = k) -> float:
            c = HazardCurve(t[: k + 1], np.array([*hz, lam]))
            prot, ann = cds_legs(c, float(t[k]), rate, recovery, freq)
            return prot - s[k] * ann

        hi = 1.0
        while gap(hi) < 0.0:
            hi *= 2.0
            if hi > 1e4:
                raise ValueError(f"no hazard reprices the {t[k]}y spread (arbitrage in the curve?)")
        if gap(0.0) > 0.0:
            raise ValueError(f"the {t[k]}y spread implies a negative hazard (arbitrage in the curve)")
        hz.append(brentq(gap, 0.0, hi, xtol=1e-15, rtol=1e-14, maxiter=500))
    return HazardCurve(t, np.array(hz))
