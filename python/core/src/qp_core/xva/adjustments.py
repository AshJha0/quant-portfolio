"""CVA, DVA and FVA from discounted exposure profiles.

Plain idea
----------
CVA is the price of the counterparty's default risk: in each small time
slice, the chance the counterparty defaults then, times what we would lose
(discounted expected exposure times one minus recovery), summed over the
life of the trades. DVA is the mirror image for our own default. FVA is the
cost of funding the uncollateralised positive exposure at our borrowing
spread (FCA), less the benefit of the negative exposure at our lending
spread (FBA).

Formal statement
----------------
On a grid 0 = t_0 < ... < t_m with survival curves S_C (counterparty) and
S_B (bank), recoveries R_C, R_B and discounted profiles EE*, ENE* (ENE* <= 0):

    CVA = (1 - R_C) sum_k EE*(t_k)  [S_C(t_{k-1}) - S_C(t_k)] w_B(t_k)
    DVA = (1 - R_B) sum_k -ENE*(t_k) [S_B(t_{k-1}) - S_B(t_k)] w_C(t_k)
    FCA = sum_k s_b EE*(t_k)   (t_k - t_{k-1}) S_C(t_k) S_B(t_k)
    FBA = sum_k s_l (-ENE*(t_k)) (t_k - t_{k-1}) S_C(t_k) S_B(t_k)

where w is 1 (unilateral) or the other party's survival to t_k
(first-to-default, independent defaults). Exposure is taken at the right
end of each slice. All four are reported as non-negative numbers; the
adjusted value of the netting set is V - CVA + DVA - FCA + FBA
(Gregory 2020, ch. 14 and 18; this is the "symmetric funding" convention,
one of several in use, chosen because it needs only the two profiles).

Survival curves are piecewise-flat hazard rates; a flat CDS spread s maps
to a hazard s / (1 - R) by the credit triangle (O'Kane 2008, sec. 6.4).

Worked example
--------------
A flat EE* of 1 over five years, hazard 2 %, recovery 40 %:
CVA = 0.6 (1 - e^{-0.1}) = 0.0571.

>>> h = HazardCurve.flat(0.02)
>>> t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
>>> round(cva(t, [1.0] * 6, h, 0.4), 4)
0.0571

Evidence class: definitions; the CVA of an IRS is cross-checked against the
swaption strip in the tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from qp_core.model import model

from ..errors import InputError

__all__ = ["HazardCurve", "XvaResult", "cva", "dva", "fva", "xva"]


@dataclass(frozen=True)
class HazardCurve:
    """Piecewise-flat hazard: ``hazards[i]`` applies on (times[i-1], times[i]]
    with times[-1] extended flat beyond."""

    times: tuple[float, ...]
    hazards: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.times) != len(self.hazards) or not self.times:
            raise InputError("times and hazards must be non-empty and equal length")
        if self.times[0] <= 0.0 or any(b <= a for a, b in zip(self.times, self.times[1:], strict=False)):
            raise InputError("times must be positive and increasing")
        if any(not (h >= 0.0 and math.isfinite(h)) for h in self.hazards):
            raise InputError("hazards must be non-negative and finite")

    @classmethod
    def flat(cls, hazard: float) -> HazardCurve:
        return cls((1.0,), (float(hazard),))

    @classmethod
    def from_spread(cls, spread: float, recovery: float) -> HazardCurve:
        """Credit triangle: lambda = s / (1 - R)."""
        if not 0.0 <= recovery < 1.0:
            raise InputError("recovery must be in [0, 1)")
        return cls.flat(spread / (1.0 - recovery))

    def survival(self, t: float) -> float:
        if t <= 0.0:
            return 1.0
        acc, prev = 0.0, 0.0
        for ti, hi in zip(self.times, self.hazards, strict=True):
            if t <= ti:
                return math.exp(-(acc + hi * (t - prev)))
            acc += hi * (ti - prev)
            prev = ti
        return math.exp(-(acc + self.hazards[-1] * (t - prev)))


def _check(times: Sequence[float], prof: Sequence[float]) -> list[float]:
    t = [float(x) for x in times]
    if len(t) != len(prof) or len(t) < 2 or t[0] != 0.0:
        raise InputError("times must start at 0 and match the profile")
    if any(b <= a for a, b in zip(t, t[1:], strict=False)):
        raise InputError("times must increase strictly")
    return t


def _default_leg(
    times: Sequence[float], expo: Sequence[float], who: HazardCurve, rec: float, other: HazardCurve | None
) -> float:
    if not 0.0 <= rec <= 1.0:
        raise InputError("recovery must be in [0, 1]")
    t = _check(times, expo)
    acc = 0.0
    for k in range(1, len(t)):
        dpd = who.survival(t[k - 1]) - who.survival(t[k])
        w = 1.0 if other is None else other.survival(t[k])
        acc += float(expo[k]) * dpd * w
    return (1.0 - rec) * acc


@model(id="CORE-XVA-ADJ")
def cva(
    times: Sequence[float],
    ee_disc: Sequence[float],
    counterparty: HazardCurve,
    recovery: float,
    *,
    bank: HazardCurve | None = None,
) -> float:
    """CVA from the discounted EE profile; pass ``bank`` for first-to-default."""
    return _default_leg(times, ee_disc, counterparty, recovery, bank)


@model(id="CORE-XVA-ADJ")
def dva(
    times: Sequence[float],
    ene_disc: Sequence[float],
    bank: HazardCurve,
    recovery: float,
    *,
    counterparty: HazardCurve | None = None,
) -> float:
    """DVA from the discounted ENE profile (values <= 0); returned >= 0."""
    return _default_leg(times, [-float(e) for e in ene_disc], bank, recovery, counterparty)


@model(id="CORE-XVA-ADJ")
def fva(
    times: Sequence[float],
    ee_disc: Sequence[float],
    ene_disc: Sequence[float],
    borrow_spread: float,
    lend_spread: float,
    counterparty: HazardCurve,
    bank: HazardCurve,
) -> tuple[float, float]:
    """(FCA, FBA), both >= 0 for non-negative spreads."""
    t = _check(times, ee_disc)
    _check(times, ene_disc)
    fca = fba = 0.0
    for k in range(1, len(t)):
        w = (t[k] - t[k - 1]) * counterparty.survival(t[k]) * bank.survival(t[k])
        fca += borrow_spread * float(ee_disc[k]) * w
        fba += lend_spread * -float(ene_disc[k]) * w
    return fca, fba


@dataclass(frozen=True)
class XvaResult:
    cva: float
    dva: float
    fca: float
    fba: float

    @property
    def total(self) -> float:
        """Adjustment to add to the risk-free value: -CVA + DVA - FCA + FBA."""
        return -self.cva + self.dva - self.fca + self.fba


@model(id="CORE-XVA-ADJ")
def xva(
    times: Sequence[float],
    ee_disc: Sequence[float],
    ene_disc: Sequence[float],
    counterparty: HazardCurve,
    bank: HazardCurve,
    *,
    r_counterparty: float = 0.4,
    r_bank: float = 0.4,
    borrow_spread: float = 0.0,
    lend_spread: float = 0.0,
    bilateral: bool = True,
) -> XvaResult:
    """All four adjustments on one grid."""
    c = cva(times, ee_disc, counterparty, r_counterparty, bank=bank if bilateral else None)
    d = dva(times, ene_disc, bank, r_bank, counterparty=counterparty if bilateral else None)
    f, b = fva(times, ee_disc, ene_disc, borrow_spread, lend_spread, counterparty, bank)
    return XvaResult(c, d, f, b)
