"""Caps and floors on dated periods, with payment lag, smiles and cap stripping.

Plain idea. A cap is a strip of caplets. Caplet ``i`` looks at the term
rate ``L_i`` fixed at ``t_fix`` for the accrual period ``[t_s, t_e]`` and
pays ``tau_i max(L_i - K, 0)`` on the payment date ``t_pay``, which can be
a few days after ``t_e`` (a payment lag). Brokers quote one flat vol per
cap maturity; a desk needs one vol per caplet. Stripping turns the first
into the second by bootstrapping: the 1y cap fixes the vol of its
caplets, the 2y cap then fixes the vol of the caplets it adds, and so on.

Formal statement.

* Caplet under the ``t_pay``-forward measure (Brigo and Mercurio 2006,
  "Interest Rate Models - Theory and Practice", 2nd ed., eq. 1.26 and
  section 6.4; Hull 2018, ch. 29, eq. 29.7)::

      caplet = N tau DF(t_pay) Black(F, K, sigma, t_fix)
      F = (P_p(t_s) / P_p(t_e) - 1) / tau     (projection curve P_p)

  ``Black`` is the undiscounted Black-76 call (``model="black"``),
  shifted Black-76 on ``F + shift`` and ``K + shift``
  (``"shifted_black"``) or Bachelier (``"normal"``). Floorlets use the put.
  Discounting to ``t_pay`` rather than ``t_e`` is the payment-lag fix the
  old ``rate_options.cap_price`` lacked (plan W15, ``rate_options.py:88-90``).
  The convexity adjustment for a lagged payment is second order in the
  lag and is not applied (Henrard 2014, "Interest Rate Modelling in the
  Multi-curve Framework", section 4.3, puts it below 0.01 bp for a
  2-business-day lag); it is a stated simplification, not hidden.
* Cap stripping (Hull 2018, ch. 29.2, "spot vols from flat vols"): with
  cap maturities ``M_1 < M_2 < ...`` and flat vols ``s_j``, the caplet
  vol is piecewise constant, ``sigma_i = v_j`` for caplets that belong to
  ``M_j`` but not ``M_{j-1}``, and ``v_j`` solves
  ``Cap(M_j; s_j) = Cap(M_{j-1}; v_1..v_{j-1}) + sum_new caplet(v_j)``
  by Brent's method. Every cap then reprices to the solver tolerance.
* SABR caplet smiles: :func:`sabr_caplet_vol` gives the caplet vol at any
  strike from per-expiry SABR parameters (:mod:`qp_core.vol.sabr`).

Dates. Times are ACT/365F year fractions from the valuation date; accruals
use the leg's day count. :func:`periods_from_schedule` is the seam to the
W13a ``Schedule``: it reads ``accrual_start``, ``accrual_end``,
``payment_date`` and (if present) ``fixing_date`` from each period, so any
object with those attributes works until that interface is frozen.

Worked example (``tests/test_caps.py``): flat 3% curve, 5y quarterly cap at
3%, 20% Black vol: the dated cap with a 2-day payment lag is worth a
little less than the same cap paid on the accrual end, because each
payment is discounted two days further.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from qp_core.daycount import year_fraction
from qp_core.vol.sabr import sabr_vol
from scipy.optimize import brentq

from .curve import DiscountCurve
from .rate_options import bachelier, black76

__all__ = [
    "CapletPeriod",
    "StrippedCapletVols",
    "cap_price",
    "caplet_price",
    "periods_from_schedule",
    "sabr_caplet_vol",
    "strip_caplet_vols",
]

MODELS = ("black", "shifted_black", "normal")


@dataclass(frozen=True)
class CapletPeriod:
    """One caplet; times in years from valuation, ``accrual`` in the leg's day count."""

    fixing: float
    start: float
    end: float
    payment: float
    accrual: float

    def __post_init__(self) -> None:
        if not (self.start < self.end and self.payment >= self.start and self.accrual > 0.0):
            raise ValueError(f"inconsistent caplet period {self}")
        if self.fixing > self.start:
            raise ValueError("fixing must be on or before the accrual start")


def periods_from_schedule(schedule, valuation: dt.date, accrual_day_count: str = "ACT/360",
                          time_day_count: str = "ACT/365F") -> list[CapletPeriod]:
    """Caplet periods from a dated schedule (seam to the W13a ``Schedule``).

    ``schedule`` is either an iterable of period objects or an object with
    a ``periods`` attribute. Each period needs ``accrual_start``,
    ``accrual_end``, ``payment_date``; ``fixing_date`` defaults to the
    accrual start (in-advance term rate). Periods that fixed on or before
    ``valuation`` are dropped: their payoff is known and belongs in the
    fixings store, not in an option price.
    """
    items: Iterable = getattr(schedule, "periods", schedule)
    out = []
    for p in items:
        fix = getattr(p, "fixing_date", None) or p.accrual_start
        if fix <= valuation:
            continue

        def t(d: dt.date) -> float:
            return year_fraction(valuation, d, time_day_count)

        out.append(CapletPeriod(t(fix), t(p.accrual_start), t(p.accrual_end), t(p.payment_date),
                                year_fraction(p.accrual_start, p.accrual_end, accrual_day_count)))
    return out


def _undiscounted(F: float, K: float, T: float, vol: float, call: bool, model: str,
                  shift: float) -> float:
    if model == "black":
        return black76(F, K, T, vol, call)
    if model == "shifted_black":
        return black76(F + shift, K + shift, T, vol, call)
    if model == "normal":
        return bachelier(F, K, T, vol, call)
    raise ValueError(f"model must be one of {MODELS}, got {model!r}")


def caplet_price(discount: DiscountCurve, projection: DiscountCurve, period: CapletPeriod,
                 strike: float, vol: float, model: str = "black", shift: float = 0.0,
                 floor: bool = False, notional: float = 1.0) -> float:
    """Caplet (floorlet) value; forward from ``projection``, discount to the payment date."""
    F = (float(projection.df(period.start)) / float(projection.df(period.end)) - 1.0) / period.accrual
    und = _undiscounted(F, strike, period.fixing, vol, not floor, model, shift)
    return notional * period.accrual * float(discount.df(period.payment)) * und


def cap_price(discount: DiscountCurve, projection: DiscountCurve,
              periods: Sequence[CapletPeriod], strike: float,
              vol: float | Sequence[float] | Callable[[CapletPeriod], float],
              model: str = "black", shift: float = 0.0, floor: bool = False,
              notional: float = 1.0) -> float:
    """Cap (floor) = sum of caplets; ``vol`` is flat, one per caplet, or a function of the period."""
    if callable(vol):
        vols = [vol(p) for p in periods]
    elif isinstance(vol, int | float):
        vols = [float(vol)] * len(periods)
    else:
        vols = [float(v) for v in vol]
        if len(vols) != len(periods):
            raise ValueError("need one vol per caplet")
    return sum(caplet_price(discount, projection, p, strike, v, model, shift, floor, notional)
               for p, v in zip(periods, vols, strict=True))


@dataclass(frozen=True)
class StrippedCapletVols:
    """Piecewise-constant caplet vols: ``vols[j]`` applies to caplets fixing in ``(cut[j-1], cut[j]]``."""

    cut: tuple[float, ...]
    vols: tuple[float, ...]
    model: str
    shift: float

    def __call__(self, period: CapletPeriod) -> float:
        for c, v in zip(self.cut, self.vols, strict=True):
            if period.fixing <= c + 1e-12:
                return v
        return self.vols[-1]


def strip_caplet_vols(discount: DiscountCurve, projection: DiscountCurve,
                      periods: Sequence[CapletPeriod], cap_maturities: Sequence[float],
                      flat_vols: Sequence[float], strike: float, model: str = "black",
                      shift: float = 0.0) -> StrippedCapletVols:
    """Bootstrap piecewise-constant caplet vols from flat cap vols at one strike.

    Cap ``j`` holds every caplet in ``periods`` whose accrual ends on or
    before ``cap_maturities[j]``. Raises if a maturity adds no caplet.
    """
    mats = [float(m) for m in cap_maturities]
    if len(mats) != len(flat_vols) or not mats or any(b <= a for a, b in zip(mats, mats[1:], strict=False)):
        raise ValueError("need increasing cap maturities, one flat vol each")
    cut: list[float] = []
    vols: list[float] = []
    prev: list[CapletPeriod] = []
    hi = 5.0 if model != "normal" else 0.5
    for m, s in zip(mats, flat_vols, strict=True):
        cap = [p for p in periods if p.end <= m + 1e-9]
        new = [p for p in cap if p not in prev]
        if not new:
            raise ValueError(f"cap maturity {m} adds no caplet")
        target = cap_price(discount, projection, cap, strike, s, model, shift)
        known = cap_price(discount, projection, prev, strike, StrippedCapletVols(
            tuple(cut), tuple(vols), model, shift), model, shift) if prev else 0.0

        def gap(v: float, new=new, known=known, target=target) -> float:
            return known + cap_price(discount, projection, new, strike, v, model, shift) - target

        if gap(1e-8) > 0.0 or gap(hi) < 0.0:
            raise ValueError(f"no caplet vol reprices the {m}y cap (calendar arbitrage in the flat vols)")
        vols.append(float(brentq(gap, 1e-8, hi, xtol=1e-15, rtol=1e-15, maxiter=200)))
        cut.append(max(p.fixing for p in new))
        prev = cap
    return StrippedCapletVols(tuple(cut), tuple(vols), model, shift)


def sabr_caplet_vol(params: Callable[[float], tuple[float, float, float, float]],
                    projection: DiscountCurve, period: CapletPeriod, strike: float,
                    shift: float = 0.0) -> float:
    """Shifted-Black caplet vol from SABR ``(alpha, beta, rho, nu) = params(t_fix)``."""
    alpha, beta, rho, nu = params(period.fixing)
    F = (float(projection.df(period.start)) / float(projection.df(period.end)) - 1.0) / period.accrual
    return sabr_vol(F, strike, period.fixing, alpha, beta, rho, nu, shift)
