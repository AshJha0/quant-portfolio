"""CDS pricing and bootstrapping on ISDA CDS Standard Model conventions.

What "ISDA standard model" pins down (ISDA CDS Standard Model, the
open-source C library published by ISDA/Markit in 2009, and the Markit
"CDS Converter Specification", 2009; O'Kane 2008, "Modelling Single-name
and Multi-name Credit Derivatives", ch. 5-7 for the maths):

* Dates. Trade date ``T``; step-in (protection effective) ``T + 1``
  calendar day; cash settlement ``T + 3`` business days. Coupons are paid on
  the 20th of Mar/Jun/Sep/Dec, business-day adjusted (following). The first
  accrual period starts on the last coupon date on or before ``T + 1``, so a
  new trade pays a full first coupon and is rebated the accrued. The last
  accrual period ends on maturity plus one day (protection covers the
  maturity date itself); every other period ends on the next adjusted
  coupon date.
* Day counts. Premium accrual ACT/360. Curve time ACT/365F from the trade
  date.
* Hazard. Piecewise flat, one knot per bootstrap maturity, flat beyond the
  last knot.
* Discounting. A real discount curve with piecewise-flat forwards (linear
  ``r t``); any object with ``df(t)`` works, and if it exposes its knot
  ``times`` they are added to the integration grid so the closed-form
  integrals below stay exact.
* Accrued on default. Paid at the default time, accrued ACT/360 from the
  period start, integrated exactly (the 2009 corrected formula, not the
  half-period approximation of ``cds.py``).

Time ``t(d) = (d - T) / 365`` is the start of day ``d``, so the start of the
step-in day is ``t = 0``. On a grid interval ``[u0, u1]`` with constant
hazard ``lambda`` and forward ``f``, ``k = lambda + f``, ``D = u1 - u0``:

    protection  = (1 - R) lambda Q(u0) P(u0) (1 - e^{-k D}) / k
    accrued-on-default = c (365/360) lambda Q(u0) P(u0)
                         [ (u0 - s) (1 - e^{-k D}) / k
                           + (1 - e^{-k D} (1 + k D)) / k^2 ]

with ``s`` the accrual start time (O'Kane 2008, eqs. 6.6 and 6.11, written
for piecewise-flat curves; a Taylor expansion is used when ``k D`` is tiny).

Quotes. A par spread is the coupon at which the clean upfront is zero. The
"quoted spread" to upfront conversion (Markit CDS Converter Specification)
bootstraps a flat hazard from the quoted spread and prices the standard
coupon on it.

Calendars. ``IsdaCds(calendar=...)`` takes a settlement centre code
(``"USD"``, ``"EUR"``, ``"GBP"``, ``"JPY"``, ``"CHF"``, ``"CAD"``,
``"AUD"``) or a joint code such as ``"EURGBP"``; the rules-based
calendars come from :mod:`qp_core.holidays` (sources listed there). The
standard contracts use the currency's centre: USD New York, EUR TARGET,
GBP London, JPY Tokyo (ISDA Standard CDS Converter, "holiday calendar"
input; the Standard North American and European Corporate contract
specifications). Coupon dates, the accrual start and ``T + 3`` cash
settlement then roll over holidays as well as weekends. The default
``calendar=None`` keeps weekends only, which is what earlier versions did
and what the weekend-only regression values in the tests pin.

Worked example: trade Fri 2026-11-06, USD calendar. Weekends only, cash
settles on Wed 11 Nov; with the USD calendar 11 Nov is Veterans Day, so it
settles on Thu 12 Nov.

Not done: the comparison with the ISDA-published test grids, which are distributed under a licence click-through
and are not in this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
from qp_core.holidays import Calendar
from qp_core.holidays import add_business_days as _add_bd
from qp_core.holidays import adjust_following as _adj
from scipy.optimize import brentq

from .cds import HazardCurve

__all__ = [
    "IsdaCds",
    "LogLinearCurve",
    "add_business_days",
    "adjust_following",
    "bootstrap_isda",
    "previous_coupon_date",
    "upfront_from_quoted_spread",
]

_DAY = 1.0 / 365.0


# ----------------------------------------------------------------- dates
def adjust_following(d: date, calendar: Calendar = None) -> date:
    """Following business-day rule; weekends only when ``calendar`` is None.

    ``calendar`` is a centre code for :mod:`qp_core.holidays`.
    """
    return _adj(d, calendar)


def add_business_days(d: date, n: int, calendar: Calendar = None) -> date:
    """``n`` business days after ``d``; weekends only when ``calendar`` is None."""
    return _add_bd(d, n, calendar)


def previous_coupon_date(d: date) -> date:
    """Latest unadjusted 20 Mar/Jun/Sep/Dec on or before ``d``."""
    y, m = d.year, d.month
    for _ in range(5):
        if m in (3, 6, 9, 12) and date(y, m, 20) <= d:
            return date(y, m, 20)
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    raise AssertionError("unreachable")


# ----------------------------------------------------------------- curve
@dataclass(frozen=True)
class LogLinearCurve:
    """Discount curve with piecewise-flat forwards (``r t`` linear), the
    ISDA model's interpolation. Flat forward before the first knot and after
    the last one."""

    times: np.ndarray
    zero_rates: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        z = np.asarray(self.zero_rates, dtype=float)
        if t.ndim != 1 or t.shape != z.shape or t.size == 0:
            raise ValueError("times and zero_rates must be equal-length 1-D arrays")
        if t[0] <= 0.0 or np.any(np.diff(t) <= 0.0):
            raise ValueError("times must be positive and strictly increasing")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "zero_rates", z)

    def df(self, t: Any) -> Any:
        x = np.asarray(t, dtype=float)
        rt = np.concatenate([[0.0], self.zero_rates * self.times])
        knots = np.concatenate([[0.0], self.times])
        f_last = (rt[-1] - rt[-2]) / (knots[-1] - knots[-2])
        inside = np.interp(np.clip(x, 0.0, knots[-1]), knots, rt)
        out = np.where(x > knots[-1], rt[-1] + f_last * (x - knots[-1]), inside)
        res = np.exp(-out)
        return float(res) if res.ndim == 0 else res


# ------------------------------------------------------------- contract
@dataclass(frozen=True)
class IsdaCds:
    """Standard CDS seen from the protection buyer, unit notional."""

    trade_date: date
    maturity: date
    coupon: float
    recovery: float = 0.4
    calendar: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.recovery < 1.0:
            raise ValueError("recovery must lie in [0, 1)")
        if self.calendar is not None:
            _adj(self.trade_date, self.calendar)  # unknown code or year raises here
        if self.maturity <= self.trade_date + timedelta(days=1):
            raise ValueError("maturity must be after the step-in date")

    @property
    def step_in(self) -> date:
        return self.trade_date + timedelta(days=1)

    @property
    def cash_settle(self) -> date:
        return add_business_days(self.trade_date, 3, self.calendar)

    @property
    def accrual_start(self) -> date:
        return adjust_following(previous_coupon_date(self.step_in), self.calendar)

    def t(self, d: date) -> float:
        """Curve time of the start of day ``d``."""
        return ((d - self.trade_date).days - 1) * _DAY

    def periods(self) -> list[tuple[date, date, date]]:
        """(accrual start, accrual end, payment date) per coupon period."""
        rolls = [previous_coupon_date(self.step_in)]
        while rolls[-1] < self.maturity:
            d = rolls[-1]
            m = d.month + 3
            rolls.append(date(d.year + (m - 1) // 12, (m - 1) % 12 + 1, 20))
        rolls[-1] = min(rolls[-1], self.maturity)
        out = []
        for i in range(1, len(rolls)):
            cal = self.calendar
            start = adjust_following(rolls[i - 1], cal)
            last = i == len(rolls) - 1
            end = self.maturity + timedelta(days=1) if last else adjust_following(rolls[i], cal)
            out.append((start, end, adjust_following(rolls[i], cal)))
        return out

    def accrued(self) -> float:
        """Accrued coupon fraction at step-in (ACT/360), rebated to the buyer."""
        return self.coupon * (self.step_in - self.accrual_start).days / 360.0


def _grid(a: float, b: float, *extra: np.ndarray) -> np.ndarray:
    pts = [np.array([a, b])]
    for e in extra:
        e = np.asarray(e, dtype=float)
        pts.append(e[(e > a) & (e < b)])
    return np.unique(np.concatenate(pts))


def _segment_terms(lam: float, f: float, d: float) -> tuple[float, float, float]:
    """(int_0^d e^{-k u} du, int_0^d u e^{-k u} du, k) with Taylor for small k d."""
    k = lam + f
    x = k * d
    if abs(x) < 1e-8:
        return d * (1 - x / 2), d * d * (0.5 - x / 3), k
    e = np.exp(-x)
    return (1 - e) / k, (1 - e * (1 + x)) / (k * k), k


def _legs(cds: IsdaCds, hazard: HazardCurve, disc: Any) -> tuple[float, float, float]:
    """(protection PV, premium PV per unit coupon incl. accrued-on-default,
    discount factor to cash settlement), all at the trade date."""
    knots_disc = getattr(disc, "times", np.array([]))
    t_end = cds.t(cds.maturity + timedelta(days=1))

    def qp(u: float) -> float:
        return float(hazard.survival(u)[0]) * float(np.asarray(disc.df(max(u, 0.0))))

    def lam_f(u0: float, u1: float) -> tuple[float, float]:
        lam = (hazard.cumulative(u1)[0] - hazard.cumulative(u0)[0]) / (u1 - u0)
        p0 = float(np.asarray(disc.df(max(u0, 0.0))))
        p1 = float(np.asarray(disc.df(u1)))
        return float(lam), float(np.log(p0 / p1) / (u1 - u0))

    prot = 0.0
    g = _grid(0.0, t_end, hazard.times, knots_disc)
    for u0, u1 in zip(g[:-1], g[1:], strict=True):
        lam, f = lam_f(u0, u1)
        i0, _, _ = _segment_terms(lam, f, u1 - u0)
        prot += lam * qp(u0) * i0
    prot *= 1.0 - cds.recovery

    prem = 0.0
    for start, end, pay in cds.periods():
        s, e = cds.t(start), cds.t(end)
        prem += (end - start).days / 360.0 * float(hazard.survival(e)[0]) \
            * float(np.asarray(disc.df(cds.t(pay) + _DAY)))
        g = _grid(max(s, 0.0), e, hazard.times, knots_disc)
        aod = 0.0
        for u0, u1 in zip(g[:-1], g[1:], strict=True):
            lam, f = lam_f(u0, u1)
            i0, i1, _ = _segment_terms(lam, f, u1 - u0)
            aod += lam * qp(u0) * ((u0 - s) * i0 + i1)
        prem += 365.0 / 360.0 * aod
    df_cash = float(np.asarray(disc.df(cds.t(cds.cash_settle) + _DAY)))
    return prot, prem, df_cash


def legs(cds: IsdaCds, hazard: HazardCurve, disc: Any) -> dict[str, float]:
    """Protection PV, risky annuity (RPV01, incl. accrued-on-default) and the
    upfront amounts, per unit notional. ``dirty`` and ``clean`` are cash
    amounts paid by the buyer at cash settlement."""
    prot, rpv01, df_cash = _legs(cds, hazard, disc)
    dirty = (prot - cds.coupon * rpv01) / df_cash
    clean = dirty + cds.accrued()
    return {"protection": prot, "rpv01": rpv01, "dirty_upfront": dirty,
            "clean_upfront": clean, "accrued": cds.accrued(), "price": 1.0 - clean}


def par_spread(cds: IsdaCds, hazard: HazardCurve, disc: Any) -> float:
    """Coupon making the clean upfront zero."""
    prot, rpv01, df_cash = _legs(cds, hazard, disc)
    acc_days = (cds.step_in - cds.accrual_start).days / 360.0
    return prot / (rpv01 - acc_days * df_cash)


def bootstrap_isda(trade_date: date, maturities: list[date], spreads: list[float],
                   disc: Any, recovery: float = 0.4,
                   calendar: str | None = None) -> HazardCurve:
    """Piecewise-flat hazard reproducing each par spread (clean upfront 0)."""
    if len(maturities) != len(spreads) or not maturities:
        raise ValueError("maturities and spreads must be non-empty and equal length")
    if any(b <= a for a, b in zip(maturities[:-1], maturities[1:], strict=True)):
        raise ValueError("maturities must be strictly increasing")
    if any(s <= 0.0 for s in spreads):
        raise ValueError("spreads must be positive")
    times: list[float] = []
    hz: list[float] = []
    for m, s in zip(maturities, spreads, strict=True):
        cds = IsdaCds(trade_date, m, s, recovery, calendar)
        tk = cds.t(m + timedelta(days=1))
        if times and tk <= times[-1]:
            raise ValueError("maturities must map to increasing curve times")

        def gap(lam: float, cds: IsdaCds = cds, tk: float = tk) -> float:
            c = HazardCurve(np.array([*times, tk]), np.array([*hz, lam]))
            return legs(cds, c, disc)["clean_upfront"]

        if gap(0.0) > 0.0:
            raise ValueError(f"spread {s} at {m} implies a negative hazard")
        hi = 1.0
        while gap(hi) < 0.0:
            hi *= 2.0
            if hi > 1e4:
                raise ValueError(f"no hazard reprices the spread at {m}")
        hz.append(brentq(gap, 0.0, hi, xtol=1e-15, rtol=1e-14, maxiter=500))
        times.append(tk)
    return HazardCurve(np.array(times), np.array(hz))


def upfront_from_quoted_spread(trade_date: date, maturity: date, quoted_spread: float,
                               coupon: float, disc: Any, recovery: float = 0.4,
                               calendar: str | None = None) -> dict[str, float]:
    """Markit conversion: flat hazard from the quoted spread, then price the
    standard ``coupon`` on it."""
    hz = bootstrap_isda(trade_date, [maturity], [quoted_spread], disc, recovery, calendar)
    flat = HazardCurve(hz.times, hz.hazards)
    return legs(IsdaCds(trade_date, maturity, coupon, recovery, calendar), flat, disc)
