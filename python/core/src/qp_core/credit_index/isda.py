"""ISDA CDS Standard Model, written so the numeric core ports line for line.

Plain idea
----------
A CDS buyer pays a fixed running coupon and receives ``1 - R`` if the name
defaults. Under the ISDA CDS Standard Model the default intensity is
piecewise flat, interest rates have piecewise-flat forwards, and both legs
are integrated exactly on the union of the two curves' knots. A quoted
spread is turned into a cash upfront by bootstrapping one flat hazard from
the spread and pricing the standard coupon on it.

Formal statement
----------------
Conventions (ISDA CDS Standard Model, the C library published by
ISDA/Markit in 2009; Markit "CDS Converter Specification", 2009;
O'Kane 2008, *Modelling Single-name and Multi-name Credit Derivatives*,
ch. 5-6):

* trade date ``T``; step-in ``T + 1`` calendar day; cash settlement
  ``T + 3`` business days;
* coupons on 20 Mar/Jun/Sep/Dec, following-adjusted; the first accrual
  period starts on the last coupon date on or before ``T + 1`` (full first
  coupon, accrued rebated); the last period ends at maturity + 1 day;
* premium accrual ACT/360, curve time ACT/365F;
* hazard time ``u = (d - T - 1)/365`` (0 at the start of the step-in day),
  discount time ``(d - T)/365``;
* on a grid interval ``[u0, u1]`` with hazard ``lambda``, forward ``f``,
  ``k = lambda + f``, ``D = u1 - u0``:
  protection ``(1-R) lambda Q P (1 - e^{-kD})/k`` and accrued-on-default
  ``c (365/360) lambda Q P [(u0 - s)(1 - e^{-kD})/k + (1 - e^{-kD}(1 + kD))/k^2]``
  (O'Kane eqs. 6.6 and 6.11), Taylor-expanded when ``|kD| < 1e-8``;
* accrued-on-default carries the ISDA half-day bias: ``u0 - s`` becomes
  ``u0 - s + 1/730`` (the published C library does this; QuantLib calls it
  ``IsdaCdsEngine.HalfDayBias``). Without it the 1000bp Markit examples
  below are off by 150 to 440 per 10mm notional; with it all twenty agree
  to 0.0004.

The arithmetic lives in :func:`isda_legs_kernel`, which takes only floats
and lists; ``rust/core/src/credit_index.rs`` and
``cpp/core/include/qp/credit_index.hpp`` repeat it in the same order (plan
decision D1, parity fixture ``rust/core/tests/data/credit_index_parity.txt``).

Worked example
--------------
Trade 21 May 2009, 5-year (20 Jun 2014) contract, 100bp coupon, quoted
spread 1000bp, recovery 40%, on the Markit-published USD curve of that day:
the converter gives an upfront of 1,646,623.67 per 10mm, which
``tests/test_credit_index_isda.py`` reproduces (the reference grid is the
Markit one carried in QuantLib's ``testIsdaEngine``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from scipy.optimize import brentq

from qp_core.holidays import Calendar
from qp_core.holidays import add_business_days as _add_bd
from qp_core.holidays import adjust_following as _adj

__all__ = [
    "CdsContract",
    "FlatForwardCurve",
    "HazardCurve",
    "bootstrap_hazard",
    "cds_legs",
    "isda_legs_kernel",
    "par_spread",
    "previous_imm_date",
    "quoted_spread_from_upfront",
    "upfront_from_quoted_spread",
]

_DAY = 1.0 / 365.0
_BIAS = 1.0 / 730.0


def _check_curve(times: Sequence[float], values: Sequence[float], what: str) -> None:
    if len(times) == 0 or len(times) != len(values):
        raise ValueError(f"{what}: times and values must be non-empty and equal length")
    if times[0] <= 0.0 or any(b <= a for a, b in zip(times[:-1], times[1:], strict=True)):
        raise ValueError(f"{what}: times must be positive and strictly increasing")
    if not all(math.isfinite(v) for v in (*times, *values)):
        raise ValueError(f"{what}: values must be finite")


@dataclass(frozen=True)
class HazardCurve:
    """Piecewise-flat hazard: ``rates[k]`` on ``(times[k-1], times[k]]``, flat after."""

    times: tuple[float, ...]
    rates: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "times", tuple(float(x) for x in self.times))
        object.__setattr__(self, "rates", tuple(float(x) for x in self.rates))
        _check_curve(self.times, self.rates, "HazardCurve")
        if any(r < 0.0 for r in self.rates):
            raise ValueError("HazardCurve: hazard rates must be non-negative")

    def cumulative(self, t: float) -> float:
        return _cum_hazard(self.times, self.rates, t)

    def survival(self, t: float) -> float:
        return math.exp(-self.cumulative(t))

    def scaled(self, factor: float) -> HazardCurve:
        """Every hazard multiplied by ``factor`` (index-to-intrinsic adjustment)."""
        return HazardCurve(self.times, tuple(factor * r for r in self.rates))


@dataclass(frozen=True)
class FlatForwardCurve:
    """Discount curve, log-linear in discount factors (piecewise-flat forwards):
    the ISDA model's interpolation. ``zero_rates`` are continuously compounded
    ACT/365F; flat forward before the first knot and after the last."""

    times: tuple[float, ...]
    zero_rates: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "times", tuple(float(x) for x in self.times))
        object.__setattr__(self, "zero_rates", tuple(float(x) for x in self.zero_rates))
        _check_curve(self.times, self.zero_rates, "FlatForwardCurve")

    def df(self, t: float) -> float:
        return _df(self.times, self.zero_rates, t)


def _cum_hazard(times: Sequence[float], rates: Sequence[float], t: float) -> float:
    out, prev = 0.0, 0.0
    n = len(times)
    for k in range(n):
        hi = times[k] if k + 1 < n else math.inf
        if t <= prev:
            break
        out += rates[k] * (min(t, hi) - prev)
        prev = hi
    return out


def _df(times: Sequence[float], zeros: Sequence[float], t: float) -> float:
    """exp(-r(t) t) with r t linear between knots (and through the origin)."""
    if t <= 0.0:
        return 1.0
    n = len(times)
    if t <= times[0]:
        return math.exp(-zeros[0] * t)
    for k in range(1, n):
        if t <= times[k]:
            a, b = zeros[k - 1] * times[k - 1], zeros[k] * times[k]
            w = (t - times[k - 1]) / (times[k] - times[k - 1])
            return math.exp(-(a + w * (b - a)))
    if n == 1:
        return math.exp(-zeros[0] * t)
    f = (zeros[-1] * times[-1] - zeros[-2] * times[-2]) / (times[-1] - times[-2])
    return math.exp(-(zeros[-1] * times[-1] + f * (t - times[-1])))


def _grid(a: float, b: float, *extra: Sequence[float]) -> list[float]:
    pts = {a, b}
    for e in extra:
        pts.update(x for x in e if a < x < b)
    return sorted(pts)


def _segment(lam: float, f: float, d: float) -> tuple[float, float]:
    """(int_0^d e^{-k u} du, int_0^d u e^{-k u} du), Taylor for small k d."""
    k = lam + f
    x = k * d
    if abs(x) < 1e-8:
        return d * (1.0 - x / 2.0), d * d * (0.5 - x / 3.0)
    e = math.exp(-x)
    return (1.0 - e) / k, (1.0 - e * (1.0 + x)) / (k * k)


def isda_legs_kernel(
    hz_times: Sequence[float],
    hz_rates: Sequence[float],
    df_times: Sequence[float],
    df_zeros: Sequence[float],
    recovery: float,
    t_end: float,
    acc_start: Sequence[float],
    acc_end: Sequence[float],
    pay_time: Sequence[float],
    acc_frac: Sequence[float],
    t_cash: float,
) -> tuple[float, float, float]:
    """(protection PV, risky annuity incl. accrued-on-default, df to cash settle).

    Hazard times (``hz_times``, ``t_end``, ``acc_start``, ``acc_end``) are
    measured from the start of the step-in day; discount times
    (``pay_time``, ``t_cash``) from the trade date. ``acc_frac`` is each
    period's ACT/360 accrual. Pure floats so Rust and C++ repeat it exactly.
    """

    def disc(u: float) -> float:
        return _df(df_times, df_zeros, u)

    def qp(u: float) -> float:
        return math.exp(-_cum_hazard(hz_times, hz_rates, u)) * disc(max(u, 0.0))

    def lam_f(u0: float, u1: float) -> tuple[float, float]:
        lam = (_cum_hazard(hz_times, hz_rates, u1) - _cum_hazard(hz_times, hz_rates, u0)) / (u1 - u0)
        f = math.log(disc(max(u0, 0.0)) / disc(u1)) / (u1 - u0)
        return lam, f

    prot = 0.0
    g = _grid(0.0, t_end, hz_times, df_times)
    for u0, u1 in zip(g[:-1], g[1:], strict=True):
        lam, f = lam_f(u0, u1)
        i0, _ = _segment(lam, f, u1 - u0)
        prot += lam * qp(u0) * i0
    prot *= 1.0 - recovery

    prem = 0.0
    for s, e, pt, af in zip(acc_start, acc_end, pay_time, acc_frac, strict=True):
        prem += af * math.exp(-_cum_hazard(hz_times, hz_rates, e)) * disc(pt)
        g = _grid(max(s, 0.0), e, hz_times, df_times)
        aod = 0.0
        for u0, u1 in zip(g[:-1], g[1:], strict=True):
            lam, f = lam_f(u0, u1)
            i0, i1 = _segment(lam, f, u1 - u0)
            aod += lam * qp(u0) * ((u0 - s + _BIAS) * i0 + i1)
        prem += 365.0 / 360.0 * aod
    return prot, prem, disc(t_cash)


def previous_imm_date(d: date) -> date:
    """Latest unadjusted 20 Mar/Jun/Sep/Dec on or before ``d``."""
    y, m = d.year, d.month
    while True:
        if m in (3, 6, 9, 12) and date(y, m, 20) <= d:
            return date(y, m, 20)
        m -= 1
        if m == 0:
            y, m = y - 1, 12


@dataclass(frozen=True)
class CdsContract:
    """Standard CDS from the protection buyer's side, unit notional."""

    trade_date: date
    maturity: date
    coupon: float
    recovery: float = 0.4
    calendar: Calendar = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.recovery < 1.0:
            raise ValueError("recovery must lie in [0, 1)")
        if not math.isfinite(self.coupon) or self.coupon < 0.0:
            raise ValueError("coupon must be finite and non-negative")
        if self.maturity <= self.trade_date + timedelta(days=1):
            raise ValueError("maturity must be after the step-in date")

    @property
    def step_in(self) -> date:
        return self.trade_date + timedelta(days=1)

    @property
    def cash_settle(self) -> date:
        return _add_bd(self.trade_date, 3, self.calendar)

    @property
    def accrual_start(self) -> date:
        return _adj(previous_imm_date(self.step_in), self.calendar)

    def hazard_time(self, d: date) -> float:
        return ((d - self.trade_date).days - 1) * _DAY

    def discount_time(self, d: date) -> float:
        return (d - self.trade_date).days * _DAY

    def periods(self) -> list[tuple[date, date, date]]:
        """(accrual start, accrual end, payment date) per coupon period."""
        rolls = [previous_imm_date(self.step_in)]
        while rolls[-1] < self.maturity:
            d = rolls[-1]
            m = d.month + 3
            rolls.append(date(d.year + (m - 1) // 12, (m - 1) % 12 + 1, 20))
        rolls[-1] = min(rolls[-1], self.maturity)
        out = []
        for i in range(1, len(rolls)):
            start = _adj(rolls[i - 1], self.calendar)
            last = i == len(rolls) - 1
            end = self.maturity + timedelta(days=1) if last else _adj(rolls[i], self.calendar)
            out.append((start, end, _adj(rolls[i], self.calendar)))
        return out

    def accrued(self) -> float:
        """Accrued coupon at step-in (ACT/360), rebated to the buyer."""
        return self.coupon * (self.step_in - self.accrual_start).days / 360.0

    def kernel_inputs(self) -> dict[str, object]:
        ps = self.periods()
        return {
            "t_end": self.hazard_time(self.maturity + timedelta(days=1)),
            "acc_start": [self.hazard_time(s) for s, _, _ in ps],
            "acc_end": [self.hazard_time(e) for _, e, _ in ps],
            "pay_time": [self.discount_time(p) for _, _, p in ps],
            "acc_frac": [(e - s).days / 360.0 for s, e, _ in ps],
            "t_cash": self.discount_time(self.cash_settle),
        }


def cds_legs(cds: CdsContract, hazard: HazardCurve, disc: FlatForwardCurve) -> dict[str, float]:
    """Protection PV, RPV01 and upfronts per unit notional.

    ``dirty_upfront`` and ``clean_upfront`` are cash paid by the buyer at
    cash settlement (``clean = dirty + accrued``, the quoted upfront).
    """
    k = cds.kernel_inputs()
    prot, rpv01, df_cash = isda_legs_kernel(
        hazard.times, hazard.rates, disc.times, disc.zero_rates, cds.recovery,
        k["t_end"], k["acc_start"], k["acc_end"], k["pay_time"], k["acc_frac"],  # type: ignore[arg-type]
        k["t_cash"],  # type: ignore[arg-type]
    )
    dirty = (prot - cds.coupon * rpv01) / df_cash
    clean = dirty + cds.accrued()
    return {"protection": prot, "rpv01": rpv01, "df_cash": df_cash,
            "dirty_upfront": dirty, "clean_upfront": clean,
            "accrued": cds.accrued(), "price": 1.0 - clean}


def par_spread(cds: CdsContract, hazard: HazardCurve, disc: FlatForwardCurve) -> float:
    """Coupon at which the clean upfront is zero."""
    legs = cds_legs(cds, hazard, disc)
    acc = (cds.step_in - cds.accrual_start).days / 360.0
    return legs["protection"] / (legs["rpv01"] - acc * legs["df_cash"])


def bootstrap_hazard(trade_date: date, maturities: Sequence[date], spreads: Sequence[float],
                     disc: FlatForwardCurve, recovery: float = 0.4,
                     calendar: Calendar = None) -> HazardCurve:
    """Piecewise-flat hazard repricing each par spread to a zero clean upfront."""
    if len(maturities) != len(spreads) or not maturities:
        raise ValueError("maturities and spreads must be non-empty and equal length")
    if any(b <= a for a, b in zip(maturities[:-1], maturities[1:], strict=True)):
        raise ValueError("maturities must be strictly increasing")
    if any(not s > 0.0 for s in spreads):
        raise ValueError("spreads must be positive")
    times: list[float] = []
    hz: list[float] = []
    for m, s in zip(maturities, spreads, strict=True):
        cds = CdsContract(trade_date, m, s, recovery, calendar)
        tk = cds.hazard_time(m + timedelta(days=1))

        def gap(lam: float, cds: CdsContract = cds, tk: float = tk) -> float:
            return cds_legs(cds, HazardCurve((*times, tk), (*hz, lam)), disc)["clean_upfront"]

        if gap(0.0) > 0.0:
            raise ValueError(f"spread {s} at {m} implies a negative hazard")
        hi = 1.0
        while gap(hi) < 0.0:
            hi *= 2.0
            if hi > 1e4:
                raise ValueError(f"no hazard reprices the spread at {m}")
        hz.append(brentq(gap, 0.0, hi, xtol=1e-15, rtol=1e-14, maxiter=500))
        times.append(tk)
    return HazardCurve(tuple(times), tuple(hz))


def upfront_from_quoted_spread(trade_date: date, maturity: date, quoted_spread: float,
                               coupon: float, disc: FlatForwardCurve, recovery: float = 0.4,
                               calendar: Calendar = None) -> dict[str, float]:
    """Markit converter: flat hazard from the quoted spread, then price ``coupon``."""
    flat = bootstrap_hazard(trade_date, [maturity], [quoted_spread], disc, recovery, calendar)
    return cds_legs(CdsContract(trade_date, maturity, coupon, recovery, calendar), flat, disc)


def quoted_spread_from_upfront(trade_date: date, maturity: date, clean_upfront: float,
                               coupon: float, disc: FlatForwardCurve, recovery: float = 0.4,
                               calendar: Calendar = None) -> float:
    """Inverse of :func:`upfront_from_quoted_spread` (Brent on the quoted spread)."""
    if not math.isfinite(clean_upfront):
        raise ValueError("clean_upfront must be finite")

    def f(s: float) -> float:
        return upfront_from_quoted_spread(trade_date, maturity, s, coupon, disc,
                                          recovery, calendar)["clean_upfront"] - clean_upfront

    lo, hi = 1e-7, 0.05
    if f(lo) > 0.0:
        raise ValueError("upfront is below the value of a near-zero spread")
    while f(hi) < 0.0:
        hi *= 2.0
        if hi > 50.0:
            raise ValueError("no quoted spread reproduces the upfront")
    return float(brentq(f, lo, hi, xtol=1e-14, rtol=1e-13, maxiter=500))
