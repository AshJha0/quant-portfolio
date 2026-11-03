"""Fixed, term-rate and overnight-compounded swap legs on dated schedules (W13a).

Plain idea
----------
A leg is a list of coupons. A fixed coupon is notional x rate x accrual
fraction. A term (IBOR-style) coupon uses one rate fixed two days before
the period starts. An overnight coupon (SOFR, ESTR, SONIA) compounds the
daily rates over the period, so its rate is only known at the end. Rates
already published come from a :class:`~fi_rates.fixings.FixingStore`;
future rates come from a projection curve. Coupons are discounted on a
separate discount curve.

Formal statement
----------------
For an overnight coupon over the interest period ``[S, E)`` with index
business days ``d_0 < ... < d_{m-1}`` and ``d_m = E``, lookback ``p``
business days, lockout ``L`` and the observation-shift flag:

* without a shift (the ARRC "lookback" / "lag"), day ``i`` accrues
  ``n_i = d_{i+1} - d_i`` calendar days at the rate fixed on
  ``d_i - p`` business days, and ``N = E - S``;
* with a shift, the observation period ``[S - p, E - p)`` supplies both
  the dates and the weights, and ``N`` is its length;
* with a lockout, the last ``L`` daily rates are replaced by the rate of
  the ``L``-th business day before the period end.

The compounded rate is ``R = (prod_i (1 + r_i n_i / B) - 1) B / N`` with
``B = 360`` (SOFR, ESTR) or ``365`` (SONIA), and the coupon is
``notional (R + spread) tau`` with ``tau`` the interest period's ACT/B
fraction. This is the ARRC (2020) "SOFR compounded in arrears" formula
and the convention of ``QuantLib.OvernightIndexedCoupon``. A future daily
rate is the projection curve's simple forward over one business day,
``(P(d_i) / P(d_{i+1}) - 1) B / n_i``, so with no lookback or lockout the
forecast part of the product telescopes to ``P(start) / P(E)``.

A rate is *published* when its fixing date is before the valuation date;
the valuation date's own fixing is used when the store has it. A missing
published fixing raises :class:`~fi_rates.fixings.MissingFixingError`.
Cash flows paid on or before the valuation date are excluded from the PV
(QuantLib's default for swap NPV).

Worked example
--------------
A 3-month SOFR coupon on 10m over a flat 5% (continuous, ACT/365F) curve
with no fixings yet: the forecast rate is the period's simple forward on
ACT/360, about 4.96% (5% continuous on ACT/365F restated as simple
ACT/360 over 89 days)::

    >>> import datetime as dt
    >>> from fi_rates.schedule import make_schedule
    >>> from fi_rates.fixings import SOFR, FixingStore
    >>> val = dt.date(2025, 1, 15)
    >>> curve = FlatCurve(0.05, val)
    >>> s = make_schedule(val, dt.date(2025, 4, 15), 3, calendar="USD-GOVT")
    >>> leg = OvernightLeg(s, 10e6, SOFR)
    >>> round(leg.coupons(curve, FixingStore(), val)[0].rate, 6)
    0.04962

References: ARRC, "SOFR Floating Rate Notes Conventions Matrix" and
"Recommended Conventions for SOFR Compounded in Arrears" (2020); ISDA 2021
Definitions, section 7; Henrard (2014), *Interest Rate Modelling in the
Multi-curve Framework*, ch. 2.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Protocol

from .daycount import year_fraction
from .fixings import FixingStore, IborIndex, MissingFixingError, OvernightIndex
from .schedule import Schedule, shift_business_days

__all__ = [
    "Coupon",
    "CurveOnDates",
    "DatedCurve",
    "FixedLeg",
    "FlatCurve",
    "IborLeg",
    "OvernightLeg",
    "Swap",
    "compounded_rate",
    "futures_convexity_adjustment",
]


class DatedCurve(Protocol):
    """Anything with a discount factor by date."""

    def df(self, date: dt.date) -> float: ...


@dataclass(frozen=True)
class FlatCurve:
    """Flat continuously compounded zero rate on ACT/365F from ``reference``."""

    rate: float
    reference: dt.date

    def df(self, date: dt.date) -> float:
        return math.exp(-self.rate * (date - self.reference).days / 365.0)


@dataclass(frozen=True)
class CurveOnDates:
    """A :class:`fi_rates.curve.DiscountCurve` (time in ACT/365F years from
    ``reference``) addressed by date."""

    curve: object
    reference: dt.date

    def df(self, date: dt.date) -> float:
        t = (date - self.reference).days / 365.0
        return float(self.curve.df(t))  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Coupon:
    """One coupon: dates, the rate used, the accrual fraction and the amount."""

    start: dt.date
    end: dt.date
    payment: dt.date
    rate: float
    accrual: float
    amount: float
    fixed_part: float = 1.0  # share of the period's days whose rate is published


def _check_notional(notional: float) -> float:
    n = float(notional)
    if not math.isfinite(n):
        raise ValueError(f"notional must be finite, got {notional}")
    return n


def _pv(coupons: list[Coupon], discount: DatedCurve, valuation: dt.date) -> float:
    return sum(c.amount * discount.df(c.payment) for c in coupons if c.payment > valuation)


@dataclass(frozen=True)
class FixedLeg:
    """Fixed coupons ``notional x rate x yf(start, end, day_count)``."""

    schedule: Schedule
    notional: float
    rate: float
    day_count: str = "ACT/360"

    def coupons(self) -> list[Coupon]:
        n = _check_notional(self.notional)
        out = []
        for p in self.schedule:
            tau = year_fraction(p.start, p.end, self.day_count)
            out.append(Coupon(p.start, p.end, p.payment, self.rate, tau, n * self.rate * tau))
        return out

    def pv(self, discount: DatedCurve, valuation: dt.date) -> float:
        return _pv(self.coupons(), discount, valuation)

    def annuity(self, discount: DatedCurve, valuation: dt.date) -> float:
        """PV of 1 (decimal) of fixed rate: ``sum notional tau P(pay)``."""
        n = _check_notional(self.notional)
        return sum(n * year_fraction(p.start, p.end, self.day_count) * discount.df(p.payment)
                   for p in self.schedule if p.payment > valuation)


def _rate_or_forecast(index_name: str, fixing_date: dt.date, valuation: dt.date,
                      fixings: FixingStore, forecast: float | None) -> tuple[float, bool]:
    if fixing_date < valuation:
        return fixings.get(index_name, fixing_date), True
    if fixings.has(index_name, fixing_date):
        return fixings.get(index_name, fixing_date), True
    if forecast is None:  # pragma: no cover - callers always pass one
        raise MissingFixingError(f"no {index_name} fixing for {fixing_date}")
    return forecast, False


@dataclass(frozen=True)
class IborLeg:
    """Term-rate coupons: one fixing per period, ``fixing_days`` before its start."""

    schedule: Schedule
    notional: float
    index: IborIndex
    spread: float = 0.0
    day_count: str | None = None  # defaults to the index day count

    def coupons(self, projection: DatedCurve, fixings: FixingStore,
                valuation: dt.date) -> list[Coupon]:
        n = _check_notional(self.notional)
        dc = self.day_count or self.index.day_count
        out = []
        for p in self.schedule:
            fix = self.index.fixing_date(p.start)
            mat = self.index.maturity(p.start)
            tau_i = year_fraction(p.start, mat, self.index.day_count)
            fwd = (projection.df(p.start) / projection.df(mat) - 1.0) / tau_i
            r, known = _rate_or_forecast(self.index.name, fix, valuation, fixings, fwd)
            tau = year_fraction(p.start, p.end, dc)
            out.append(Coupon(p.start, p.end, p.payment, r + self.spread, tau,
                              n * (r + self.spread) * tau, 1.0 if known else 0.0))
        return out

    def pv(self, projection: DatedCurve, discount: DatedCurve, fixings: FixingStore,
           valuation: dt.date) -> float:
        return _pv(self.coupons(projection, fixings, valuation), discount, valuation)


def _business_days(cal, start: dt.date, end: dt.date) -> list[dt.date]:  # type: ignore[no-untyped-def]
    """Business days in ``[start, end)``."""
    out = []
    d = start
    while d < end:
        if cal.is_business_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def compounded_rate(
    index: OvernightIndex,
    start: dt.date,
    end: dt.date,
    projection: DatedCurve | None,
    fixings: FixingStore,
    valuation: dt.date,
    *,
    lookback_days: int = 0,
    lockout_days: int = 0,
    observation_shift: bool = False,
) -> tuple[float, float]:
    """Compounded-in-arrears rate over ``[start, end)`` and the share of
    days already fixed (see the module docstring for the formula)."""
    for name, v in (("lookback_days", lookback_days), ("lockout_days", lockout_days)):
        if int(v) != v or v < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {v}")
    if not start < end:
        raise ValueError(f"start {start} must precede end {end}")
    cal = index.calendar
    basis = index.basis
    if observation_shift:
        o_start = shift_business_days(cal, cal.adjust(start, "following"), -lookback_days)
        o_end = shift_business_days(cal, cal.adjust(end, "following"), -lookback_days)
        obs = _business_days(cal, o_start, o_end)
        if not obs:
            raise ValueError(f"no {index.name} business day in [{o_start}, {o_end})")
        bounds = [*obs[1:], o_end]
        weights = [(b - a).days for a, b in zip(obs, bounds, strict=True)]
        total = (o_end - o_start).days
    else:
        days = _business_days(cal, start, end)
        if not days:
            raise ValueError(f"no {index.name} business day in [{start}, {end})")
        bounds = [*days[1:], end]
        weights = [(b - a).days for a, b in zip(days, bounds, strict=True)]
        weights[0] += (days[0] - start).days  # a period starting on a holiday
        obs = [shift_business_days(cal, d, -lookback_days) for d in days]
        total = (end - start).days
    m = len(obs)
    if lockout_days >= m:
        raise ValueError(f"lockout_days {lockout_days} must be below the {m} fixings in the period")
    growth = 1.0
    fixed_days = 0
    locked: tuple[float, bool] | None = None
    for i, (o, n_i) in enumerate(zip(obs, weights, strict=True)):
        if lockout_days and i >= m - lockout_days:
            assert locked is not None
            r, known = locked
        else:
            nxt = shift_business_days(cal, o, 1)
            fwd = None
            if projection is not None:
                fwd = (projection.df(o) / projection.df(nxt) - 1.0) * basis / (nxt - o).days
            if fwd is None and not (o < valuation or fixings.has(index.name, o)):
                raise ValueError("a projection curve is needed for unpublished fixings")
            r, known = _rate_or_forecast(index.name, o, valuation, fixings, fwd)
            if lockout_days and i == m - lockout_days - 1:
                locked = (r, known)
        growth *= 1.0 + r * n_i / basis
        fixed_days += n_i if known else 0
    return (growth - 1.0) * basis / total, fixed_days / sum(weights)


@dataclass(frozen=True)
class OvernightLeg:
    """Overnight-compounded coupons (SOFR, ESTR, SONIA OIS legs and FRNs)."""

    schedule: Schedule
    notional: float
    index: OvernightIndex
    spread: float = 0.0
    lookback_days: int = 0
    lockout_days: int = 0
    observation_shift: bool = False

    def coupons(self, projection: DatedCurve | None, fixings: FixingStore,
                valuation: dt.date) -> list[Coupon]:
        n = _check_notional(self.notional)
        out = []
        for p in self.schedule:
            r, share = compounded_rate(
                self.index, p.start, p.end, projection, fixings, valuation,
                lookback_days=self.lookback_days, lockout_days=self.lockout_days,
                observation_shift=self.observation_shift)
            tau = (p.end - p.start).days / self.index.basis
            out.append(Coupon(p.start, p.end, p.payment, r + self.spread, tau,
                              n * (r + self.spread) * tau, share))
        return out

    def pv(self, projection: DatedCurve | None, discount: DatedCurve, fixings: FixingStore,
           valuation: dt.date) -> float:
        return _pv(self.coupons(projection, fixings, valuation), discount, valuation)


@dataclass(frozen=True)
class Swap:
    """Receive-fixed (``payer=False``) or pay-fixed swap of a fixed and a floating leg."""

    fixed: FixedLeg
    floating: OvernightLeg | IborLeg
    payer: bool = True  # pay fixed, receive floating

    def npv(self, projection: DatedCurve, discount: DatedCurve, fixings: FixingStore,
            valuation: dt.date) -> float:
        flt = self.floating.pv(projection, discount, fixings, valuation)
        fix = self.fixed.pv(discount, valuation)
        return (flt - fix) if self.payer else (fix - flt)

    def par_rate(self, projection: DatedCurve, discount: DatedCurve, fixings: FixingStore,
                 valuation: dt.date) -> float:
        """Fixed rate that sets the NPV to zero."""
        flt = self.floating.pv(projection, discount, fixings, valuation)
        ann = self.fixed.annuity(discount, valuation)
        if ann == 0.0:
            raise ValueError("fixed leg has no remaining coupons")
        return flt / ann


def futures_convexity_adjustment(t1: float, t2: float, sigma: float,
                                 mean_reversion: float = 0.0) -> float:
    """Futures rate minus forward rate for a rate future on ``[t1, t2]``.

    Hull-White one-factor model at a stated flat short-rate volatility
    ``sigma`` and mean reversion ``a`` (Kirikos and Novak 1997; Hull,
    *Options, Futures and Other Derivatives*, section 6.3)::

        B(s, t) = (1 - exp(-a (t - s))) / a
        adj = B(t1, t2) / (t2 - t1) * [B(t1, t2) (1 - exp(-2 a t1)) + 2 a B(0, t1)^2]
              * sigma^2 / (4 a)

    and, for ``a = 0`` (Ho-Lee), ``adj = sigma^2 t1 t2 / 2``. Both rates are
    continuously compounded; the forward is the futures rate minus ``adj``.
    Hull's worked example: ``sigma = 0.012``, ``t1 = 8``, ``t2 = 8.25``
    gives 0.00475 (0.475 %). This is a model number at a stated vol, not a
    market quote; W13b labels it so when bootstrapping from SOFR futures.
    """
    for name, v in (("t1", t1), ("t2", t2), ("sigma", sigma), ("mean_reversion", mean_reversion)):
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v}")
    if not 0.0 <= t1 < t2:
        raise ValueError(f"need 0 <= t1 < t2, got t1={t1}, t2={t2}")
    if sigma < 0.0 or mean_reversion < 0.0:
        raise ValueError("sigma and mean_reversion must be non-negative")
    a = mean_reversion
    if a < 1e-8:
        return 0.5 * sigma * sigma * t1 * t2
    b12 = -math.expm1(-a * (t2 - t1)) / a
    b01 = -math.expm1(-a * t1) / a
    return b12 / (t2 - t1) * (b12 * -math.expm1(-2.0 * a * t1) + 2.0 * a * b01 * b01) \
        * sigma * sigma / (4.0 * a)
