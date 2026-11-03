"""Sequential curve bootstrapping from market quotes.

Instruments (all quote conventions simplified and documented in
``docs/METHODOLOGY.md``):

* :class:`Deposit`: simple interest over ``[0, T]``:
  ``P(T) = 1 / (1 + r * T)``.  Closed form.
* :class:`FRA`: forward simple rate over ``[T1, T2]``:
  ``P(T2) = P(T1) / (1 + r * (T2 - T1))``.  Requires ``P(T1)`` from
  already-bootstrapped pillars (interpolated if T1 is not a pillar).
* :class:`ParSwap`: par swap with an annual (or semi/quarterly) fixed leg,
  equal accruals ``1/frequency``:
  ``r * sum_i alpha * P(t_i) + P(T) = 1``.
  Solved by 1-D root search (Brent) on the new pillar's discount factor,
  because intermediate coupon dates are interpolated off the partially built
  curve, the standard sequential bootstrap.

Bond-curve bootstrap: :func:`bootstrap_bond_curve` solves each maturity
pillar's discount factor so the quoted coupon bond reprices exactly (dirty
price), again with intermediate cashflows interpolated.

Round-trip guarantee (depends on the interpolation, measured in VALIDATION.md §1.1):

* ``loglinear_df`` and ``linear_zero`` are **local**: appending a pillar
  cannot change the curve before it, so the sequential bootstrap is exact and
  every input instrument reprices to its quote to machine precision
  (measured max error 4.0e-16 in rate terms).
* ``pchip_zero`` is **non-local**: the monotone cubic is reshaped over earlier
  segments each time a pillar is appended, so already-solved instruments are
  no longer reproduced exactly. Measured max
  error 2.8e-6 in rate terms on the standard quote set (well inside a
  basis point, but ~10 orders of magnitude worse than the local schemes).
  If exact quote reproduction matters (most production risk systems require
  it), use a local scheme or iterate the bootstrap to a fixed point.

Both behaviours are pinned by tests in ``tests/test_properties.py``
(``test_local_schemes_reprice_every_instrument_exactly``,
``test_pchip_bootstrap_is_only_approximately_exact``).

Times are year fractions (ACT/365F when derived from dates); rates are
simple/par rates in decimal (0.05 = 5%).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from qp_core.model import model
from scipy.optimize import brentq

from .curve import DiscountCurve, ExtrapolationWarning

__all__ = [
    "FRA",
    "DatedBootstrapResult",
    "DatedCurve",
    "Deposit",
    "FixedPeriod",
    "FloatPeriod",
    "ForwardQuote",
    "ParSwap",
    "QuoteDiagnostic",
    "SwapQuote",
    "bootstrap_bond_curve",
    "bootstrap_curve",
    "bootstrap_dated",
    "bootstrap_ois_dated",
    "dated_quote_residual",
    "futures_rate",
    "hull_white_convexity",
    "reprice_instruments",
]

_DF_LO, _DF_HI = 1e-8, 20.0  # brentq bracket for a pillar discount factor


@dataclass(frozen=True)
class Deposit:
    """Money-market deposit: simple interest, ``P(T) = 1/(1 + rate * maturity)``.

    ``maturity`` in years, ``rate`` decimal simple annualised.
    """

    maturity: float
    rate: float

    @property
    def pillar(self) -> float:
        return self.maturity


@dataclass(frozen=True)
class FRA:
    """Forward rate agreement paying simple rate over ``[start, end]`` (years)."""

    start: float
    end: float
    rate: float

    @property
    def pillar(self) -> float:
        return self.end


@dataclass(frozen=True)
class ParSwap:
    """Par swap: fixed leg pays ``rate / frequency`` on a unit notional at
    ``1/frequency, 2/frequency, ..., maturity``; the (implicit) float leg is
    worth par, single-curve simplification (see METHODOLOGY assumptions)."""

    maturity: float
    rate: float
    frequency: int = 1

    @property
    def pillar(self) -> float:
        return self.maturity

    def payment_times(self) -> np.ndarray:
        n = int(round(self.maturity * self.frequency))
        if n < 1 or abs(n / self.frequency - self.maturity) > 1e-9:
            raise ValueError(
                f"swap maturity {self.maturity} is not a whole number of "
                f"periods at frequency {self.frequency}"
            )
        return np.arange(1, n + 1) / self.frequency


Instrument = Deposit | FRA | ParSwap


def _validate(instruments: list[Instrument]) -> list[Instrument]:
    if not instruments:
        raise ValueError("no instruments supplied to bootstrap")
    for ins in instruments:
        if isinstance(ins, Deposit) and ins.maturity <= 0:
            raise ValueError(f"deposit maturity must be > 0, got {ins.maturity}")
        if isinstance(ins, FRA) and not 0 <= ins.start < ins.end:
            raise ValueError(
                f"FRA requires 0 <= start < end, got [{ins.start}, {ins.end}]"
            )
        if isinstance(ins, ParSwap):
            if ins.maturity <= 0:
                raise ValueError(f"swap maturity must be > 0, got {ins.maturity}")
            if ins.frequency not in (1, 2, 4):
                raise ValueError(
                    f"swap frequency must be 1, 2 or 4, got {ins.frequency}"
                )
    ordered = sorted(instruments, key=lambda i: i.pillar)
    pillars = [i.pillar for i in ordered]
    for a, b in zip(pillars, pillars[1:], strict=False):
        if abs(a - b) < 1e-12:
            raise ValueError(
                f"duplicate pillar maturity {a}: each instrument must add a "
                "distinct pillar"
            )
    return ordered


def _pv_error(ins: Instrument, curve: DiscountCurve) -> float:
    """Pricing error: model par/forward rate minus quoted rate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExtrapolationWarning)
        if isinstance(ins, Deposit):
            p = float(np.asarray(curve.df(ins.maturity)))
            model = (1.0 / p - 1.0) / ins.maturity
            return model - ins.rate
        if isinstance(ins, FRA):
            model = curve.simple_forward_rate(ins.start, ins.end)
            return model - ins.rate
        model = curve.par_rate(ins.maturity, ins.frequency)
        return model - ins.rate


@model(id="EQ-FI-BOOTSTRAP")
def bootstrap_curve(
    instruments: list[Instrument],
    interpolation: str = "loglinear_df",
) -> DiscountCurve:
    """Sequentially bootstrap a discount curve.

    Instruments are sorted by pillar maturity (the result is therefore
    independent of input order); each adds one pillar whose discount factor is
    solved so the instrument reprices exactly.

    Raises
    ------
    ValueError
        On empty input, duplicate pillars, invalid instrument definitions, or
        when no discount factor in ``(1e-8, 20)`` reprices an instrument
        (unsolvable input, e.g. wildly inconsistent quotes).
    """
    ordered = _validate(instruments)
    times: list[float] = []
    dfs: list[float] = []
    for ins in ordered:
        t_new = ins.pillar

        def objective(df_new: float, _ins: Instrument = ins, _t_new: float = t_new) -> float:
            trial = DiscountCurve([*times, _t_new], [*dfs, df_new], interpolation)
            return _pv_error(_ins, trial)

        try:
            lo, hi = objective(_DF_LO), objective(_DF_HI)
            if lo * hi > 0:
                raise ValueError("no sign change in bracket")
            df_solved = brentq(objective, _DF_LO, _DF_HI, xtol=1e-16, rtol=1e-15)
        except ValueError as exc:
            raise ValueError(
                f"bootstrap failed at pillar t={t_new:.4f}y for {ins!r}: "
                f"no discount factor in ({_DF_LO}, {_DF_HI}) reprices the "
                f"quote (rate={ins.rate:+.4%}). Check quote consistency "
                f"(e.g. implied forward rates may be impossible). [{exc}]"
            ) from exc
        times.append(t_new)
        dfs.append(float(df_solved))
    return DiscountCurve(times, dfs, interpolation)


@dataclass(frozen=True)
class BondQuote:
    """Coupon bond quote for bond-curve bootstrapping (time-based cashflows).

    ``maturity`` in years, ``coupon`` decimal annual coupon rate paid
    ``frequency`` times per year on unit face, ``dirty_price`` per unit face.
    """

    maturity: float
    coupon: float
    dirty_price: float
    frequency: int = 2

    @property
    def pillar(self) -> float:
        return self.maturity

    def cashflows(self) -> tuple[np.ndarray, np.ndarray]:
        n = int(round(self.maturity * self.frequency))
        if n < 1 or abs(n / self.frequency - self.maturity) > 1e-9:
            raise ValueError(
                f"bond maturity {self.maturity} is not a whole number of "
                f"periods at frequency {self.frequency}"
            )
        t = np.arange(1, n + 1) / self.frequency
        cf = np.full(n, self.coupon / self.frequency)
        cf[-1] += 1.0
        return t, cf


def bootstrap_bond_curve(
    quotes: list[BondQuote],
    interpolation: str = "loglinear_df",
) -> DiscountCurve:
    """Bootstrap a zero curve from coupon bond dirty prices.

    Sequential: bonds sorted by maturity; each maturity pillar's discount
    factor is solved so the bond's PV matches its dirty price, with earlier
    cashflows discounted off the partially built (interpolated) curve.
    """
    if not quotes:
        raise ValueError("no bond quotes supplied to bootstrap")
    ordered = sorted(quotes, key=lambda q: q.maturity)
    for a, b in zip(ordered, ordered[1:], strict=False):
        if abs(a.maturity - b.maturity) < 1e-12:
            raise ValueError(f"duplicate bond maturity {a.maturity}")
    times: list[float] = []
    dfs: list[float] = []
    for q in ordered:
        t_cf, cf = q.cashflows()

        def objective(
            df_new: float, _q: BondQuote = q, _cf: np.ndarray = cf, _t_cf: np.ndarray = t_cf
        ) -> float:
            trial = DiscountCurve(
                [*times, _q.maturity], [*dfs, df_new], interpolation
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ExtrapolationWarning)
                pv = float(np.sum(_cf * np.asarray(trial.df(_t_cf))))
            return pv - _q.dirty_price

        try:
            if objective(_DF_LO) * objective(_DF_HI) > 0:
                raise ValueError("no sign change in bracket")
            df_solved = brentq(objective, _DF_LO, _DF_HI, xtol=1e-16, rtol=1e-15)
        except ValueError as exc:
            raise ValueError(
                f"bond bootstrap failed at t={q.maturity:.4f}y "
                f"(coupon={q.coupon:.4%}, price={q.dirty_price:.6f}): "
                f"no admissible discount factor reprices the bond. [{exc}]"
            ) from exc
        times.append(q.maturity)
        dfs.append(float(df_solved))
    return DiscountCurve(times, dfs, interpolation)


def reprice_instruments(
    instruments: list[Instrument], curve: DiscountCurve
) -> list[tuple[Instrument, float]]:
    """Model-minus-quote rate error for each instrument (round-trip check)."""
    return [(ins, _pv_error(ins, curve)) for ins in instruments]


# ---------------------------------------------------------------------------
# Dated bootstrap (W13b)
# ---------------------------------------------------------------------------
#
# Plain idea. A curve is a list of discount factors on dates. Each market
# quote (a deposit, a future, an FRA, an OIS or a term swap) adds one date,
# its last date on the curve being built, and we pick the discount factor on
# that date so the quote is reproduced exactly. Between dates the log of the
# discount factor is linear in time (piecewise-flat forwards); after the last
# date the last forward is held flat, so the forward curve has no jump at the
# end (the year-based ``DiscountCurve`` holds the zero rate flat instead,
# which does make the forward jump there).
#
# Formal statement. Valuation date day 0, curve time t = days/365 (ACT/365F).
# ``ln P`` is linear in t on [0, t_1], [t_1, t_2], ...; for t > t_n,
# ``ln P(t) = ln P(t_n) - f_n (t - t_n)`` with ``f_n`` the last segment's
# forward. Instruments, with P the curve being projected on and D the
# discount curve (the same curve for an OIS curve):
#
#   forward rate R on [s, e], accrual tau:   (P(s)/P(e) - 1) / tau = R
#     (deposits, FRAs, and futures with R = (100 - price)/100 - convexity)
#   swap with fixed rate K:
#     sum_j D(p_j) (P(s_j)/P(e_j) - 1) = K sum_i tau_i D(p_i)
#
# For a compounded overnight leg the daily compounding on one curve
# telescopes exactly to P(s)/P(e) (Henrard 2014, ch. 2; Andersen and
# Piterbarg 2010, vol. 1 sec. 6.5), so an OIS float period needs only its
# accrual start and end. Each pillar's log discount factor is found by
# bisection on the fixed bracket ln P in [ln 1e-8, ln 20] (wide on purpose,
# and stated: a one-year zero rate anywhere from about -300% to +1800%)
# until the bracket cannot shrink in floating point. With a local
# interpolation the earlier pillars do not move, so every quote reprices to
# rounding.
#
# Periods are given as integer days from the valuation date plus the accrual
# fraction the schedule computed. Building them (calendars, rolls, stubs,
# payment lags) is the job of ``fi_rates.schedule`` / ``fi_rates.legs``
# (W13a); this module only needs objects with the attributes of
# :class:`FixedPeriod` and :class:`FloatPeriod`, which keeps that seam small.
#
# Worked example and oracle: ``tests/data/dated_bootstrap_usd.txt`` is
# written by QuantLib 1.44 (``tests/oracle/gen_dated_bootstrap_fixture.py``);
# ``tests/test_bootstrap_dated.py`` rebuilds the curves in it and matches
# QuantLib's pillar discount factors. The Rust (``rust/rates-curves``) and
# C++ (``cpp/rates-curves``) ports read the same file.

LN_DF_LO = math.log(1e-8)
LN_DF_HI = math.log(20.0)
DAYS_PER_YEAR = 365.0


class DatedCurve:
    """Discount curve on integer day pillars, log-linear, flat-forward beyond.

    ``days`` strictly increasing and positive; ``dfs`` positive. ``df(day)``
    accepts any real day >= 0 (fractional days are allowed).
    """

    __slots__ = ("_t", "_x", "days", "dfs")

    def __init__(self, days: Sequence[int], dfs: Sequence[float]) -> None:
        if len(days) == 0 or len(days) != len(dfs):
            raise ValueError("days and dfs must be non-empty and of equal length")
        prev = 0
        for d in days:
            if int(d) != d or d <= prev:
                raise ValueError("pillar days must be integers, positive and strictly increasing")
            prev = int(d)
        for x in dfs:
            if not (math.isfinite(x) and x > 0.0):
                raise ValueError(f"discount factors must be finite and positive, got {x}")
        self.days = tuple(int(d) for d in days)
        self.dfs = tuple(float(x) for x in dfs)
        self._t = [0.0] + [d / DAYS_PER_YEAR for d in self.days]
        self._x = [0.0] + [math.log(x) for x in self.dfs]

    def log_df(self, day: float) -> float:
        if not day >= 0.0:
            raise ValueError(f"day must be >= 0, got {day}")
        t = day / DAYS_PER_YEAR
        ts, xs = self._t, self._x
        n = len(ts) - 1
        if t >= ts[n]:
            slope = (xs[n] - xs[n - 1]) / (ts[n] - ts[n - 1])
            return xs[n] + slope * (t - ts[n])
        lo, hi = 0, n
        while hi - lo > 1:  # binary search: ts[lo] <= t < ts[hi]
            mid = (lo + hi) // 2
            if ts[mid] <= t:
                lo = mid
            else:
                hi = mid
        w = (t - ts[lo]) / (ts[hi] - ts[lo])
        return xs[lo] + w * (xs[hi] - xs[lo])

    def df(self, day: float) -> float:
        return math.exp(self.log_df(day))

    def zero_rate(self, day: float) -> float:
        """Continuously compounded ACT/365F zero rate."""
        if not day > 0:
            raise ValueError("zero rate needs day > 0")
        return -self.log_df(day) / (day / DAYS_PER_YEAR)

    def forward_rate(self, day: float) -> float:
        """Instantaneous forward (continuously compounded) just after ``day``.

        Piecewise flat: the slope of ``-ln P`` on the segment holding ``day``,
        and the last segment's slope beyond the last pillar.
        """
        if not day >= 0.0:
            raise ValueError(f"day must be >= 0, got {day}")
        t = day / DAYS_PER_YEAR
        ts, xs = self._t, self._x
        k = len(ts) - 1
        for i in range(1, len(ts)):
            if t < ts[i]:
                k = i
                break
        return -(xs[k] - xs[k - 1]) / (ts[k] - ts[k - 1])

    def to_discount_curve(self) -> DiscountCurve:
        """The same pillars as a year-based :class:`DiscountCurve` (log-linear)."""
        return DiscountCurve([d / DAYS_PER_YEAR for d in self.days], list(self.dfs), "loglinear_df")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"DatedCurve({len(self.days)} pillars, last day {self.days[-1]})"


class FixedPeriodLike(Protocol):
    @property
    def pay(self) -> int: ...
    @property
    def accrual(self) -> float: ...


class FloatPeriodLike(Protocol):
    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...
    @property
    def pay(self) -> int: ...
    @property
    def accrual(self) -> float: ...


class DfSource(Protocol):
    def df(self, day: float) -> float: ...


@dataclass(frozen=True)
class FixedPeriod:
    """Fixed coupon: paid on day ``pay`` with accrual fraction ``accrual``."""

    pay: int
    accrual: float


@dataclass(frozen=True)
class FloatPeriod:
    """Floating coupon projected on ``[start, end]`` and paid on ``pay``.

    The rate is ``(P(start)/P(end) - 1) / accrual``: a term rate fixed at par
    over the accrual period, or a compounded overnight rate, which telescopes
    to the same expression on one curve.
    """

    start: int
    end: int
    pay: int
    accrual: float


@dataclass(frozen=True)
class ForwardQuote:
    """Simple rate ``rate`` on ``[start, end]`` with accrual ``accrual``.

    A deposit (start = spot), an FRA, or a futures contract with
    ``rate = futures_rate(price, convexity)``.
    """

    start: int
    end: int
    accrual: float
    rate: float
    label: str = ""


@dataclass(frozen=True)
class SwapQuote:
    """Par swap: fixed ``rate`` on ``fixed`` against floating ``floating``."""

    rate: float
    fixed: tuple[FixedPeriodLike, ...]
    floating: tuple[FloatPeriodLike, ...]
    label: str = ""


DatedQuote = ForwardQuote | SwapQuote


def futures_rate(price: float, convexity: float = 0.0) -> float:
    """Forward rate implied by a futures price: ``(100 - price)/100 - convexity``."""
    return (100.0 - price) / 100.0 - convexity


def hull_white_convexity(price: float, t1: float, t2: float, sigma: float, a: float) -> float:
    """Futures-minus-forward convexity bias under Hull-White (ACT/365F years).

    Plain idea: a futures contract is settled daily, so a long position pays
    margin when rates rise, which is when money is dear; the fair futures
    rate is therefore above the forward rate. Formal statement: the bias of
    QuantLib's ``HullWhite::convexityBias`` (after Kirikos and Novak 1997,
    "Convexity conundrums", Risk 10(3)). With ``B(x) = (1 - e^{-a x})/a``,
    ``z = sigma^2/2 [ (1 - e^{-2 a t1})/a * B(t2-t1)^2 + B(t2-t1) B(t1)^2 ]``
    and the bias is ``(1 - e^{-z}) (F + 1/(t2 - t1))``, ``F = (100 - price)/100``.
    For ``a -> 0`` the limits ``B(x) = x`` and ``(1 - e^{-2 a t1})/a = 2 t1``
    are used, and the bias is close to the Ho-Lee
    ``sigma^2 t1 (t2 + (t2 - t1)) / 2``. Worked example (fixture ``hwconv`` lines): price 95,
    t1 = 0.25, t2 = 0.5, sigma = 0.85%, a = 3% gives about 5.1e-6 (0.05 bp).
    The volatility is a stated assumption, not calibrated.
    """
    if not (t2 > t1 >= 0.0) or sigma < 0.0 or a < 0.0:
        raise ValueError("need 0 <= t1 < t2, sigma >= 0 and a >= 0")
    dt = t2 - t1
    if a < 1e-12:
        b_dt, b_t1, var = dt, t1, 2.0 * t1
    else:
        b_dt = (1.0 - math.exp(-a * dt)) / a
        b_t1 = (1.0 - math.exp(-a * t1)) / a
        var = (1.0 - math.exp(-2.0 * a * t1)) / a
    half_s2 = 0.5 * sigma * sigma
    z = half_s2 * var * b_dt * b_dt + half_s2 * b_dt * b_t1 * b_t1
    return (1.0 - math.exp(-z)) * ((100.0 - price) / 100.0 + 1.0 / dt)


@dataclass(frozen=True)
class QuoteDiagnostic:
    """One line of the bootstrap report.

    ``residual`` is model minus quote in rate units after the whole curve is
    built (it should be at rounding level); ``iterations`` is the number of
    bisection steps taken for that pillar.
    """

    label: str
    pillar: int
    df: float
    residual: float
    iterations: int


@dataclass(frozen=True)
class DatedBootstrapResult:
    curve: DatedCurve
    diagnostics: tuple[QuoteDiagnostic, ...]

    @property
    def max_abs_residual(self) -> float:
        return max(abs(d.residual) for d in self.diagnostics)

    def report(self) -> str:
        """Quote-level table: label, pillar day, discount factor, error (bp)."""
        rows = [f"{'quote':<34}{'pillar':>7}{'df':>20}{'error bp':>12}{'iter':>6}"]
        rows += [f"{d.label[:33]:<34}{d.pillar:>7}{d.df:>20.15f}{d.residual * 1e4:>12.2e}"
                 f"{d.iterations:>6}" for d in self.diagnostics]
        return "\n".join(rows)


def _quote_pillar(q: DatedQuote, self_projected: bool, self_discounted: bool) -> int:
    if isinstance(q, ForwardQuote):
        return int(q.end)
    days: list[int] = []
    if self_projected:
        days += [p.end for p in q.floating]
    if self_discounted:
        days += [p.pay for p in q.floating] + [p.pay for p in q.fixed]
    return int(max(days))


def _validate_quote(q: DatedQuote) -> None:
    if isinstance(q, ForwardQuote):
        if not (0 <= q.start < q.end) or not q.accrual > 0 or not math.isfinite(q.rate):
            raise ValueError(f"bad forward quote {q!r}")
        return
    if not math.isfinite(q.rate) or not q.fixed or not q.floating:
        raise ValueError(f"bad swap quote {q.label!r}: need a finite rate and both legs")
    for f in q.fixed:
        if f.pay < 0 or not f.accrual > 0:
            raise ValueError(f"bad fixed period {f!r} in {q.label!r}")
    for p in q.floating:
        if not (0 <= p.start < p.end) or p.pay < 0 or not p.accrual > 0:
            raise ValueError(f"bad float period {p!r} in {q.label!r}")


def dated_quote_residual(q: DatedQuote, discount: DfSource, projection: DfSource) -> float:
    """Model minus quoted rate for one instrument (rate units)."""
    if isinstance(q, ForwardQuote):
        return (projection.df(q.start) / projection.df(q.end) - 1.0) / q.accrual - q.rate
    flt = 0.0
    for p in q.floating:
        flt += discount.df(p.pay) * (projection.df(p.start) / projection.df(p.end) - 1.0)
    ann = 0.0
    for f in q.fixed:
        ann += f.accrual * discount.df(f.pay)
    return flt / ann - q.rate


def bisect_log_df(f: Callable[[float], float], lo: float = LN_DF_LO,
                  hi: float = LN_DF_HI) -> tuple[float, int]:
    """Bisection to floating-point resolution; returns (root, iterations).

    The same loop runs in the Rust and C++ ports, so all three stop on the
    same bracket up to libm rounding. Raises ``ValueError`` without a sign
    change on the bracket.
    """
    flo, fhi = f(lo), f(hi)
    if not (math.isfinite(flo) and math.isfinite(fhi)) or (flo > 0) == (fhi > 0):
        raise ValueError(f"no sign change on the bracket (f(lo)={flo:.3e}, f(hi)={fhi:.3e})")
    it = 0
    while it < 200:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        it += 1
        fm = f(mid)
        if fm == 0.0:
            return mid, it
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return (lo if abs(flo) <= abs(fhi) else hi), it


def bootstrap_dated(
    quotes: Sequence[DatedQuote],
    *,
    discount: DfSource | None = None,
    projection: DfSource | None = None,
) -> DatedBootstrapResult:
    """Bootstrap a dated curve that reprices every quote.

    ``discount`` and ``projection`` default to the curve being built (an OIS
    curve is projected and discounted on itself). For a term-index
    projection curve pass ``discount=<OIS curve>``.

    Quotes are ordered by pillar (their last day on the curve being built);
    two quotes on one pillar raise. Each pillar is solved by bisection on
    ``ln P`` in ``[ln 1e-8, ln 20]``; a quote that no factor in that bracket
    reprices raises ``ValueError`` naming the quote.
    """
    if not quotes:
        raise ValueError("no quotes supplied to bootstrap")
    self_proj, self_disc = projection is None, discount is None
    if not (self_proj or self_disc):
        raise ValueError("at least one of discount/projection must be the curve being built")
    for q in quotes:
        _validate_quote(q)
    keyed = sorted(((_quote_pillar(q, self_proj, self_disc), i, q) for i, q in enumerate(quotes)),
                   key=lambda k: (k[0], k[1]))
    for (a, _, qa), (b, _, qb) in zip(keyed, keyed[1:], strict=False):
        if a == b:
            raise ValueError(f"quotes {qa.label!r} and {qb.label!r} share pillar day {a}")
    if keyed[0][0] <= 0:
        raise ValueError("pillars must be after the valuation date")
    days: list[int] = []
    dfs: list[float] = []
    iters: list[int] = []
    for pillar, _, q in keyed:
        def f(x: float, _p: int = pillar, _q: DatedQuote = q) -> float:
            trial = DatedCurve([*days, _p], [*dfs, math.exp(x)])
            return dated_quote_residual(_q, discount or trial, projection or trial)
        try:
            x, n = bisect_log_df(f)
        except ValueError as exc:
            raise ValueError(
                f"bootstrap failed at pillar day {pillar} for {q.label or q!r}: {exc}") from exc
        days.append(pillar)
        dfs.append(math.exp(x))
        iters.append(n)
    curve = DatedCurve(days, dfs)
    diags = tuple(
        QuoteDiagnostic(q.label, p, curve.dfs[k],
                        dated_quote_residual(q, discount or curve, projection or curve), iters[k])
        for k, (p, _, q) in enumerate(keyed))
    return DatedBootstrapResult(curve, diags)


def bootstrap_ois_dated(quotes: Sequence[DatedQuote]) -> DatedBootstrapResult:
    """OIS discount curve: every quote projected and discounted on itself."""
    return bootstrap_dated(quotes)
