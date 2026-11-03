"""Cross-currency swaps: fixed-fixed with notional exchange, and the
float-float basis swap that quotes the cross-currency basis.

The basis swap (:class:`BasisSwap`) has constant notionals; the
mark-to-market resetting variant is not implemented.

Structure (receive_base = True, e.g. receive EUR / pay USD on EURUSD):

- optional initial exchange on the spot date t_s: pay ``notional_base``
  (base ccy), receive ``notional_quote`` (quote ccy);
- periodic fixed coupons: receive ``rate_base`` on ``notional_base`` in the
  base currency, pay ``rate_quote`` on ``notional_quote`` in the quote
  currency, ``frequency`` payments per year with unit accrual 1/frequency
  (30/360-style simplification, assumption A3);
- final exchange at maturity: receive ``notional_base``, pay
  ``notional_quote``.

Valuation is the classic two-bond decomposition: each leg is a fixed-rate
bond in its own currency; the base-currency bond is discounted on the
**basis-adjusted** foreign curve and converted at spot:

    V_quote = sign * ( S * PV_base_bond[DF_f_adj] - PV_quote_bond[DF_d] )

with sign = +1 for receive_base.  Setting the basis to zero recovers pure
CIP pricing (tested).

Initial exchange. A spot-starting swap exchanges notionals on the spot
date t_s at the quoted spot rate S, which is the price for delivery on t_s
(Clark 2011, ch. 1). Today's value of that exchange is

    sign * ( N_quote * P_d(t_s) - N_base * P_f(t_s) * S_today )
        = sign * P_d(t_s) * (N_quote - S * N_base),

because ``S_today = S * P_d(t_s) / P_f(t_s)`` (``MarketState.spot_today``).
It is zero at inception when ``N_quote = S * N_base``. ``exchange_time``
overrides t_s (forward-starting swaps). Until round 7 the exchange was
valued undiscounted as ``sign * (N_quote - S * N_base)``; for a 150,000 USD
off-market exchange on EURUSD at 4.5% with spot four days out that is
150,000.00 against the correct 149,926.05. The basis swap had the same
issue in a worse form: it converted the base notional at ``S_today`` and
left the quote notional undiscounted, so an at-market exchange on 10m EUR
was worth -2,377.82 USD instead of zero.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import brentq

from ._validation import require_finite
from .fxforward import MarketState

__all__ = [
    "BasisSwap",
    "CrossCurrencySwap",
    "XccyBasisQuote",
    "XccyBootstrapResult",
    "XccyCurve",
    "XccyFloatPeriod",
    "bootstrap_xccy_discount",
    "solve_par_basis",
    "solve_par_rate_base",
    "solve_par_rate_quote",
    "xccy_basis_residual",
]


def _check_exchange_time(te: float | None) -> None:
    if te is not None:
        require_finite(exchange_time=te)
        if te < 0.0:
            raise ValueError(f"exchange_time must be >= 0, got {te}")


def _exchange_time(te: float | None, market: MarketState | None) -> float:
    if te is not None:
        return float(te)
    if market is None or market.spot_time is None:
        return 0.0
    return float(market.spot_time)


@dataclass(frozen=True)
class CrossCurrencySwap:
    """Fixed-fixed cross-currency swap with final (and optional initial)
    notional exchange.

    Parameters
    ----------
    notional_base, notional_quote : float
        Leg notionals in their own currencies (both >= 0; typically
        ``notional_quote = notional_base * spot_at_inception``).
    rate_base, rate_quote : float
        Annualised fixed coupon rates (simple, paid ``frequency`` times a
        year with accrual 1/frequency).
    maturity : float
        Years to final exchange; ``maturity * frequency`` must be integral.
    frequency : int
        Coupon payments per year (1 = annual, 2 = semi-annual).
    receive_base : bool
        True: receive base-currency leg, pay quote leg.
    include_initial_exchange : bool
        Include the initial notional exchange (at-inception valuation).  At
        inception with ``notional_quote = notional_base * spot`` the initial
        exchange is worth zero, so seasoned valuation usually omits it.
    exchange_time : float or None
        Time (years) of the initial exchange; ``None`` means the market's
        spot date (``MarketState.spot_time``).
    """

    notional_base: float
    notional_quote: float
    rate_base: float
    rate_quote: float
    maturity: float
    frequency: int = 1
    receive_base: bool = True
    include_initial_exchange: bool = False
    pair: tuple[str, str] = ("EUR", "USD")
    label: str = ""
    exchange_time: float | None = None

    def __post_init__(self) -> None:
        if self.pair[0] == self.pair[1]:
            raise ValueError(
                f"same-currency 'cross' {self.pair[0]}/{self.pair[1]} rejected"
            )
        _check_exchange_time(self.exchange_time)
        require_finite(notional_base=self.notional_base,
                       notional_quote=self.notional_quote,
                       rate_base=self.rate_base, rate_quote=self.rate_quote,
                       maturity=self.maturity)
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be > 0, got {self.maturity}")
        if self.frequency < 1 or int(self.frequency) != self.frequency:
            raise ValueError(f"frequency must be a positive integer, got {self.frequency}")
        if self.notional_base < 0.0 or self.notional_quote < 0.0:
            raise ValueError("notionals must be >= 0 (direction is set by receive_base)")
        n_pay = self.maturity * self.frequency
        if abs(n_pay - round(n_pay)) > 1e-9:
            raise ValueError(
                f"maturity * frequency must be integral, got {self.maturity} * {self.frequency}"
            )
        if not self.label:
            object.__setattr__(
                self,
                "label",
                f"XCCY {''.join(self.pair)} {self.maturity:.0f}y "
                f"{'rec' if self.receive_base else 'pay'} {self.pair[0]}",
            )

    # ------------------------------------------------------------------ #
    def payment_times(self) -> np.ndarray:
        """Coupon payment times (years): 1/f, 2/f, ..., maturity."""
        n = int(round(self.maturity * self.frequency))
        return np.arange(1, n + 1, dtype=float) / self.frequency

    def cashflows(self, market: MarketState | None = None) -> list[tuple[str, float, float]]:
        """All cashflows as (currency, time, signed amount) from the
        perspective of the position holder. The initial exchange is at
        ``exchange_time``, else the spot time of ``market``, else 0."""
        sign = 1.0 if self.receive_base else -1.0
        base, quote = self.pair
        cfs: list[tuple[str, float, float]] = []
        if self.include_initial_exchange:
            te = _exchange_time(self.exchange_time, market)
            cfs.append((base, te, -sign * self.notional_base))
            cfs.append((quote, te, sign * self.notional_quote))
        tau = 1.0 / self.frequency
        times = self.payment_times()
        for t in times:
            cfs.append((base, float(t), sign * self.rate_base * tau * self.notional_base))
            cfs.append((quote, float(t), -sign * self.rate_quote * tau * self.notional_quote))
        cfs.append((base, float(times[-1]), sign * self.notional_base))
        cfs.append((quote, float(times[-1]), -sign * self.notional_quote))
        return cfs

    def leg_pv_base(self, market: MarketState) -> float:
        """PV of the base-currency bond leg, in base currency (coupons +
        final notional), discounted on the basis-adjusted foreign curve.
        Unsigned (direction applied in ``value``)."""
        times = self.payment_times()
        dfs = market.foreign_curve_adjusted.df(times)
        tau = 1.0 / self.frequency
        pv = self.rate_base * tau * self.notional_base * float(np.sum(dfs))
        pv += self.notional_base * float(dfs[-1])
        return pv

    def leg_pv_quote(self, market: MarketState) -> float:
        """PV of the quote-currency bond leg, in quote currency. Unsigned."""
        times = self.payment_times()
        dfs = market.domestic_curve_adjusted.df(times)
        tau = 1.0 / self.frequency
        pv = self.rate_quote * tau * self.notional_quote * float(np.sum(dfs))
        pv += self.notional_quote * float(dfs[-1])
        return pv

    def value(self, market: MarketState) -> float:
        """MTM in the quote (domestic) currency via two-bond decomposition."""
        if market.pair != self.pair:
            raise ValueError(
                f"market pair {market.pair} does not match position pair {self.pair}"
            )
        sign = 1.0 if self.receive_base else -1.0
        v = sign * (market.spot_today * self.leg_pv_base(market) - self.leg_pv_quote(market))
        if self.include_initial_exchange:
            te = _exchange_time(self.exchange_time, market)
            v += sign * (self.notional_quote * market.domestic_curve_adjusted.df(te)
                         - self.notional_base * market.foreign_curve_adjusted.df(te)
                         * market.spot_today)
        return v


# ---------------------------------------------------------------------- #
# float-float basis swap
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class BasisSwap:
    """Float-float cross-currency basis swap, constant notionals.

    The leg in the market's non-collateral currency (``market.basis_ccy``)
    pays its floating rate plus ``spread``; the collateral leg pays its
    floating rate flat. Floating rates are projected off each currency's
    own curve (``domestic_curve`` / ``foreign_curve``) and discounted on
    the collateral-consistent curves. Notionals are exchanged at maturity
    and, when ``include_initial_exchange``, on the spot date (or at
    ``exchange_time``), valued as for :class:`CrossCurrencySwap`.

    Parameters
    ----------
    notional_base, notional_quote : float
        Leg notionals (>= 0), usually ``notional_quote = notional_base * spot``.
    spread : float
        Basis spread (decimal) on the non-collateral leg.
    maturity : float
        Years; ``maturity * frequency`` must be integral.
    frequency : int
        Resets per year on both legs (default 4).
    receive_base : bool
        True: receive the base-currency leg, pay the quote leg.
    """

    notional_base: float
    notional_quote: float
    spread: float
    maturity: float
    frequency: int = 4
    receive_base: bool = True
    include_initial_exchange: bool = False
    pair: tuple[str, str] = ("EUR", "USD")
    label: str = ""
    exchange_time: float | None = None

    def __post_init__(self) -> None:
        if self.pair[0] == self.pair[1]:
            raise ValueError(f"same-currency 'cross' {self.pair[0]}/{self.pair[1]} rejected")
        _check_exchange_time(self.exchange_time)
        require_finite(notional_base=self.notional_base, notional_quote=self.notional_quote,
                       spread=self.spread, maturity=self.maturity)
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be > 0, got {self.maturity}")
        if self.frequency < 1 or int(self.frequency) != self.frequency:
            raise ValueError(f"frequency must be a positive integer, got {self.frequency}")
        if self.notional_base < 0.0 or self.notional_quote < 0.0:
            raise ValueError("notionals must be >= 0 (direction is set by receive_base)")
        n = self.maturity * self.frequency
        if abs(n - round(n)) > 1e-9:
            raise ValueError("maturity * frequency must be integral")
        if not self.label:
            object.__setattr__(
                self, "label", f"BASIS {''.join(self.pair)} {self.maturity:g}y"
            )

    def payment_times(self) -> np.ndarray:
        n = int(round(self.maturity * self.frequency))
        return np.arange(1, n + 1, dtype=float) / self.frequency

    def _leg_pv(self, projection, discount, notional: float, spread: float,
                te: float = 0.0) -> float:
        t = self.payment_times()
        tau = 1.0 / self.frequency
        p = np.asarray(projection.df(t))
        p_prev = np.concatenate(([1.0], p[:-1]))
        d = np.asarray(discount.df(t))
        coupons = float(np.sum((p_prev / p - 1.0 + spread * tau) * d))
        pv = notional * (coupons + d[-1])
        if self.include_initial_exchange:
            pv -= notional * float(discount.df(te))
        return pv

    def value(self, market: MarketState) -> float:
        """MTM in the quote currency."""
        if market.pair != self.pair:
            raise ValueError(
                f"market pair {market.pair} does not match position pair {self.pair}"
            )
        s_base = self.spread if market.basis_ccy == market.base_ccy else 0.0
        s_quote = self.spread if market.basis_ccy == market.quote_ccy else 0.0
        te = _exchange_time(self.exchange_time, market)
        pv_base = self._leg_pv(market.foreign_curve, market.foreign_curve_adjusted,
                               self.notional_base, s_base, te)
        pv_quote = self._leg_pv(market.domestic_curve, market.domestic_curve_adjusted,
                                self.notional_quote, s_quote, te)
        sign = 1.0 if self.receive_base else -1.0
        return sign * (market.spot_today * pv_base - pv_quote)


# ---------------------------------------------------------------------- #
# par solvers
# ---------------------------------------------------------------------- #
def solve_par_rate_quote(swap: CrossCurrencySwap, market: MarketState) -> float:
    """Quote-leg fixed rate that makes the (seasoned) swap PV exactly zero.

    Closed form, PV is linear in the coupon:
    ``c_d* = (S * PV_base - N_d * DF_d(T)) / (N_d * tau * sum DF_d(t_i))``.
    """
    times = swap.payment_times()
    dfs = market.domestic_curve_adjusted.df(times)
    tau = 1.0 / swap.frequency
    annuity = swap.notional_quote * tau * float(np.sum(dfs))
    if annuity <= 0.0:
        raise ValueError("cannot solve par rate for zero quote notional")
    target = market.spot_today * swap.leg_pv_base(market)
    return (target - swap.notional_quote * float(dfs[-1])) / annuity


def solve_par_rate_base(swap: CrossCurrencySwap, market: MarketState) -> float:
    """Base-leg fixed rate that makes the (seasoned) swap PV exactly zero."""
    times = swap.payment_times()
    dfs = market.foreign_curve_adjusted.df(times)
    tau = 1.0 / swap.frequency
    annuity = swap.notional_base * tau * float(np.sum(dfs))
    if annuity <= 0.0:
        raise ValueError("cannot solve par rate for zero base notional")
    target = swap.leg_pv_quote(market) / market.spot_today
    return (target - swap.notional_base * float(dfs[-1])) / annuity


def solve_par_basis(
    swap: CrossCurrencySwap,
    market: MarketState,
    bracket_bp: float = 1000.0,
) -> float:
    """Parallel shift (decimal) of the quoted basis-swap spreads that zeroes the PV.

    This is a shift of the market's basis quotes, not the par spread of the
    swap itself; use :class:`BasisSwap` for a quoted float-float spread.

    Solves ``PV(basis + x) = 0`` with Brent's method to ~1e-14 in ``x``.
    Returns the shift ``x``; the *level* of the par basis at tenor T is the
    existing spread plus ``x``. PV is monotone in ``x`` for any swap with a
    non-zero leg in the basis currency.
    """
    if swap.notional_base == 0.0:
        raise ValueError("par basis undefined for a swap with zero base notional")
    spreads = market.basis_spreads or ((swap.maturity, 0.0),)

    def pv(x: float) -> float:
        shifted = tuple((t, s + x) for t, s in spreads)
        return swap.value(market.replace(basis_spreads=shifted))

    lo, hi = -bracket_bp * 1e-4, bracket_bp * 1e-4
    flo, fhi = pv(lo), pv(hi)
    if flo * fhi > 0.0:
        raise ValueError(
            f"par basis not bracketed in [{-bracket_bp}, {bracket_bp}] bp "
            f"(PV({-bracket_bp}bp)={flo:.2f}, PV({bracket_bp}bp)={fhi:.2f})"
        )
    return float(brentq(pv, lo, hi, xtol=1e-16, rtol=8.9e-16))


# ---------------------------------------------------------------------------
# Dated cross-currency curve building (W13b)
# ---------------------------------------------------------------------------
#
# Plain idea. A EUR cash flow paid under a USD collateral agreement is not
# worth the same as one discounted on the EUR OIS curve: the market price of
# swapping EUR funding into USD funding, the cross-currency basis, says how
# much less. We build that "EUR in USD collateral" discount curve so that the
# quoted constant-notional EURUSD float/float basis swaps are worth zero.
#
# Formal statement (Fujii, Shimada and Takahashi 2010, "A note on
# construction of multiple swap curves with and without collateral", sec. 4;
# Henrard 2014, ch. 7). Per unit of each notional, with notional exchanges at
# the start s0 and end eN, the foreign leg is
#
#   V_f = -X(s0) + sum_j (F_j + q) tau_j X(p_j) + X(eN),
#   F_j tau_j = P_f(s_j)/P_f(e_j) - 1,
#
# with X the curve being built, P_f the foreign projection (OIS) curve and q
# the quoted basis on the foreign leg; the domestic leg V_d is the same with
# the domestic curve D in place of X and P_f and no spread. Spot-starting at
# the spot rate S, the swap is at par when
#
#   V_f / X(s0) = V_d / D(s0),
#
# independent of the notionals. Each quote adds the pillar eN (or its last
# payment day) and the pillar's log discount factor is solved by the same
# bisection on [ln 1e-8, ln 20] as ``fi_rates.bootstrap.bootstrap_dated``;
# the curve is log-linear in t = days/365 with the last forward held flat.
# The fx package does not depend on fi_rates, so the 40-line curve and the
# bisection are repeated here; the Rust and C++ ports share one copy each.
#
# Worked example: the ``eur_xccy`` block of
# ``python/equity/04-fixed-income/tests/data/dated_bootstrap_usd.txt``,
# written by QuantLib 1.44's ConstNotionalCrossCurrencyBasisSwapRateHelper;
# at -20 bp over 10 years the EUR-in-USD 10Y discount factor is QuantLib's
# to 1e-12 (``tests/test_xccy_bootstrap.py``).


_LN_DF_LO = math.log(1e-8)
_LN_DF_HI = math.log(20.0)
_DAYS_PER_YEAR = 365.0


class DfSource(Protocol):
    def df(self, day: float) -> float: ...


class XccyCurve:
    """Discount curve on integer day pillars, log-linear in t = days/365,
    flat forward beyond the last pillar (same rule as
    ``fi_rates.bootstrap.DatedCurve``)."""

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
        self._t = [0.0] + [d / _DAYS_PER_YEAR for d in self.days]
        self._x = [0.0] + [math.log(x) for x in self.dfs]

    def df(self, day: float) -> float:
        if not day >= 0.0:
            raise ValueError(f"day must be >= 0, got {day}")
        t = day / _DAYS_PER_YEAR
        ts, xs = self._t, self._x
        n = len(ts) - 1
        if t >= ts[n]:
            slope = (xs[n] - xs[n - 1]) / (ts[n] - ts[n - 1])
            return math.exp(xs[n] + slope * (t - ts[n]))
        lo, hi = 0, n
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ts[mid] <= t:
                lo = mid
            else:
                hi = mid
        w = (t - ts[lo]) / (ts[hi] - ts[lo])
        return math.exp(xs[lo] + w * (xs[hi] - xs[lo]))


@dataclass(frozen=True)
class XccyFloatPeriod:
    """Float coupon projected on ``[start, end]``, paid on ``pay`` (days)."""

    start: int
    end: int
    pay: int
    accrual: float


@dataclass(frozen=True)
class XccyBasisQuote:
    """Constant-notional float/float basis swap, spread ``basis`` on the
    foreign leg, notionals exchanged on ``start`` and ``end`` (days)."""

    basis: float
    start: int
    end: int
    foreign: tuple[XccyFloatPeriod, ...]
    domestic: tuple[XccyFloatPeriod, ...]
    label: str = ""

    def __post_init__(self) -> None:
        require_finite(basis=self.basis)
        if not (0 <= self.start < self.end) or not self.foreign or not self.domestic:
            raise ValueError(f"bad basis quote {self.label!r}")
        for p in (*self.foreign, *self.domestic):
            if not (0 <= p.start < p.end) or p.pay < 0 or not p.accrual > 0:
                raise ValueError(f"bad period {p!r} in {self.label!r}")

    @property
    def pillar(self) -> int:
        return max(self.end, *(p.pay for p in self.foreign))


def _leg_value(periods: Sequence[XccyFloatPeriod], spread: float, start: int, end: int,
               disc: DfSource, proj: DfSource) -> tuple[float, float]:
    """(value per unit notional incl. exchanges, annuity), both undivided."""
    v = -disc.df(start) + disc.df(end)
    ann = 0.0
    for p in periods:
        d = disc.df(p.pay)
        v += d * (proj.df(p.start) / proj.df(p.end) - 1.0 + spread * p.accrual)
        ann += p.accrual * d
    return v, ann


def xccy_basis_residual(q: XccyBasisQuote, foreign_discount: DfSource,
                        foreign_projection: DfSource, domestic: DfSource) -> float:
    """Par condition error in spread units: ``(V_f/X(s0) - V_d/D(s0)) / A_f``."""
    vf, af = _leg_value(q.foreign, q.basis, q.start, q.end, foreign_discount, foreign_projection)
    vd, _ = _leg_value(q.domestic, 0.0, q.start, q.end, domestic, domestic)
    x0, d0 = foreign_discount.df(q.start), domestic.df(q.start)
    return (vf / x0 - vd / d0) / (af / x0)


def _bisect(f: Callable[[float], float]) -> tuple[float, int]:
    lo, hi = _LN_DF_LO, _LN_DF_HI
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


@dataclass(frozen=True)
class XccyBootstrapResult:
    curve: XccyCurve
    residuals: tuple[float, ...]  # spread units, in pillar order
    iterations: tuple[int, ...]

    @property
    def max_abs_residual(self) -> float:
        return max(abs(r) for r in self.residuals)


def bootstrap_xccy_discount(quotes: Sequence[XccyBasisQuote], foreign_projection: DfSource,
                            domestic: DfSource) -> XccyBootstrapResult:
    """Foreign-currency discount curve under domestic collateral.

    ``foreign_projection`` is the foreign OIS (or index) curve the foreign
    coupons are projected on; ``domestic`` is the collateral currency's OIS
    curve. Quotes are sorted by pillar; duplicates raise. With every basis
    zero and a domestic leg at par, the result reproduces the foreign
    projection curve on the pillars (tested).
    """
    if not quotes:
        raise ValueError("no basis quotes supplied")
    ordered = sorted(quotes, key=lambda q: q.pillar)
    for a, b in zip(ordered, ordered[1:], strict=False):
        if a.pillar == b.pillar:
            raise ValueError(f"quotes {a.label!r} and {b.label!r} share pillar day {a.pillar}")
    days: list[int] = []
    dfs: list[float] = []
    its: list[int] = []
    for q in ordered:
        def f(x: float, _q: XccyBasisQuote = q) -> float:
            trial = XccyCurve([*days, _q.pillar], [*dfs, math.exp(x)])
            return xccy_basis_residual(_q, trial, foreign_projection, domestic)
        try:
            x, n = _bisect(f)
        except ValueError as exc:
            raise ValueError(f"xccy bootstrap failed at {q.label or q.pillar}: {exc}") from exc
        days.append(q.pillar)
        dfs.append(math.exp(x))
        its.append(n)
    curve = XccyCurve(days, dfs)
    res = tuple(xccy_basis_residual(q, curve, foreign_projection, domestic) for q in ordered)
    return XccyBootstrapResult(curve, res, tuple(its))
