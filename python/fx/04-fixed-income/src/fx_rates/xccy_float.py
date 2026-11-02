"""Float-float cross-currency swaps: constant notional and mark-to-market reset.

Each leg pays its own floating index plus a spread, projected on that
currency's projection curve and discounted on its collateral-consistent
discount curve (multi-curve; Fujii, Shimada and Takahashi 2010, "A note on
construction of multiple swap curves with and without collateral", FSA
Research Review 6, Section 3). Values are in the quote currency.

Constant-notional leg, notional ``N`` (initial and final exchange):

    PV = N * ( -D(t_0) + sum_i D(t_i) tau (L_i + s) + D(t_n) ),
    L_i = (P(t_{i-1}) / P(t_i) - 1) / tau.

Mark-to-market (resetting) quote leg: at each reset ``t_{i-1}`` the quote
notional is set to ``N_base * S(t_{i-1})``, the difference being exchanged
then. That is a strip of one-period exchanges: pay ``S(t_{i-1})`` at
``t_{i-1}``, get back ``S(t_{i-1}) (1 + tau (L_i + s))`` at ``t_i``.

Write ``1 + tau L_i = beta_i / P_q(t_{i-1}, t_i)`` with the deterministic
spread ``beta_i = (P^proj(t_{i-1}) / P^proj(t_i)) (D_q(t_i) / D_q(t_{i-1}))``
(Henrard 2010, Wilmott Journal 2(6), Section 3). The ``beta_i`` part of the
return is worth exactly ``beta_i S_0 D_b(t_{i-1})``: dividing by the
quote bond turns the payment at ``t_i`` back into one at ``t_{i-1}``. Only
the spread part ``S(t_{i-1}) tau s`` is a payment fixed at one date and paid
at a later one, and that needs a convexity adjustment:

    value = D_q(t_i) E^{T_i}[S(t_{i-1})] tau s
          = c_i S_0 D_b(t_{i-1}) (D_q(t_i) / D_q(t_{i-1})) tau s.

Convexity model (``MtmConvexity``): spot lognormal with vol ``sigma_S``,
the quote short rate Hull-White with mean reversion ``a`` and normal vol
``sigma_r``, correlation ``rho`` between the two Brownian motions, base
rates deterministic. Changing measure from ``T_{i-1}`` to ``T_i`` with the
Gaussian change-of-numeraire rule (Brigo and Mercurio 2006, "Interest Rate
Models - Theory and Practice", 2nd ed., Sections 2.3 and 3.3; the FX/rates
correlation effect on MtM swaps is Fujii, Shimada and Takahashi 2010,
Section 4) gives, with ``t = t_{i-1}``, ``B = (1 - e^{-a tau}) / a``,

    ln c_i = -B * ( rho sigma_S sigma_r I_1(t) + sigma_r^2 I_2(t) ),
    I_1(t) = (1 - e^{-a t}) / a,
    I_2(t) = (I_1(t) - (1 - e^{-2 a t}) / (2 a)) / a.

The first term is the FX/rates correlation; the second comes from the
forward FX rate's own exposure to quote rates and is there even at
``rho = 0``. A consequence worth knowing: on a single curve with zero
spread the MtM leg is worth exactly zero whatever the vols, because the
adjustment only touches the spread. With ``convexity=None`` every
``c_i = 1``, the deterministic-curve value.

Seasoned trades (``elapsed > 0``). The schedule is shifted so today is
``t = 0``. Periods that start at or after today are valued as above.
The period running through today has its index rates and, for the MtM
leg, its FX notional already fixed, so its flows are known amounts:
``N_base * fx_fixing * (1 + tau (quote_fixing + s_q))`` at ``t_1`` on the MtM
leg, coupons from ``base_fixing`` / ``quote_fixing`` on the others. The
initial exchange of a constant-notional leg is in the past and dropped;
the final exchange stays. On a reset date of a seasoned trade, today's
flows count as settled: the new MtM period's notional is today's spot and
only its return leg is valued, which keeps the value continuous through
the reset (tested).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class XccyCurves:
    """Spot (quote per base) and the four curves; any object with ``.df(t)`` works."""

    spot: float
    base_projection: object
    base_discount: object
    quote_projection: object
    quote_discount: object

    def __post_init__(self) -> None:
        if not np.isfinite(self.spot) or self.spot <= 0.0:
            raise ValueError(f"spot must be finite and > 0, got {self.spot}")


@dataclass(frozen=True)
class MtmConvexity:
    """Parameters of the MtM convexity adjustment (see module docstring)."""

    fx_vol: float
    rate_vol: float
    mean_reversion: float
    correlation: float

    def __post_init__(self) -> None:
        if self.fx_vol < 0.0 or self.rate_vol < 0.0:
            raise ValueError("vols must be >= 0")
        if self.mean_reversion <= 0.0:
            raise ValueError("mean_reversion must be > 0")
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError("correlation must lie in [-1, 1]")

    def factor(self, t_fix: float, tau: float) -> float:
        """``c = E^{T_fix + tau}[S(t_fix)] / F(0, t_fix)``."""
        if t_fix <= 0.0:
            return 1.0
        a = self.mean_reversion
        B = (1.0 - np.exp(-a * tau)) / a
        i1 = (1.0 - np.exp(-a * t_fix)) / a
        i2 = (i1 - (1.0 - np.exp(-2.0 * a * t_fix)) / (2.0 * a)) / a
        cov = self.correlation * self.fx_vol * self.rate_vol * i1 + self.rate_vol**2 * i2
        return float(np.exp(-B * cov))


def _df(curve: Any, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.where(t <= 0.0, 1.0, np.asarray(curve.df(np.maximum(t, 1e-12)), dtype=float))


@dataclass(frozen=True)
class FloatFloatXccy:
    """Float-float cross-currency swap.

    ``receive_base=True``: receive the base leg, pay the quote leg.
    ``mtm_reset=True``: the quote leg's notional resets to ``N_base * S(t)``
    each period; otherwise it is ``notional_quote`` throughout, fixed at
    trade. ``notional_quote=None`` means ``N_base * spot`` of the curves
    passed in, i.e. a swap valued on its trade date.

    ``elapsed`` is the time since the trade started; when today falls inside
    a period, that period's fixings must be given: ``base_fixing`` and
    ``quote_fixing`` (simple index rates) and, for an MtM trade,
    ``fx_fixing`` (the spot set at the last reset). ``convexity`` switches on
    the MtM convexity adjustment.
    """

    notional_base: float
    maturity: float
    frequency: int = 4
    spread_base: float = 0.0
    spread_quote: float = 0.0
    mtm_reset: bool = False
    receive_base: bool = True
    notional_quote: float | None = None
    convexity: MtmConvexity | None = None
    elapsed: float = 0.0
    base_fixing: float | None = None
    quote_fixing: float | None = None
    fx_fixing: float | None = None

    def __post_init__(self) -> None:
        if not (self.notional_base > 0.0 and self.maturity > 0.0):
            raise ValueError("notional_base and maturity must be > 0")
        if self.frequency < 1 or int(self.frequency) != self.frequency:
            raise ValueError("frequency must be a positive integer")
        n = self.maturity * self.frequency
        if abs(n - round(n)) > 1e-9:
            raise ValueError("maturity * frequency must be integral")
        if not 0.0 <= self.elapsed < self.maturity:
            raise ValueError("elapsed must lie in [0, maturity)")
        if self._in_period():
            if self.base_fixing is None or self.quote_fixing is None:
                raise ValueError("a seasoned trade inside a period needs base_fixing and quote_fixing")
            if self.mtm_reset and (self.fx_fixing is None or self.fx_fixing <= 0.0):
                raise ValueError("a seasoned MtM trade inside a period needs fx_fixing > 0")

    # ------------------------------------------------------------ schedule
    def _all_times(self) -> np.ndarray:
        return np.arange(0, int(round(self.maturity * self.frequency)) + 1) / self.frequency

    def _in_period(self) -> bool:
        on_reset = bool(np.any(np.abs(self._all_times() - self.elapsed) < 1e-12))
        return self.elapsed > 0.0 and not on_reset

    def times(self) -> np.ndarray:
        """Remaining schedule from today; the first entry is <= 0 (the
        current period's start, or today itself on a reset date)."""
        t = self._all_times() - self.elapsed
        t[np.abs(t) < 1e-12] = 0.0
        k = int(np.searchsorted(t, 0.0, side="right")) - 1
        return t[k:]

    # ---------------------------------------------------------------- legs
    def _leg_constant(self, t: np.ndarray, proj: Any, disc: Any, spread: float,
                      fixing: float | None) -> float:
        """Constant-notional leg per unit notional."""
        tau = 1.0 / self.frequency
        p = _df(proj, t)
        d = _df(disc, t)
        coupons = d[1:] * ((p[:-1] / p[1:] - 1.0) + spread * tau)
        if t[0] < 0.0:
            assert fixing is not None
            coupons[0] = d[1] * tau * (fixing + spread)
        initial = -d[0] if self.elapsed == 0.0 else 0.0
        return float(initial + np.sum(coupons) + d[-1])

    def base_leg_pv(self, c: XccyCurves) -> float:
        """Base leg in quote currency."""
        return c.spot * self.notional_base * self._leg_constant(
            self.times(), c.base_projection, c.base_discount, self.spread_base, self.base_fixing)

    def quote_leg_pv(self, c: XccyCurves) -> float:
        t = self.times()
        if not self.mtm_reset:
            nq = c.spot * self.notional_base if self.notional_quote is None else self.notional_quote
            return nq * self._leg_constant(
                t, c.quote_projection, c.quote_discount, self.spread_quote, self.quote_fixing)
        tau = 1.0 / self.frequency
        db = _df(c.base_discount, t)
        dq = _df(c.quote_discount, t)
        pp = _df(c.quote_projection, t)
        total = 0.0
        for i in range(1, t.size):
            if t[i - 1] < 0.0:  # current period: notional and coupon fixed
                assert self.fx_fixing is not None and self.quote_fixing is not None
                total += self.fx_fixing * dq[i] * (1.0 + tau * (self.quote_fixing + self.spread_quote))
                continue
            beta = (pp[i - 1] / pp[i]) * (dq[i] / dq[i - 1])
            ci = 1.0 if self.convexity is None else self.convexity.factor(float(t[i - 1]), tau)
            # on a reset date of a seasoned trade, today's exchange is settled
            start = 0.0 if (t[i - 1] == 0.0 and self.elapsed > 0.0) else -1.0
            total += c.spot * db[i - 1] * (start + beta
                                           + ci * tau * self.spread_quote * dq[i] / dq[i - 1])
        return float(self.notional_base * total)

    def value(self, c: XccyCurves) -> float:
        sign = 1.0 if self.receive_base else -1.0
        return sign * (self.base_leg_pv(c) - self.quote_leg_pv(c))


def par_base_spread(swap: FloatFloatXccy, c: XccyCurves) -> float:
    """Spread on the base leg that sets the swap value to zero."""
    def f(s: float) -> float:
        return replace(swap, spread_base=s).value(c)
    return float(brentq(f, -0.5, 0.5, xtol=1e-15))
