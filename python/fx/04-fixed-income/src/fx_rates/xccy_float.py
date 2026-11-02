"""Float-float cross-currency swaps: constant notional and mark-to-market reset.

Each leg pays its own floating index plus a spread, projected on that
currency's projection curve and discounted on its collateral-consistent
discount curve (multi-curve; Fujii, Shimada and Takahashi 2010, "A note on
construction of multiple swap curves with and without collateral", FSA
Research Review 6, Section 3). Values are in the quote currency.

Constant-notional leg, notional ``N`` (initial and final exchange):

    PV = N * ( -D(t_0) + sum_i D(t_i) tau (L_i + s) + D(t_n) ),
    L_i = (P(t_{i-1}) / P(t_i) - 1) / tau.

Mark-to-market (resetting) quote leg: at each reset ``t_{i-1}`` its
notional is set to ``N_base * S(t_{i-1})``, the difference being exchanged
then. That is a strip of one-period exchanges; with deterministic curves
the value of ``S(t_{i-1})`` paid at ``t_{i-1}`` is
``S_0 D_base(t_{i-1})`` (CIP on the discount curves), so

    PV_mtm = N_base * S_0 * sum_i D_base(t_{i-1})
             * ( -1 + (D_q(t_i) / D_q(t_{i-1})) (1 + tau (L^q_i + s_q)) ).

Assumption: no FX/rates correlation (the MtM convexity adjustment of
Fujii et al., Section 4, is ignored). Consequence (tested): the MtM swap's
value is proportional to spot, so at its par spread it has no FX delta,
which is why the market trades the resetting form.
"""

from __future__ import annotations

from dataclasses import dataclass
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
class FloatFloatXccy:
    """Float-float cross-currency swap.

    ``receive_base=True``: receive the base leg, pay the quote leg.
    ``mtm_reset=True``: the quote leg's notional resets to ``N_base * S(t)``
    each period; otherwise it is ``notional_quote`` throughout, fixed at
    trade. ``notional_quote=None`` means ``N_base * spot`` of the curves
    passed in, i.e. a swap valued on its trade date.
    """

    notional_base: float
    maturity: float
    frequency: int = 4
    spread_base: float = 0.0
    spread_quote: float = 0.0
    mtm_reset: bool = False
    receive_base: bool = True
    notional_quote: float | None = None

    def __post_init__(self) -> None:
        if not (self.notional_base > 0.0 and self.maturity > 0.0):
            raise ValueError("notional_base and maturity must be > 0")
        if self.frequency < 1 or int(self.frequency) != self.frequency:
            raise ValueError("frequency must be a positive integer")
        n = self.maturity * self.frequency
        if abs(n - round(n)) > 1e-9:
            raise ValueError("maturity * frequency must be integral")

    def times(self) -> np.ndarray:
        return np.arange(0, int(round(self.maturity * self.frequency)) + 1) / self.frequency

    @staticmethod
    def _period_values(t: np.ndarray, proj, disc, spread: float) -> np.ndarray:
        """Value at t_{i-1}, per unit notional, of the one-period exchange
        (-1 at t_{i-1}, 1 + tau (L_i + s) at t_i), discounted to time 0."""
        tau = t[1] - t[0]
        p = np.asarray(proj.df(t), dtype=float)
        d = np.asarray(disc.df(t), dtype=float)
        d = np.where(t <= 0.0, 1.0, d)
        p = np.where(t <= 0.0, 1.0, p)
        coupon = p[:-1] / p[1:] - 1.0 + spread * tau
        return -d[:-1] + d[1:] * (1.0 + coupon)

    def _leg_constant(self, t, proj, disc, spread: float) -> float:
        tau = t[1] - t[0]
        p = np.where(t <= 0.0, 1.0, np.asarray(proj.df(t), dtype=float))
        d = np.where(t <= 0.0, 1.0, np.asarray(disc.df(t), dtype=float))
        coupons = np.sum(d[1:] * (p[:-1] / p[1:] - 1.0 + spread * tau))
        return float(-d[0] + coupons + d[-1])

    def base_leg_pv(self, c: XccyCurves) -> float:
        """Base leg in quote currency."""
        t = self.times()
        return c.spot * self.notional_base * self._leg_constant(
            t, c.base_projection, c.base_discount, self.spread_base)

    def quote_leg_pv(self, c: XccyCurves) -> float:
        t = self.times()
        if not self.mtm_reset:
            nq = c.spot * self.notional_base if self.notional_quote is None else self.notional_quote
            return nq * self._leg_constant(
                t, c.quote_projection, c.quote_discount, self.spread_quote)
        # value at 0 of S(t_{i-1}) paid at t_{i-1} is S_0 D_base(t_{i-1})
        bd: Any = c.base_discount
        qd: Any = c.quote_discount
        db = np.where(t[:-1] <= 0.0, 1.0, np.asarray(bd.df(t[:-1]), dtype=float))
        dq = np.where(t[:-1] <= 0.0, 1.0, np.asarray(qd.df(t[:-1]), dtype=float))
        per = self._period_values(t, c.quote_projection, c.quote_discount, self.spread_quote)
        return float(c.spot * self.notional_base * np.sum(db / dq * per))

    def value(self, c: XccyCurves) -> float:
        sign = 1.0 if self.receive_base else -1.0
        return sign * (self.base_leg_pv(c) - self.quote_leg_pv(c))


def par_base_spread(swap: FloatFloatXccy, c: XccyCurves) -> float:
    """Spread on the base leg that sets the swap value to zero."""
    def f(s: float) -> float:
        return FloatFloatXccy(swap.notional_base, swap.maturity, swap.frequency, s,
                              swap.spread_quote, swap.mtm_reset, swap.receive_base,
                              swap.notional_quote).value(c)
    return float(brentq(f, -0.5, 0.5, xtol=1e-15))
