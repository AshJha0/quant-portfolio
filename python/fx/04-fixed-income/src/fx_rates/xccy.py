"""Fixed-fixed cross-currency swap with notional exchange.

Structure (receive_base = True, e.g. receive EUR / pay USD on EURUSD):

- optional initial exchange at t = 0: pay ``notional_base`` (base ccy),
  receive ``notional_quote`` (quote ccy);
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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ._validation import require_finite
from .fxforward import MarketState

__all__ = [
    "CrossCurrencySwap",
    "solve_par_rate_quote",
    "solve_par_rate_base",
    "solve_par_basis",
]


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
        Include the t = 0 notional exchange (at-inception valuation).  At
        inception with ``notional_quote = notional_base * spot`` the initial
        exchange is worth zero, so seasoned valuation usually omits it.
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

    def __post_init__(self) -> None:
        if self.pair[0] == self.pair[1]:
            raise ValueError(
                f"same-currency 'cross' {self.pair[0]}/{self.pair[1]} rejected"
            )
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

    def cashflows(self) -> list[tuple[str, float, float]]:
        """All cashflows as (currency, time, signed amount) from the
        perspective of the position holder."""
        sign = 1.0 if self.receive_base else -1.0
        base, quote = self.pair
        cfs: list[tuple[str, float, float]] = []
        if self.include_initial_exchange:
            cfs.append((base, 0.0, -sign * self.notional_base))
            cfs.append((quote, 0.0, sign * self.notional_quote))
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
        dfs = market.domestic_curve.df(times)
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
        v = sign * (market.spot * self.leg_pv_base(market) - self.leg_pv_quote(market))
        if self.include_initial_exchange:
            v += sign * (self.notional_quote - market.spot * self.notional_base)
        return v


# ---------------------------------------------------------------------- #
# par solvers
# ---------------------------------------------------------------------- #
def solve_par_rate_quote(swap: CrossCurrencySwap, market: MarketState) -> float:
    """Quote-leg fixed rate that makes the (seasoned) swap PV exactly zero.

    Closed form — PV is linear in the coupon:
    ``c_d* = (S * PV_base - N_d * DF_d(T)) / (N_d * tau * sum DF_d(t_i))``.
    """
    times = swap.payment_times()
    dfs = market.domestic_curve.df(times)
    tau = 1.0 / swap.frequency
    annuity = swap.notional_quote * tau * float(np.sum(dfs))
    if annuity <= 0.0:
        raise ValueError("cannot solve par rate for zero quote notional")
    target = market.spot * swap.leg_pv_base(market)
    return (target - swap.notional_quote * float(dfs[-1])) / annuity


def solve_par_rate_base(swap: CrossCurrencySwap, market: MarketState) -> float:
    """Base-leg fixed rate that makes the (seasoned) swap PV exactly zero."""
    times = swap.payment_times()
    dfs = market.foreign_curve_adjusted.df(times)
    tau = 1.0 / swap.frequency
    annuity = swap.notional_base * tau * float(np.sum(dfs))
    if annuity <= 0.0:
        raise ValueError("cannot solve par rate for zero base notional")
    target = swap.leg_pv_quote(market) / market.spot
    return (target - swap.notional_base * float(dfs[-1])) / annuity


def solve_par_basis(
    swap: CrossCurrencySwap,
    market: MarketState,
    bracket_bp: float = 1000.0,
) -> float:
    """Flat shift (decimal) to the basis spread curve that zeroes the swap PV.

    Solves ``PV(basis + x) = 0`` with Brent's method to ~1e-14 in ``x``.
    Returns the shift ``x``; the *level* of the par basis at tenor T is the
    existing spread plus ``x``.  PV is monotone in ``x`` for any swap with a
    non-zero base leg (all base-leg cashflows scale by ``exp(-x t)``).
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
