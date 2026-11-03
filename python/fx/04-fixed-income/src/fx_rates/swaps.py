"""Dated FX swaps: near and far legs, swap points, tom-next, rolls, both
legs' mark-to-market, carry and uneven swaps.

Plain idea. An FX swap is two outright trades in opposite directions on
two value dates: buy euros for value on the near date and sell them back
for value on the far date (a buy-sell), or the reverse (a sell-buy). The
difference between the far and near rates is the swap points; at market
it equals the interest-rate differential over the period, by covered
interest parity. Banks use swaps to move a position from one value date
to another: tom-next (TN) rolls a spot position from tomorrow to spot, and
a roll moves a maturing forward to a later date.

Formal statement (Clark 2011, ch. 1-2; Wystup 2017, ch. 1). Quotes are
BASE/QUOTE, values are in the quote currency, and times are ACT/365F years
from the valuation date. With ``n`` base units bought on the near date
``T_n`` at ``K_n`` and ``n_f`` sold on the far date ``T_f`` at ``K_f``:

    V_near = n   * ( S_0 P_f(T_n) - K_n P_d(T_n) )
    V_far  = -n_f * ( S_0 P_f(T_f) - K_f P_d(T_f) )

where ``S_0 = S P_d(t_s) / P_f(t_s)`` is today's value of one base unit
delivered at spot (``MarketState.spot_today``) and ``P_d``, ``P_f`` are
the collateral-consistent discount curves. A leg dated before the
valuation date has settled and is worth nothing here. Each leg equals an
:class:`~fx_rates.fxforward.FXForward` (tested). The fair swap points
between two dates are ``(F(T_f) - F(T_n)) * factor`` with
``F(T) = S P_f(t_s, T) / P_d(t_s, T)``.

* Pricing (:func:`price_fx_swap`): the near rate is the market forward
  for the near date unless given; the far rate is the one that makes the
  swap worth zero,
  ``K_f = F(T_f) - V_near / (n_f P_d(T_f))``. With an off-market near
  rate the far rate absorbs the near leg's value with interest.
* Uneven swap: ``n_f != n``. It is an even swap plus an outright forward
  of ``n - n_f`` base on the far date (``outright_exposure``), so it has
  spot risk; an even swap has almost none.
* Tom-next (:func:`tom_next_swap`): near = the next business day,
  far = spot (``fx_rates.daycount.tenor_dates(..., "TN")``). Over a
  holiday weekend the points cover every night to spot.
* Roll (:func:`roll_position`): a long ``N`` base forward maturing on
  ``T_n`` is rolled to ``T_f`` by a sell-buy swap of ``N``. At market
  rates the near rate is the market forward and the profit on the old
  trade, ``N (K_n - K_orig)``, is paid on the near date
  (:meth:`DatedFXSwap.realised_on_near`). A historical-rate roll sets
  ``K_n = K_orig`` and moves that profit, with interest, into the far rate.
  It defers a realised result, so it is offered here for analysis only.
* Carry (:meth:`DatedFXSwap.carry`): the change in value from today to a
  horizon date with the spot rate and the zero curves (as functions of
  time to maturity) unchanged. A leg that settles before the horizon is
  counted as cash ``n (S - K)`` at the unchanged spot, undiscounted.

Worked example. Thursday 5 June 2025, EURUSD spot 1.0850 for Monday
9 June, flat 4.5% USD and 2.5% EUR. Tom is Friday 6 June and the TN points
are ``(S - F(tom)) * 1e4 = 1.0850 (1 - e^{-0.02 * 3/365}) * 1e4 = 1.78``.
Spot to 9 September (92 days) is 54.8 points.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from ._validation import require_finite
from .daycount import tenor_dates
from .fxforward import MarketState, _point_factor, forward_for_date, year_fraction_365

__all__ = [
    "DatedFXSwap",
    "fair_swap_points",
    "price_fx_swap",
    "roll_position",
    "tom_next_swap",
]


def _require_dated(market: MarketState) -> dt.date:
    if market.valuation_date is None:
        raise ValueError("dated FX swaps need a MarketState with valuation_date")
    return market.valuation_date


def _leg_value(market: MarketState, n: float, k: float, d: dt.date) -> float:
    """Value of buying ``n`` base at ``k`` for value ``d``; 0 if settled."""
    val = _require_dated(market)
    if d < val or n == 0.0:
        return 0.0
    t = year_fraction_365(val, d)
    return n * (market.spot_today * float(market.foreign_curve_adjusted.df(t))
                - k * float(market.domestic_curve_adjusted.df(t)))


@dataclass(frozen=True)
class DatedFXSwap:
    """FX swap on two value dates.

    ``notional_base > 0`` buys base on ``near_date`` at ``near_rate`` and
    sells ``far_notional_base`` (default the same amount) on ``far_date``
    at ``far_rate``: a buy-sell. A negative notional is a sell-buy.
    """

    near_date: dt.date
    far_date: dt.date
    near_rate: float
    far_rate: float
    notional_base: float
    far_notional_base: float | None = None
    pair: tuple[str, str] = ("EUR", "USD")
    label: str = ""

    def __post_init__(self) -> None:
        if self.pair[0] == self.pair[1]:
            raise ValueError(f"same-currency 'cross' {self.pair[0]}/{self.pair[1]} rejected")
        require_finite(near_rate=self.near_rate, far_rate=self.far_rate,
                       notional_base=self.notional_base, far_notional=self.far_notional)
        if self.near_rate <= 0.0 or self.far_rate <= 0.0:
            raise ValueError("FX swap rates must be > 0")
        if not self.near_date < self.far_date:
            raise ValueError(f"near date {self.near_date} must precede far date {self.far_date}")
        if not self.label:
            object.__setattr__(self, "label",
                               f"FXSwap {''.join(self.pair)} {self.near_date}/{self.far_date}")

    @property
    def far_notional(self) -> float:
        """Base amount sold on the far date (bought if negative)."""
        return self.notional_base if self.far_notional_base is None else self.far_notional_base

    @property
    def outright_exposure(self) -> float:
        """Net base bought over both legs, ``n - n_f``; zero for an even swap."""
        return self.notional_base - self.far_notional

    def contract_points(self, point_factor: float | None = None) -> float:
        """Contract swap points ``(K_f - K_n) * factor``."""
        pf = _point_factor(self.pair) if point_factor is None else point_factor
        return (self.far_rate - self.near_rate) * pf

    def points_per_day(self, point_factor: float | None = None) -> float:
        """Contract points per calendar day between the value dates."""
        return self.contract_points(point_factor) / (self.far_date - self.near_date).days

    def _check(self, market: MarketState) -> None:
        if market.pair != self.pair:
            raise ValueError(f"market pair {market.pair} does not match position pair {self.pair}")
        _require_dated(market)

    def leg_values(self, market: MarketState) -> tuple[float, float]:
        """(near, far) mark-to-market in the quote currency."""
        self._check(market)
        return (_leg_value(market, self.notional_base, self.near_rate, self.near_date),
                _leg_value(market, -self.far_notional, self.far_rate, self.far_date))

    def value(self, market: MarketState) -> float:
        """Mark-to-market of both legs in the quote currency."""
        near, far = self.leg_values(market)
        return near + far

    def realised_on_near(self, original_rate: float) -> float:
        """Cash on the near date from closing a forward dealt at
        ``original_rate`` with this swap's near leg (rolls):
        ``-n (K_n - K_orig)`` in the quote currency."""
        require_finite(original_rate=original_rate)
        return -self.notional_base * (self.near_rate - original_rate)

    def carry(self, market: MarketState, horizon: dt.date) -> float:
        """Value change to ``horizon`` with spot and zero curves unchanged.

        Legs that settle in ``[valuation_date, horizon)`` count as cash
        ``n (S - K)`` at the unchanged spot, undiscounted.
        """
        self._check(market)
        val = _require_dated(market)
        if horizon < val:
            raise ValueError(f"horizon {horizon} precedes valuation date {val}")
        later = market.replace(valuation_date=horizon, spot_time=None)
        settled = sum(n * (market.spot - k)
                      for n, k, d in ((self.notional_base, self.near_rate, self.near_date),
                                      (-self.far_notional, self.far_rate, self.far_date))
                      if val <= d < horizon)
        return self.value(later) + settled - self.value(market)


def fair_swap_points(market: MarketState, near_date: dt.date, far_date: dt.date,
                     point_factor: float | None = None) -> float:
    """Market swap points between two value dates, ``(F(T_f) - F(T_n)) * factor``."""
    if not near_date < far_date:
        raise ValueError(f"near date {near_date} must precede far date {far_date}")
    pf = _point_factor(market.pair) if point_factor is None else point_factor
    return (forward_for_date(market, far_date) - forward_for_date(market, near_date)) * pf


def price_fx_swap(market: MarketState, near_date: dt.date, far_date: dt.date,
                  notional_base: float, far_notional_base: float | None = None,
                  near_rate: float | None = None) -> DatedFXSwap:
    """FX swap worth zero today.

    ``near_rate`` defaults to the market forward for ``near_date``; the far
    rate is solved: ``K_f = F(T_f) - V_near / (n_f P_d(T_f))``.
    """
    val = _require_dated(market)
    kn = forward_for_date(market, near_date) if near_rate is None else float(near_rate)
    nf = notional_base if far_notional_base is None else far_notional_base
    if nf == 0.0:
        raise ValueError("far notional must be non-zero to solve the far rate")
    if far_date < val:
        raise ValueError(f"far date {far_date} precedes valuation date {val}")
    v_near = _leg_value(market, notional_base, kn, near_date)
    tf = year_fraction_365(val, far_date)
    kf = forward_for_date(market, far_date) - v_near / (
        nf * float(market.domestic_curve_adjusted.df(tf)))
    return DatedFXSwap(near_date, far_date, kn, kf, notional_base, far_notional_base,
                       pair=market.pair)


def tom_next_swap(market: MarketState, notional_base: float) -> DatedFXSwap:
    """At-market tom-next swap: near = next business day, far = spot."""
    val = _require_dated(market)
    tom, spot = tenor_dates(val, "TN", market.pair)
    return price_fx_swap(market, tom, spot, notional_base)


def roll_position(market: MarketState, maturing_date: dt.date, new_date: dt.date,
                  position_base: float, original_rate: float,
                  historical: bool = False) -> DatedFXSwap:
    """Swap that rolls a forward of ``position_base`` (long if > 0), dealt at
    ``original_rate`` and maturing on ``maturing_date``, to ``new_date``.

    At market (default) the near rate is the market forward and the old
    trade's result is paid on the near date; ``historical=True`` deals the
    near leg at ``original_rate`` and carries the result into the far rate.
    """
    require_finite(original_rate=original_rate, position_base=position_base)
    return price_fx_swap(market, maturing_date, new_date, -position_base,
                         near_rate=original_rate if historical else None)
