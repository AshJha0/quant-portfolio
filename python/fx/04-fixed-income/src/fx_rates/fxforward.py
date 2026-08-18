"""FX forwards via covered interest parity, forward points, outright MTM and
FX swaps, plus the ``MarketState`` container all pricers consume.

FX conventions (portfolio-wide, see CONVENTIONS.md)
---------------------------------------------------
Pairs are quoted BASE/QUOTE: EURUSD = USD per 1 EUR.  The **domestic**
currency is the *quote* currency (USD for EURUSD) and the **foreign**
currency is the *base* currency (EUR).  Covered interest parity (CIP):

    F(T) = S * DF_f(T) / DF_d(T)

Market forwards embed the cross-currency basis; with the basis spread
``s(T)`` (decimal, negative for EURUSD post-2008):

    F_mkt(T) = S * DF_f(T) * exp(-s(T) * T) / DF_d(T)
             = S * DF_f_adj(T) / DF_d(T).

All position values are expressed in the **quote (domestic) currency**.
A positive ``notional_base`` means long the base currency forward
(receive base, pay quote at expiry).
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from functools import cached_property
from typing import Sequence

import numpy as np
import pandas as pd

from .bootstrap import basis_adjusted_curve
from .curve import DiscountCurve

__all__ = [
    "MarketState",
    "cip_forward",
    "market_forward",
    "forward_points",
    "forward_points_table",
    "FXForward",
    "FXSwap",
]

POINT_FACTORS = {"EURUSD": 1e4, "USDJPY": 1e2, "EURJPY": 1e2}
_DEFAULT_POINT_FACTOR = 1e4


@dataclass(eq=False)
class MarketState:
    """Market snapshot for one currency pair.

    Attributes
    ----------
    spot : float
        Spot rate, quote currency per 1 base currency (e.g. EURUSD 1.0850).
        Interpreted at t = 0 (the T+2 spot-settlement subtlety is absorbed
        into the curves — assumption A6).
    domestic_curve : DiscountCurve
        Quote-currency discount curve (USD for EURUSD).
    foreign_curve : DiscountCurve
        Base-currency discount curve built from that currency's *own*
        deposits/swaps (pure CIP curve, no basis).
    basis_spreads : tuple of (T, s)
        Cross-currency basis spread quotes in decimal (e.g. (5.0, -0.0025)).
        Empty tuple = pure CIP pricing.
    pair : (base, quote)
        Currency codes, e.g. ("EUR", "USD").  Must differ.
    """

    spot: float
    domestic_curve: DiscountCurve
    foreign_curve: DiscountCurve
    basis_spreads: tuple[tuple[float, float], ...] = ()
    pair: tuple[str, str] = ("EUR", "USD")

    def __post_init__(self) -> None:
        if self.spot <= 0.0:
            raise ValueError(f"spot must be > 0, got {self.spot}")
        base, quote = self.pair
        if base == quote:
            raise ValueError(
                f"same-currency 'cross' {base}/{quote} rejected: base and quote must differ"
            )
        self.basis_spreads = tuple((float(t), float(s)) for t, s in self.basis_spreads)

    @cached_property
    def foreign_curve_adjusted(self) -> DiscountCurve:
        """Basis-adjusted foreign discount curve (== foreign_curve if no basis)."""
        return basis_adjusted_curve(self.foreign_curve, self.basis_spreads)

    def replace(self, **kwargs) -> "MarketState":
        """Return a copy with fields replaced (cached curve is rebuilt)."""
        return _dc_replace(self, **kwargs)

    @property
    def base_ccy(self) -> str:
        return self.pair[0]

    @property
    def quote_ccy(self) -> str:
        return self.pair[1]


# ---------------------------------------------------------------------- #
# forward curve
# ---------------------------------------------------------------------- #
def cip_forward(
    spot: float,
    domestic_curve: DiscountCurve,
    foreign_curve: DiscountCurve,
    expiry: float,
):
    """Pure covered-interest-parity forward ``F(T) = S * DF_f(T) / DF_d(T)``.

    ``expiry`` may be a scalar or array of year fractions (> 0 not required;
    T = 0 returns spot).
    """
    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot}")
    t = np.asarray(expiry, dtype=float)
    out = spot * np.asarray(foreign_curve.df(t)) / np.asarray(domestic_curve.df(t))
    return float(out) if t.ndim == 0 else out


def market_forward(market: MarketState, expiry) -> float:
    """Basis-consistent market forward ``S * DF_f_adj(T) / DF_d(T)``."""
    return cip_forward(
        market.spot, market.domestic_curve, market.foreign_curve_adjusted, expiry
    )


def forward_points(
    market: MarketState, expiry, point_factor: float | None = None
):
    """Forward points ``(F_mkt - S) * factor`` (factor 1e4 for EURUSD pips)."""
    if point_factor is None:
        point_factor = POINT_FACTORS.get("".join(market.pair), _DEFAULT_POINT_FACTOR)
    f = market_forward(market, expiry)
    return (f - market.spot) * point_factor


def forward_points_table(
    market: MarketState,
    tenors: Sequence[float],
    point_factor: float | None = None,
) -> pd.DataFrame:
    """Forward points table: CIP vs market (basis-adjusted) forwards.

    Columns: tenor (y), CIP forward, market forward, CIP points, market
    points, basis effect in points, implied basis spread in bp.
    """
    if point_factor is None:
        point_factor = POINT_FACTORS.get("".join(market.pair), _DEFAULT_POINT_FACTOR)
    rows = []
    for t in tenors:
        f_cip = cip_forward(market.spot, market.domestic_curve, market.foreign_curve, t)
        f_mkt = market_forward(market, t)
        s_bp = -np.log(f_mkt / f_cip) / t * 1e4 if t > 0 else 0.0
        rows.append(
            {
                "tenor_y": t,
                "cip_forward": f_cip,
                "market_forward": f_mkt,
                "cip_points": (f_cip - market.spot) * point_factor,
                "market_points": (f_mkt - market.spot) * point_factor,
                "basis_points_effect": (f_mkt - f_cip) * point_factor,
                "basis_spread_bp": s_bp,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- #
# instruments
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class FXForward:
    """Outright FX forward.

    Long ``notional_base`` units of the base currency at ``strike`` (quote
    per base), settling at ``expiry`` (years).  ``notional_base`` may be
    negative (short base) or zero.

    Valuation (in quote currency), two equivalent methods:

    - ``"cashflows"``: two discounted cashflows,
      ``V = N * S * DF_f_adj(T) - N * K * DF_d(T)``.
    - ``"forward"``: forward-vs-forward,
      ``V = N * DF_d(T) * (F_mkt(T) - K)``.

    The two are algebraically identical (identity-tested to 1e-10).
    """

    notional_base: float
    strike: float
    expiry: float
    pair: tuple[str, str] = ("EUR", "USD")
    label: str = ""

    def __post_init__(self) -> None:
        if self.pair[0] == self.pair[1]:
            raise ValueError(
                f"same-currency 'cross' {self.pair[0]}/{self.pair[1]} rejected"
            )
        if self.strike <= 0.0:
            raise ValueError(f"strike must be > 0, got {self.strike}")
        if self.expiry <= 0.0:
            raise ValueError(f"expiry must be > 0, got {self.expiry}")
        if not self.label:
            object.__setattr__(
                self, "label", f"FXFwd {''.join(self.pair)} {self.expiry:.2f}y"
            )

    def _check_pair(self, market: MarketState) -> None:
        if market.pair != self.pair:
            raise ValueError(
                f"market pair {market.pair} does not match position pair {self.pair}"
            )

    def cashflows(self) -> list[tuple[str, float, float]]:
        """Cashflows as (currency, time, amount): receive base, pay quote."""
        return [
            (self.pair[0], self.expiry, self.notional_base),
            (self.pair[1], self.expiry, -self.notional_base * self.strike),
        ]

    def value(self, market: MarketState, method: str = "cashflows") -> float:
        """Mark-to-market in the quote (domestic) currency."""
        self._check_pair(market)
        n, k, t = self.notional_base, self.strike, self.expiry
        df_d = market.domestic_curve.df(t)
        df_f = market.foreign_curve_adjusted.df(t)
        if method == "cashflows":
            return n * market.spot * df_f - n * k * df_d
        if method == "forward":
            return n * df_d * (market_forward(market, t) - k)
        raise ValueError(f"unknown valuation method {method!r}")


@dataclass(frozen=True)
class FXSwap:
    """FX swap: buy-sell (or sell-buy) the base currency.

    ``notional_base > 0``: buy ``notional_base`` base at the near date at
    ``near_strike``, sell it back at the far date at ``far_strike``
    (buy-sell); negative notional flips both legs.  Equal to the sum of two
    outright forwards with opposite notionals (identity-tested).
    """

    notional_base: float
    near_strike: float
    near_expiry: float
    far_strike: float
    far_expiry: float
    pair: tuple[str, str] = ("EUR", "USD")
    label: str = ""

    def __post_init__(self) -> None:
        if self.pair[0] == self.pair[1]:
            raise ValueError(
                f"same-currency 'cross' {self.pair[0]}/{self.pair[1]} rejected"
            )
        if self.near_strike <= 0.0 or self.far_strike <= 0.0:
            raise ValueError("FX swap strikes must be > 0")
        if not (0.0 < self.near_expiry < self.far_expiry):
            raise ValueError(
                f"require 0 < near_expiry < far_expiry, got "
                f"{self.near_expiry}, {self.far_expiry}"
            )
        if not self.label:
            object.__setattr__(
                self,
                "label",
                f"FXSwap {''.join(self.pair)} "
                f"{self.near_expiry:.2f}y/{self.far_expiry:.2f}y",
            )

    def legs(self) -> tuple[FXForward, FXForward]:
        """Decompose into (near, far) outright forwards."""
        near = FXForward(
            self.notional_base, self.near_strike, self.near_expiry, self.pair,
            label=self.label + " near",
        )
        far = FXForward(
            -self.notional_base, self.far_strike, self.far_expiry, self.pair,
            label=self.label + " far",
        )
        return near, far

    def value(self, market: MarketState) -> float:
        """MTM in quote currency — explicit four-cashflow sum (independent of
        the two-forward decomposition, which is tested against this)."""
        if market.pair != self.pair:
            raise ValueError(
                f"market pair {market.pair} does not match position pair {self.pair}"
            )
        n = self.notional_base
        s = market.spot
        dfd, dff = market.domestic_curve.df, market.foreign_curve_adjusted.df
        v = 0.0
        v += n * s * dff(self.near_expiry)                     # receive base near
        v -= n * self.near_strike * dfd(self.near_expiry)      # pay quote near
        v -= n * s * dff(self.far_expiry)                      # pay base far
        v += n * self.far_strike * dfd(self.far_expiry)        # receive quote far
        return v
