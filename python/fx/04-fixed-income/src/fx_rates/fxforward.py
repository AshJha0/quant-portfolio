"""FX forwards via covered interest parity, forward points, outright MTM and
FX swaps, plus the ``MarketState`` container all pricers consume.

FX conventions (portfolio-wide, see CONVENTIONS.md)
---------------------------------------------------
Pairs are quoted BASE/QUOTE: EURUSD = USD per 1 EUR.  The **domestic**
currency is the *quote* currency (USD for EURUSD) and the **foreign**
currency is the *base* currency (EUR).  Covered interest parity (CIP):

    F(T) = S * DF_f(T) / DF_d(T)

Market forwards embed the cross-currency basis. Cashflows are discounted
on collateral-consistent curves: the collateral currency (USD by default)
on its own curve, the other currency on the curve bootstrapped from
float-float basis-swap quotes (``bootstrap.basis_discount_curve``):

    F_mkt(T) = S * D_base(T) / D_quote(T).

For EURUSD the EUR curve carries the basis; for USDJPY it is the JPY
(quote) curve, not the USD one.

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

from ._validation import require_finite
import warnings

from .bootstrap import basis_adjusted_curve, basis_discount_curve
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


def _point_factor(pair: tuple[str, str]) -> float:
    """Pip factor: 1e2 for JPY-quoted pairs, 1e4 otherwise, unless listed."""
    key = "".join(pair)
    if key in POINT_FACTORS:
        return POINT_FACTORS[key]
    return 1e2 if pair[1] == "JPY" else _DEFAULT_POINT_FACTOR


@dataclass(eq=False, frozen=True)
class MarketState:
    """Market snapshot for one currency pair.

    **Immutable by design.**  ``foreign_curve_adjusted`` is a
    :func:`functools.cached_property` derived from ``foreign_curve`` *and*
    ``basis_spreads``; if the instance were mutable, assigning to either
    field would leave that cache stale and every subsequent forward and
    valuation would silently use the old basis curve.  Use :meth:`replace`
    (which builds a fresh instance, and therefore a fresh cache) to make a
    shocked copy, that is what the risk and scenario modules do.

    Attributes
    ----------
    spot : float
        Spot rate, quote currency per 1 base currency (e.g. EURUSD 1.0850).
        Interpreted at t = 0: curve times run from the valuation date and the
        spot lag is not discounted separately.
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
    collateral_ccy : str, optional
        Currency of the collateral. Its curve discounts its own cashflows
        unchanged; the other currency's discount curve is bootstrapped from
        the basis quotes. Default: USD when in the pair, else the quote
        currency.
    basis_model : str
        ``"float_float"`` (default): ``basis_spreads`` are par spreads of
        float-float basis swaps. ``"zero_spread"``: the old shortcut
        ``z_adj = z_f + s`` on the base curve (deprecated).
    basis_frequency : int
        Resets per year of the quoted basis swaps (default 4).
    """

    spot: float
    domestic_curve: DiscountCurve
    foreign_curve: DiscountCurve
    basis_spreads: tuple[tuple[float, float], ...] = ()
    pair: tuple[str, str] = ("EUR", "USD")
    collateral_ccy: str | None = None
    basis_model: str = "float_float"
    basis_frequency: int = 4

    def __post_init__(self) -> None:
        require_finite(spot=self.spot)
        for _t, _s in self.basis_spreads:
            require_finite(basis_tenor=_t, basis_spread=_s)
        if self.spot <= 0.0:
            raise ValueError(f"spot must be > 0, got {self.spot}")
        base, quote = self.pair
        if base == quote:
            raise ValueError(
                f"same-currency 'cross' {base}/{quote} rejected: base and quote must differ"
            )
        object.__setattr__(
            self,
            "basis_spreads",
            tuple((float(t), float(s)) for t, s in self.basis_spreads),
        )
        if self.collateral_ccy is None:
            object.__setattr__(
                self, "collateral_ccy", "USD" if "USD" in self.pair else quote
            )
        if self.collateral_ccy not in self.pair:
            raise ValueError(
                f"collateral_ccy {self.collateral_ccy!r} must be one of {self.pair}"
            )
        if self.basis_model not in ("float_float", "zero_spread"):
            raise ValueError(f"unknown basis_model {self.basis_model!r}")
        if self.basis_model == "zero_spread":
            warnings.warn(
                "basis_model='zero_spread' is deprecated: it makes basis risk "
                "identical to base-curve rate risk. Use 'float_float'.",
                DeprecationWarning,
                stacklevel=3,
            )

    @property
    def basis_ccy(self) -> str:
        """The non-collateral currency, whose discount curve carries the basis."""
        return self.pair[1] if self.collateral_ccy == self.pair[0] else self.pair[0]

    def _discount(self, projection: DiscountCurve, ccy: str) -> DiscountCurve:
        if self.basis_model == "zero_spread":
            if ccy != self.base_ccy:
                return projection
            return basis_adjusted_curve(projection, self.basis_spreads)
        if ccy != self.basis_ccy:
            return projection
        return basis_discount_curve(projection, self.basis_spreads, self.basis_frequency)

    @cached_property
    def foreign_curve_adjusted(self) -> DiscountCurve:
        """Discount curve for base-currency cashflows."""
        return self._discount(self.foreign_curve, self.base_ccy)

    @cached_property
    def domestic_curve_adjusted(self) -> DiscountCurve:
        """Discount curve for quote-currency cashflows."""
        return self._discount(self.domestic_curve, self.quote_ccy)

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
    require_finite(spot=spot, expiry=expiry)
    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot}")
    t = np.asarray(expiry, dtype=float)
    out = spot * np.asarray(foreign_curve.df(t)) / np.asarray(domestic_curve.df(t))
    return float(out) if t.ndim == 0 else out


def market_forward(market: MarketState, expiry) -> float:
    """Basis-consistent market forward ``S * D_base(T) / D_quote(T)``."""
    return cip_forward(
        market.spot, market.domestic_curve_adjusted, market.foreign_curve_adjusted, expiry
    )


def forward_points(
    market: MarketState, expiry, point_factor: float | None = None
):
    """Forward points ``(F_mkt - S) * factor`` (factor 1e4 for EURUSD pips)."""
    if point_factor is None:
        point_factor = _point_factor(market.pair)
    require_finite(point_factor=point_factor)
    if point_factor <= 0.0:
        raise ValueError(f"point_factor must be > 0, got {point_factor}")
    f = market_forward(market, expiry)
    return (f - market.spot) * point_factor


def forward_points_table(
    market: MarketState,
    tenors: Sequence[float],
    point_factor: float | None = None,
) -> pd.DataFrame:
    """Forward points table: CIP vs market (basis-adjusted) forwards.

    Columns: tenor (y), CIP forward, market forward, CIP points, market
    points, basis effect in points, forward-implied zero-spread basis in bp (not the float-float par spread).
    """
    if point_factor is None:
        point_factor = _point_factor(market.pair)
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
        require_finite(notional_base=self.notional_base, strike=self.strike,
                       expiry=self.expiry)
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
        df_d = market.domestic_curve_adjusted.df(t)
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
        require_finite(notional_base=self.notional_base,
                       near_strike=self.near_strike,
                       far_strike=self.far_strike,
                       near_expiry=self.near_expiry,
                       far_expiry=self.far_expiry)
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
        """MTM in quote currency, explicit four-cashflow sum (independent of
        the two-forward decomposition, which is tested against this)."""
        if market.pair != self.pair:
            raise ValueError(
                f"market pair {market.pair} does not match position pair {self.pair}"
            )
        n = self.notional_base
        s = market.spot
        dfd, dff = market.domestic_curve_adjusted.df, market.foreign_curve_adjusted.df
        v = 0.0
        v += n * s * dff(self.near_expiry)                     # receive base near
        v -= n * self.near_strike * dfd(self.near_expiry)      # pay quote near
        v -= n * s * dff(self.far_expiry)                      # pay base far
        v += n * self.far_strike * dfd(self.far_expiry)        # receive quote far
        return v
