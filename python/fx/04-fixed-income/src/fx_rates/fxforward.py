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

Spot is a price for delivery on the spot date ``t_s`` (T+2, or T+1 for
USDCAD, USDTRY, USDRUB, USDPHP; see :func:`qp_core.holidays.fx_spot_date`),
not today. The forward to time ``T`` (years from the valuation date) is

    F(T) = S * DF_f(t_s, T) / DF_d(t_s, T) = S * [P_f(T)/P_f(t_s)] / [P_d(T)/P_d(t_s)]

(Clark 2011, ch. 1; Wystup 2017, ch. 1). The value today of one unit of
base currency delivered at spot is ``S * P_d(t_s) / P_f(t_s)`` in quote
currency, :attr:`MarketState.spot_today`. Until round 6 the spot rate was
treated as a t = 0 price, which overstated a 1y EURUSD forward by about
2.4 pips at a 2% rate differential. That behaviour is kept, with a
``DeprecationWarning``, when a ``MarketState`` gets neither a
``valuation_date`` nor a ``spot_time`` (or gets ``spot_at_t0=True``); it
will be removed in v4.0.0.

All position values are expressed in the **quote (domestic) currency**.
A positive ``notional_base`` means long the base currency forward
(receive base, pay quote at expiry).
"""

from __future__ import annotations

import datetime as dt
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from functools import cached_property

import numpy as np
import pandas as pd
from qp_core.holidays import fx_spot_date

from ._validation import require_finite
from .bootstrap import basis_adjusted_curve, basis_discount_curve
from .curve import DiscountCurve

__all__ = [
    "FXForward",
    "FXSwap",
    "MarketState",
    "cip_forward",
    "curve_from_forward_points",
    "forward_for_date",
    "forward_points",
    "forward_points_for_date",
    "forward_points_table",
    "market_forward",
    "year_fraction_365",
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
        Spot rate, quote currency per 1 base currency (e.g. EURUSD 1.0850),
        for delivery on the spot date. Curve times run from the valuation
        date (ACT/365F).
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
    valuation_date : datetime.date, optional
        Valuation (trade) date. Sets ``spot_date`` from the pair's
        calendars and ``spot_time = (spot_date - valuation_date) / 365``.
        Needed for date-based (broken-date) forwards and NDFs.
    spot_time : float, optional
        Year fraction from valuation to the spot date, when no dates are
        at hand. Give this or ``valuation_date``, not both.
    spot_at_t0 : bool
        Deprecated: treat spot as a t = 0 price (the pre-round-6
        behaviour). It is also the fallback, with a warning, when neither
        ``valuation_date`` nor ``spot_time`` is given.
    """

    spot: float
    domestic_curve: DiscountCurve
    foreign_curve: DiscountCurve
    basis_spreads: tuple[tuple[float, float], ...] = ()
    pair: tuple[str, str] = ("EUR", "USD")
    collateral_ccy: str | None = None
    basis_model: str = "float_float"
    basis_frequency: int = 4
    valuation_date: dt.date | None = None
    spot_time: float | None = None
    spot_at_t0: bool = False

    def __post_init__(self) -> None:
        self._resolve_spot_time()
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

    def _resolve_spot_time(self) -> None:
        if self.spot_at_t0:
            if self.valuation_date is not None or self.spot_time not in (None, 0.0):
                raise ValueError("spot_at_t0=True excludes valuation_date and spot_time")
            warnings.warn(
                "spot_at_t0=True is deprecated: a quoted spot rate is for the spot "
                "date, not today; it will be removed in v4.0.0. Pass valuation_date "
                "or spot_time.",
                DeprecationWarning, stacklevel=4)
            object.__setattr__(self, "spot_time", 0.0)
            return
        if self.valuation_date is not None:
            if isinstance(self.valuation_date, dt.datetime) or not isinstance(
                    self.valuation_date, dt.date):
                raise TypeError("valuation_date must be a datetime.date")
            ts = year_fraction_365(self.valuation_date,
                                   fx_spot_date(self.valuation_date, self.pair))
            if self.spot_time is not None and abs(float(self.spot_time) - ts) > 1e-15:
                raise ValueError(
                    f"spot_time {self.spot_time} contradicts valuation_date "
                    f"{self.valuation_date} (spot time {ts}); give one of them")
            object.__setattr__(self, "spot_time", ts)
            return
        if self.spot_time is None:
            warnings.warn(
                "MarketState without valuation_date or spot_time treats spot as a "
                "t = 0 price; this is deprecated and will be removed in v4.0.0. Pass "
                "valuation_date (or spot_time) so forwards run from the spot date.",
                DeprecationWarning, stacklevel=4)
            object.__setattr__(self, "spot_time", 0.0)
            return
        require_finite(spot_time=self.spot_time)
        if self.spot_time < 0.0:
            raise ValueError(f"spot_time must be >= 0, got {self.spot_time}")
        object.__setattr__(self, "spot_time", float(self.spot_time))

    @property
    def spot_date(self) -> dt.date | None:
        """Spot value date, when ``valuation_date`` is set."""
        if self.valuation_date is None:
            return None
        return fx_spot_date(self.valuation_date, self.pair)

    @cached_property
    def spot_today(self) -> float:
        """Today's value, in quote currency, of one base unit delivered at spot:
        ``S * P_d(t_s) / P_f(t_s)`` on the discount curves."""
        return _spot_t0(self.spot, self.spot_time, self.domestic_curve_adjusted,
                        self.foreign_curve_adjusted)

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

    def replace(self, **kwargs) -> MarketState:
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
def year_fraction_365(start: dt.date, end: dt.date) -> float:
    """ACT/365F year fraction, the time basis of every curve in this module."""
    return (end - start).days / 365.0


def _spot_t0(spot: float, ts: float | None, dom: DiscountCurve,
             for_: DiscountCurve) -> float:
    if not ts:
        return float(spot)
    return float(spot * dom.df(ts) / for_.df(ts))

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
        market.spot_today, market.domestic_curve_adjusted, market.foreign_curve_adjusted,
        expiry,
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
    s0_cip = _spot_t0(market.spot, market.spot_time, market.domestic_curve,
                      market.foreign_curve)
    for t in tenors:
        f_cip = cip_forward(s0_cip, market.domestic_curve, market.foreign_curve, t)
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
            return n * market.spot_today * df_f - n * k * df_d
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
        s = market.spot_today
        dfd, dff = market.domestic_curve_adjusted.df, market.foreign_curve_adjusted.df
        v = 0.0
        v += n * s * dff(self.near_expiry)                     # receive base near
        v -= n * self.near_strike * dfd(self.near_expiry)      # pay quote near
        v -= n * s * dff(self.far_expiry)                      # pay base far
        v += n * self.far_strike * dfd(self.far_expiry)        # receive quote far
        return v


# ---------------------------------------------------------------------- #
# dated (broken-date) forwards
# ---------------------------------------------------------------------- #
def _time_to(market: MarketState, value_date: dt.date) -> float:
    if market.valuation_date is None:
        raise ValueError("date-based forwards need a MarketState with valuation_date")
    if value_date < market.valuation_date:
        raise ValueError(
            f"value date {value_date} precedes valuation date {market.valuation_date}")
    return year_fraction_365(market.valuation_date, value_date)


def forward_for_date(market: MarketState, value_date: dt.date) -> float:
    """Outright forward for any value date, a quoted tenor or a broken date.

    ``T = (value_date - valuation_date) / 365`` and the forward is
    ``S * DF_f(t_s, T) / DF_d(t_s, T)`` on the curves. A broken date gets
    the rate the curves imply between the neighbouring pillars (log-linear
    discount factors), not a straight-line blend of quoted points. Value
    dates before spot (ON, TN) are allowed: the same formula discounts back
    from spot.
    """
    return market_forward(market, _time_to(market, value_date))


def forward_points_for_date(market: MarketState, value_date: dt.date,
                            point_factor: float | None = None) -> float:
    """Forward points ``(F - S) * factor`` for a value date."""
    return forward_points(market, _time_to(market, value_date), point_factor)


def curve_from_forward_points(
    spot: float,
    domestic_curve: DiscountCurve,
    strip: Sequence[tuple[dt.date, float]],
    valuation_date: dt.date,
    pair: tuple[str, str] = ("EUR", "USD"),
    point_factor: float | None = None,
    name: str = "implied-foreign",
) -> DiscountCurve:
    """Base-currency discount curve that reprices a forward-points strip.

    ``strip`` holds ``(value_date, points)`` pairs. Each gives
    ``F_i = S + points_i / factor``, and inverting the spot-date forward
    gives

        P_f(T_i) = F_i * P_d(T_i) * P_f(t_s) / (S * P_d(t_s)).

    ``P_f(t_s)`` lies before the first pillar, where the curve is
    log-linear from ``P_f(0) = 1``, so ``ln P_f(t_s) = (t_s / T_1) ln P_f(T_1)``
    and the first pillar is solved in closed form. A ``MarketState`` with
    this curve as ``foreign_curve`` (no basis quotes) and the same
    ``valuation_date`` reprices every input point to rounding.
    """
    require_finite(spot=spot)
    if spot <= 0.0:
        raise ValueError(f"spot must be > 0, got {spot}")
    pf = _point_factor((pair[0], pair[1])) if point_factor is None else point_factor
    ts = year_fraction_365(valuation_date, fx_spot_date(valuation_date, pair))
    rows = sorted((year_fraction_365(valuation_date, d), spot + pts / pf) for d, pts in strip)
    if not rows:
        raise ValueError("strip must not be empty")
    if rows[0][0] <= ts:
        raise ValueError("every strip date must be after the spot date")
    if any(f <= 0.0 for _, f in rows):
        raise ValueError("forward points imply a non-positive forward")
    pd_ts = domestic_curve.df(ts)
    t1, f1 = rows[0]
    ln_p1 = (math.log(f1) - math.log(spot) - math.log(pd_ts)
             + math.log(domestic_curve.df(t1))) / (1.0 - ts / t1)
    pf_ts = math.exp(ts / t1 * ln_p1)
    times = [t for t, _ in rows]
    dfs = [f * domestic_curve.df(t) * pf_ts / (spot * pd_ts) for t, f in rows]
    return DiscountCurve(times, dfs, name=name)
