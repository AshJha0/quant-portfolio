"""Multi-currency FX book: positions, base-currency P&L, triangulation.

Every currency is represented by its USD factor (USD price of 1 unit, see
:mod:`fx_var.common`).  A position in a *cross* pair (EURJPY) is decomposed
into its two USD legs - long EUR leg, short JPY leg - so cross risk is
triangulated by construction and the book's P&L in any base currency is
consistent with the USD-pair factor set.

Position types
--------------
``Cash``     : a currency balance.  Riskless when denominated in the book's
               base currency.
``Spot``     : a spot FX position, long ``notional`` of the pair's base ccy
               against the quote ccy, struck at ``entry_rate`` (defaults to
               the reference market's cross rate, i.e. zero initial value).
``Forward``  : an outright forward, represented as spot plus two deposit
               legs:  +N e^{-r_f T} of base ccy and -N K e^{-r_d T} of quote
               ccy.  Forward points therefore expose the position to both
               currencies' interest-rate factors (CIP-consistent
               revaluation; tested in tests/test_forwards.py).
``Option``   : a European FX option revalued with Garman-Kohlhagen, either
               by full revaluation or a delta-vega(-gamma) mapping.

P&L convention
--------------
``Book.pnl`` returns profit (+) / loss (-) in the book's **base currency**:
``PnL = V1_usd / S1_base - V0_usd / S0_base`` where the base currency's own
USD price is shocked consistently.  A pure base-ccy cash balance therefore
carries exactly zero risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union

import numpy as np
import pandas as pd

from .common import fx_factor, ir_factor, split_pair, validate_finite, vol_factor
from .gk import gk_delta, gk_gamma, gk_price, gk_vega

__all__ = ["Market", "Cash", "Spot", "Forward", "Option", "Book", "Position"]

ShockLike = Union[Mapping[str, "float | np.ndarray"], pd.Series, pd.DataFrame]


# --------------------------------------------------------------------------
# Market snapshot
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Market:
    """Point-in-time market snapshot.

    Parameters
    ----------
    spot_usd : mapping ccy -> float
        USD price of 1 unit of each currency (``spot_usd["EUR"] = 1.08``
        means EURUSD = 1.08).  ``"USD"`` is implied at 1.0 and may be
        omitted; if present it must equal 1.0.
    rates : mapping ccy -> float
        Continuously compounded zero rates, annualised, ACT/365 (flat curve
        per currency - adequate at VaR granularity, see METHODOLOGY.md A7).
    vols : mapping pair -> float
        Annualised lognormal implied vol per option pair, e.g.
        ``{"EURUSD": 0.075}``.
    """

    spot_usd: Mapping[str, float]
    rates: Mapping[str, float] = field(default_factory=dict)
    vols: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spots = dict(self.spot_usd)
        if "USD" in spots and abs(spots["USD"] - 1.0) > 1e-12:
            raise ValueError("spot_usd['USD'] must be 1.0 (USD per USD)")
        spots.setdefault("USD", 1.0)
        for c, s in spots.items():
            if not np.isfinite(s) or s <= 0:
                raise ValueError(f"spot_usd[{c!r}] must be a positive number, got {s}")
        rates = dict(self.rates)
        for c, r in rates.items():
            # NaN policy: refuse.  A NaN rate otherwise propagates silently
            # into a NaN P&L and a NaN VaR.
            if not np.isfinite(r):
                raise ValueError(
                    f"rates[{c!r}] must be a finite number, got {r} "
                    "(NaN policy: refuse, never impute)"
                )
        vols = {k.upper(): v for k, v in self.vols.items()}
        for p, v in vols.items():
            # A NaN vol is *especially* dangerous here: the Garman-Kohlhagen
            # degenerate-limit branch treats a non-finite sigma*sqrt(T) as the
            # zero-vol case and silently returns forward intrinsic instead of
            # a price.  Refuse it at the boundary.
            if not np.isfinite(v):
                raise ValueError(
                    f"vols[{p!r}] must be a finite number, got {v} "
                    "(NaN policy: refuse, never impute)"
                )
            if v < 0:
                raise ValueError(f"vols[{p!r}] must be non-negative, got {v}")
        object.__setattr__(self, "spot_usd", spots)
        object.__setattr__(self, "rates", rates)
        object.__setattr__(self, "vols", vols)

    def spot(self, ccy: str) -> float:
        """USD price of 1 unit of ``ccy``."""
        try:
            return self.spot_usd[ccy]
        except KeyError:
            raise KeyError(f"no USD spot for currency {ccy!r} in Market") from None

    def rate(self, ccy: str) -> float:
        """cc zero rate for ``ccy`` (ACT/365, annualised)."""
        try:
            return self.rates[ccy]
        except KeyError:
            raise KeyError(f"no interest rate for currency {ccy!r} in Market") from None

    def cross(self, pair: str) -> float:
        """Cross rate of ``pair`` (QUOTE per 1 BASE) by USD triangulation."""
        base, quote = split_pair(pair)
        return self.spot(base) / self.spot(quote)

    def forward(self, pair: str, expiry: float) -> float:
        """CIP forward: ``F = X * exp((r_d - r_f) * T)`` (QUOTE per BASE)."""
        base, quote = split_pair(pair)
        return self.cross(pair) * np.exp((self.rate(quote) - self.rate(base)) * expiry)

    def vol(self, pair: str) -> float:
        """Annualised implied vol for ``pair``."""
        try:
            return self.vols[pair.upper()]
        except KeyError:
            raise KeyError(f"no implied vol for pair {pair!r} in Market") from None


# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Cash:
    """A cash balance of ``amount`` units of ``ccy``."""

    ccy: str
    amount: float

    def __post_init__(self) -> None:
        validate_finite(amount=self.amount)

    def currencies(self) -> set[str]:
        return {self.ccy.upper()}

    def rate_currencies(self) -> set[str]:
        return set()

    def vol_pairs(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class Spot:
    """Spot FX position: long ``notional`` of BASE ccy vs QUOTE at ``entry_rate``.

    ``notional`` is in units of the pair's base currency; negative = short.
    ``entry_rate=None`` means struck at the reference market's cross rate,
    i.e. zero initial value.  Internally the position is two cash legs
    (+N BASE, -N*X0 QUOTE) - the USD triangulation of the cross.
    """

    pair: str
    notional: float
    entry_rate: float | None = None

    def __post_init__(self) -> None:
        split_pair(self.pair)  # validate
        validate_finite(notional=self.notional, entry_rate=self.entry_rate)
        if self.entry_rate is not None and self.entry_rate <= 0:
            raise ValueError("entry_rate must be > 0")

    def currencies(self) -> set[str]:
        return set(split_pair(self.pair))

    def rate_currencies(self) -> set[str]:
        return set()

    def vol_pairs(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class Forward:
    """Outright FX forward: long ``notional`` BASE at strike ``strike``, expiry ``expiry`` (years).

    Represented as spot + two deposit legs (CIP): value in USD is
    ``N e^{-r_f T} S_base_usd - N K e^{-r_d T} S_quote_usd``, so the position
    responds to both FX factors and both IR factors (forward-point risk).
    ``strike=None`` resolves to the ATM CIP forward of the reference market.
    """

    pair: str
    notional: float
    expiry: float
    strike: float | None = None

    def __post_init__(self) -> None:
        split_pair(self.pair)
        validate_finite(notional=self.notional, expiry=self.expiry,
                        strike=self.strike)
        if self.expiry < 0:
            raise ValueError("expiry must be >= 0")
        if self.strike is not None and self.strike <= 0:
            raise ValueError("strike must be > 0")

    def currencies(self) -> set[str]:
        return set(split_pair(self.pair))

    def rate_currencies(self) -> set[str]:
        return set(split_pair(self.pair))

    def vol_pairs(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class Option:
    """European FX option (Garman-Kohlhagen), ``notional`` in BASE ccy units.

    ``kind='call'`` is a call on the BASE currency (right to buy BASE at K
    units of QUOTE).  Premium/value is carried in QUOTE ccy per unit BASE
    notional and converted to USD/base via the quote leg.  Implied vol is
    read from ``Market.vols[pair]`` and shocked via the ``VOL:PAIR`` factor.
    """

    pair: str
    notional: float
    strike: float
    expiry: float
    kind: str = "call"

    def __post_init__(self) -> None:
        split_pair(self.pair)
        validate_finite(notional=self.notional, strike=self.strike,
                        expiry=self.expiry)
        if self.strike <= 0:
            raise ValueError("strike must be > 0")
        if self.expiry < 0:
            raise ValueError("expiry must be >= 0")
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")

    def currencies(self) -> set[str]:
        return set(split_pair(self.pair))

    def rate_currencies(self) -> set[str]:
        return set(split_pair(self.pair))

    def vol_pairs(self) -> set[str]:
        return {self.pair.upper()}


Position = Union[Cash, Spot, Forward, Option]


# --------------------------------------------------------------------------
# Book
# --------------------------------------------------------------------------
class Book:
    """A multi-currency book with a designated base (reporting) currency.

    Parameters
    ----------
    positions : sequence of Position
        Cash / Spot / Forward / Option positions (may be empty).
    base : str
        Reporting currency for P&L (default ``"USD"``).
    """

    def __init__(self, positions: Sequence[Position] = (), base: str = "USD") -> None:
        self.positions: list[Position] = list(positions)
        self.base = base.upper()

    # -- introspection ----------------------------------------------------
    def currencies(self) -> set[str]:
        """All currencies the book touches, including the base currency."""
        ccys = {self.base}
        for p in self.positions:
            ccys |= p.currencies()
        return ccys

    def factors(self, market: Market | None = None) -> list[str]:
        """Sorted risk-factor names the book is exposed to.

        FX factors for every non-USD currency involved (incl. base if not
        USD), IR factors for forward/option leg currencies, VOL factors for
        option pairs.
        """
        fx = {fx_factor(c) for c in self.currencies() if c != "USD"}
        ir: set[str] = set()
        vol: set[str] = set()
        for p in self.positions:
            ir |= {ir_factor(c) for c in p.rate_currencies()}
            vol |= {vol_factor(q) for q in p.vol_pairs()}
        return sorted(fx) + sorted(ir) + sorted(vol)

    # -- valuation --------------------------------------------------------
    @staticmethod
    def _normalise_shocks(shocks: ShockLike | None) -> dict[str, np.ndarray | float]:
        if shocks is None:
            return {}
        if isinstance(shocks, pd.DataFrame):
            out: dict[str, np.ndarray | float] = {
                c: shocks[c].to_numpy(dtype=float) for c in shocks.columns
            }
        elif isinstance(shocks, pd.Series):
            out = {str(k): float(v) for k, v in shocks.items()}
        else:
            out = {
                str(k): (np.asarray(v, dtype=float) if np.ndim(v) else float(v))
                for k, v in shocks.items()
            }
        if "FX:USD" in out:
            raise ValueError(
                "shock to 'FX:USD' is not a valid factor: USD is the pivot "
                "(its USD price is identically 1). Shock the other leg(s)."
            )
        # NaN policy: refuse.  A single NaN scenario entry otherwise
        # propagates to a NaN P&L, a NaN quantile and a NaN VaR without any
        # exception being raised anywhere on the path.
        for name, value in out.items():
            if not np.all(np.isfinite(np.asarray(value, dtype=float))):
                raise ValueError(
                    f"shock for factor {name!r} contains NaN or infinite "
                    "values (NaN policy: refuse, never impute)"
                )
        return out

    def _shocked_curves(
        self, market: Market, shocks: dict[str, np.ndarray | float]
    ):
        """Build shocked spot/rate/vol lookups (values broadcastable)."""
        ccys = self.currencies()
        pairs: set[str] = set()
        rate_ccys: set[str] = set()
        for p in self.positions:
            pairs |= p.vol_pairs()
            rate_ccys |= p.rate_currencies()
        spot = {}
        for c in ccys:
            if c == "USD":
                spot[c] = 1.0
                continue
            s0 = market.spot(c)
            spot[c] = s0 * np.exp(shocks.get(fx_factor(c), 0.0))
        rates = {c: market.rate(c) + shocks.get(ir_factor(c), 0.0) for c in rate_ccys}
        vols = {q: market.vol(q) + shocks.get(vol_factor(q), 0.0) for q in pairs}
        for q, v in vols.items():
            if np.any(np.asarray(v) < 0):
                raise ValueError(
                    f"shocked implied vol for {q} is negative; cap the vol "
                    "shock or use a floored scenario"
                )
        return spot, rates, vols

    def _position_value_usd(self, pos, spot, rates, vols, ref: Market, option_method: str):
        if isinstance(pos, Cash):
            return pos.amount * spot[pos.ccy.upper()]
        b, q = split_pair(pos.pair)
        if isinstance(pos, Spot):
            x0 = pos.entry_rate if pos.entry_rate is not None else ref.cross(pos.pair)
            return pos.notional * spot[b] - pos.notional * x0 * spot[q]
        if isinstance(pos, Forward):
            k = pos.strike if pos.strike is not None else ref.forward(pos.pair, pos.expiry)
            df_f = np.exp(-rates[b] * pos.expiry)
            df_d = np.exp(-rates[q] * pos.expiry)
            return pos.notional * df_f * spot[b] - pos.notional * k * df_d * spot[q]
        if isinstance(pos, Option):
            x = spot[b] / spot[q]
            sigma = vols[pos.pair.upper()]
            if option_method == "full":
                price_q = gk_price(x, pos.strike, pos.expiry, rates[q], rates[b], sigma, pos.kind)
                return pos.notional * price_q * spot[q]
            # delta-vega(-gamma) mapping around the reference market
            x0 = ref.cross(pos.pair)
            s0 = ref.vol(pos.pair)
            rd0, rf0 = ref.rate(q), ref.rate(b)
            p0 = gk_price(x0, pos.strike, pos.expiry, rd0, rf0, s0, pos.kind)
            delta = gk_delta(x0, pos.strike, pos.expiry, rd0, rf0, s0, pos.kind)
            vega = gk_vega(x0, pos.strike, pos.expiry, rd0, rf0, s0)
            dx = x - x0
            dsig = sigma - s0
            dp = delta * dx + vega * dsig
            if option_method == "delta_vega_gamma":
                gamma = gk_gamma(x0, pos.strike, pos.expiry, rd0, rf0, s0)
                dp = dp + 0.5 * gamma * dx**2
            elif option_method != "delta_vega":
                raise ValueError(
                    "option_method must be 'full', 'delta_vega' or "
                    f"'delta_vega_gamma', got {option_method!r}"
                )
            return pos.notional * (p0 + dp) * spot[q]
        raise TypeError(f"unknown position type {type(pos).__name__}")

    def value_usd(self, market: Market, shocks: ShockLike | None = None,
                  option_method: str = "full"):
        """Book value in USD at ``market`` after applying ``shocks``.

        Broadcasts over scenario arrays in ``shocks``; scalar in/scalar out.
        """
        if option_method not in ("full", "delta_vega", "delta_vega_gamma"):
            # Validated up front so a typo is caught even on option-free
            # books (where the per-position branch would never see it).
            raise ValueError(
                "option_method must be 'full', 'delta_vega' or "
                f"'delta_vega_gamma', got {option_method!r}"
            )
        sh = self._normalise_shocks(shocks)
        spot, rates, vols = self._shocked_curves(market, sh)
        total = 0.0
        for p in self.positions:
            total = total + self._position_value_usd(p, spot, rates, vols, market, option_method)
        # empty book / all-scalar path returns a python float
        return total

    def pnl(self, market: Market, shocks: ShockLike, option_method: str = "full"):
        """Base-currency P&L of the book under factor ``shocks``.

        Parameters
        ----------
        market : Market
            Reference (pre-shock) market snapshot.
        shocks : mapping / Series / DataFrame
            Factor shocks: ``FX:*`` are log returns of CCYUSD, ``IR:*`` and
            ``VOL:*`` absolute changes.  Array-valued entries broadcast to
            scenario vectors; a DataFrame gives one scenario per row.
        option_method : {"full", "delta_vega", "delta_vega_gamma"}
            Option revaluation: full Garman-Kohlhagen repricing or Greek
            mapping around the reference market.

        Returns
        -------
        float or numpy.ndarray
            Profit (+) / loss (-) in the base currency, one value per
            scenario.
        """
        sh = self._normalise_shocks(shocks)
        v0 = self.value_usd(market, None, option_method="full")
        v1 = self.value_usd(market, sh, option_method=option_method)
        if self.base == "USD":
            s0b, s1b = 1.0, 1.0
        else:
            s0b = market.spot(self.base)
            s1b = s0b * np.exp(sh.get(fx_factor(self.base), 0.0))
        out = v1 / s1b - v0 / s0b
        if np.ndim(out) == 0:
            return float(out)
        return np.asarray(out, dtype=float)

    def linear_exposures(self, market: Market, factors: Sequence[str] | None = None,
                         bump: float = 1e-6) -> pd.Series:
        """Delta exposures dPnL/dfactor by central finite differences.

        Units: base-ccy P&L per unit factor move - for ``FX:*`` per unit log
        return (i.e. base-ccy notional exposure), for ``IR:*`` per 1.00 of
        rate (divide by 1e4 for a DV01), for ``VOL:*`` per 1.00 of vol.

        These are the mapping weights used by the variance-covariance method
        and the reverse-stress closed form.
        """
        if factors is None:
            factors = self.factors(market)
        vals = {}
        for f in factors:
            up = self.pnl(market, {f: bump})
            dn = self.pnl(market, {f: -bump})
            vals[f] = (up - dn) / (2.0 * bump)
        return pd.Series(vals, dtype=float).reindex(list(factors))
