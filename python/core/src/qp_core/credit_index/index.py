"""CDX and iTraxx credit indices: series and roll, factor and defaults,
quoted spread/price to upfront, intrinsic value and the index basis.

Plain idea
----------
A credit index is an equally weighted (CDX.NA.IG: 125 names, iTraxx
Europe: 125 names) portfolio of standard single-name CDS that trades as one
contract with one fixed coupon. It is quoted either as a spread (IG,
iTraxx Main) or as a price (CDX HY: ``100 - upfront%``), and the quote is
converted to cash with the ISDA model on **one flat curve** at a standard
recovery. Pricing every constituent on its own curve and adding up gives
the *intrinsic* value; the gap between the two is the index basis.

Formal statement
----------------
* Rolls: a new series starts on 20 Mar and 20 Sep (the 2015 move to
  semi-annual rolls for single names kept the index cadence); the 5y
  contract of the series launched on 20 Mar ``Y`` matures 20 Jun ``Y+5``,
  the one launched 20 Sep ``Y`` matures 20 Dec ``Y+5`` (IHS Markit CDX and
  iTraxx index rules). Anchors used in the tests: CDX.NA.IG series 33
  launched 20 Sep 2019 (5y maturity 20 Dec 2024); iTraxx Europe series 40
  launched 20 Sep 2023 (5y maturity 20 Dec 2028).
* Defaults: a default of a name with weight ``w`` cuts the index factor by
  ``w`` and creates a new version; the protection buyer receives
  ``w (1 - R_auction)`` per unit original notional. The running coupon and
  all upfronts after that scale with the factor.
* Quote to cash: ``U = factor * clean_upfront(flat hazard from quoted
  spread, index coupon, index recovery)`` (Markit converter, O'Kane 2008
  ch. 10).
* Intrinsic: ``U_intr = sum_{i alive} w_i clean_upfront_i``, each name on
  its own bootstrapped curve. Intrinsic spread
  ``= sum w_i Prot_i / sum w_i A_i`` with ``A_i`` the par-spread annuity
  (RPV01 less the accrued rebate), the weighted analogue of each name's
  par spread (O'Kane 2008, eq. 10.4).
* Index adjustment ("skew"): one multiplier ``beta`` on every constituent
  hazard so that ``U_intr(beta) = U_index``; used before pricing tranches
  or options on the index (O'Kane 2008, s. 10.6).

Worked example
--------------
125 names, all at 100bp quoted spread, 40% recovery, coupon 100bp: the
index upfront and the intrinsic upfront agree to 1e-12 (every constituent
curve is the index's flat curve), so the basis is zero. Default one name at 30% auction recovery: the
factor is 0.992, the buyer receives 0.008 * 0.7 = 0.0056 per unit original
notional, and every later upfront is the old one times 0.992.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from scipy.optimize import brentq

from qp_core.model import model

from .isda import (
    CdsContract,
    FlatForwardCurve,
    HazardCurve,
    cds_legs,
    quoted_spread_from_upfront,
    upfront_from_quoted_spread,
)

__all__ = [
    "CDX_HY",
    "CDX_IG",
    "ITRAXX_MAIN",
    "CreditIndex",
    "IndexFamily",
    "adjust_to_index",
    "index_basis",
    "index_maturity",
    "intrinsic",
    "roll_date",
    "series_on",
]


@dataclass(frozen=True)
class IndexFamily:
    """Roll anchor of an index family: series ``series`` launched on ``launch``."""

    name: str
    series: int
    launch: date
    coupon: float
    recovery: float
    quote: str = "spread"  # "spread" or "price"


CDX_IG = IndexFamily("CDX.NA.IG", 33, date(2019, 9, 20), 0.01, 0.4)
CDX_HY = IndexFamily("CDX.NA.HY", 33, date(2019, 9, 20), 0.05, 0.3, "price")
ITRAXX_MAIN = IndexFamily("iTraxx Europe", 40, date(2023, 9, 20), 0.01, 0.4)


def _rolls_since(anchor: date, d: date) -> int:
    """Number of 20 Mar / 20 Sep roll dates in ``(anchor, d]`` (negative before)."""
    def idx(x: date) -> int:
        # index of the latest roll on or before x, counting 2 per year
        half = 0 if (x.month, x.day) < (3, 20) else (1 if (x.month, x.day) < (9, 20) else 2)
        return 2 * x.year + half
    return idx(d) - idx(anchor)


def series_on(family: IndexFamily, d: date) -> int:
    """On-the-run series number on date ``d`` (the roll date belongs to the new series)."""
    return family.series + _rolls_since(family.launch, d)


def roll_date(family: IndexFamily, series: int) -> date:
    """Launch date of ``series``."""
    k = series - family.series
    m = family.launch.year * 2 + (0 if family.launch.month == 3 else 1) + k
    return date(m // 2, 3 if m % 2 == 0 else 9, 20)


def index_maturity(family: IndexFamily, series: int, tenor_years: int = 5) -> date:
    """Scheduled maturity: 20 Jun (March series) or 20 Dec (September), ``tenor`` years on."""
    r = roll_date(family, series)
    return date(r.year + tenor_years, 6 if r.month == 3 else 12, 20)


@dataclass(frozen=True)
class CreditIndex:
    """One index series/tenor with its constituents and the defaults so far.

    ``weights`` default to equal; ``defaults`` maps a defaulted name to its
    auction recovery.
    """

    family: IndexFamily
    series: int
    maturity: date
    names: tuple[str, ...]
    weights: tuple[float, ...] = ()
    defaults: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("names must be non-empty and unique")
        w = self.weights or tuple(1.0 / len(self.names) for _ in self.names)
        if len(w) != len(self.names) or any(not x > 0.0 for x in w):
            raise ValueError("one positive weight per name")
        if abs(math.fsum(w) - 1.0) > 1e-12:
            raise ValueError("weights must sum to 1")
        object.__setattr__(self, "weights", tuple(w))
        for n, r in self.defaults.items():
            if n not in self.names:
                raise ValueError(f"defaulted name {n!r} is not a constituent")
            if not 0.0 <= r <= 1.0:
                raise ValueError("auction recovery must lie in [0, 1]")

    @property
    def version(self) -> int:
        return 1 + len(self.defaults)

    @property
    def factor(self) -> float:
        return 1.0 - math.fsum(w for n, w in zip(self.names, self.weights, strict=True)
                               if n in self.defaults)

    def default_settlement(self) -> float:
        """Total paid to the protection buyer for defaults, per unit original notional."""
        return math.fsum(w * (1.0 - self.defaults[n])
                         for n, w in zip(self.names, self.weights, strict=True)
                         if n in self.defaults)

    def with_default(self, name: str, auction_recovery: float) -> CreditIndex:
        if name in self.defaults:
            raise ValueError(f"{name!r} has already defaulted")
        return CreditIndex(self.family, self.series, self.maturity, self.names,
                           self.weights, {**self.defaults, name: auction_recovery})

    def alive(self) -> list[tuple[str, float]]:
        return [(n, w) for n, w in zip(self.names, self.weights, strict=True)
                if n not in self.defaults]

    # ----------------------------------------------------- quote conversion
    def upfront_from_spread(self, trade_date: date, quoted_spread: float,
                            disc: FlatForwardCurve) -> float:
        """Clean upfront per unit original notional for a spread quote."""
        u = upfront_from_quoted_spread(trade_date, self.maturity, quoted_spread,
                                       self.family.coupon, disc, self.family.recovery)
        return self.factor * u["clean_upfront"]

    def spread_from_upfront(self, trade_date: date, upfront: float,
                            disc: FlatForwardCurve) -> float:
        if self.factor <= 0.0:
            raise ValueError("every name has defaulted")
        return quoted_spread_from_upfront(trade_date, self.maturity, upfront / self.factor,
                                          self.family.coupon, disc, self.family.recovery)

    @staticmethod
    def price_from_upfront(upfront_per_unit_remaining: float) -> float:
        """CDX HY price quote: ``100 * (1 - upfront)`` per 100 remaining notional."""
        return 100.0 * (1.0 - upfront_per_unit_remaining)

    @staticmethod
    def upfront_from_price(price: float) -> float:
        return 1.0 - price / 100.0


def _alive_legs(index: CreditIndex, trade_date: date, curves: Mapping[str, HazardCurve],
                disc: FlatForwardCurve, recoveries: Mapping[str, float] | None,
                beta: float = 1.0) -> list[tuple[float, dict[str, float]]]:
    out = []
    for n, w in index.alive():
        if n not in curves:
            raise ValueError(f"no curve for constituent {n!r}")
        rec = (recoveries or {}).get(n, index.family.recovery)
        cds = CdsContract(trade_date, index.maturity, index.family.coupon, rec)
        out.append((w, cds_legs(cds, curves[n].scaled(beta), disc)))
    return out


@model(id="CORE-CR-INDEX")
def intrinsic(index: CreditIndex, trade_date: date, curves: Mapping[str, HazardCurve],
              disc: FlatForwardCurve, recoveries: Mapping[str, float] | None = None
              ) -> dict[str, float]:
    """Intrinsic clean upfront and intrinsic spread from the constituent curves."""
    legs = _alive_legs(index, trade_date, curves, disc, recoveries)
    prot = math.fsum(w * lg["protection"] for w, lg in legs)
    # par-spread annuity: RPV01 less the accrued rebated at cash settlement
    rpv = math.fsum(w * (lg["rpv01"] - lg["accrued"] / index.family.coupon * lg["df_cash"])
                    for w, lg in legs)
    up = math.fsum(w * lg["clean_upfront"] for w, lg in legs)
    return {"upfront": up, "spread": prot / rpv if rpv > 0 else math.nan}


@model(id="CORE-CR-INDEX")
def index_basis(index: CreditIndex, trade_date: date, quoted_spread: float,
                curves: Mapping[str, HazardCurve], disc: FlatForwardCurve,
                recoveries: Mapping[str, float] | None = None) -> float:
    """Index minus intrinsic clean upfront, per unit original notional."""
    return (index.upfront_from_spread(trade_date, quoted_spread, disc)
            - intrinsic(index, trade_date, curves, disc, recoveries)["upfront"])


@model(id="CORE-CR-INDEX")
def adjust_to_index(index: CreditIndex, trade_date: date, quoted_spread: float,
                    curves: Mapping[str, HazardCurve], disc: FlatForwardCurve,
                    recoveries: Mapping[str, float] | None = None) -> tuple[float, dict[str, HazardCurve]]:
    """Common hazard multiplier making intrinsic equal to the index upfront."""
    target = index.upfront_from_spread(trade_date, quoted_spread, disc)

    def gap(beta: float) -> float:
        legs = _alive_legs(index, trade_date, curves, disc, recoveries, beta)
        return math.fsum(w * lg["clean_upfront"] for w, lg in legs) - target

    lo, hi = 1e-6, 2.0
    if gap(lo) > 0.0:
        raise ValueError("index upfront below the zero-hazard intrinsic value")
    while gap(hi) < 0.0:
        hi *= 2.0
        if hi > 1e4:
            raise ValueError("no hazard multiplier matches the index")
    beta = float(brentq(gap, lo, hi, xtol=1e-14, rtol=1e-13))
    return beta, {n: curves[n].scaled(beta) for n, _ in index.alive()}
