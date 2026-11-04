"""SA-CCR: the Basel standardised approach for counterparty credit risk.

Plain idea
----------
A bank that trades derivatives with a counterparty must hold capital against
the chance that the counterparty defaults while the trades are worth money to
the bank. SA-CCR measures that "exposure at default" (EAD) with a fixed recipe
instead of a model: what the trades are worth today if positive (replacement
cost, RC) plus a supervisory allowance for how far they could move before the
bank closes them out (potential future exposure, PFE). The total is scaled by
1.4.

Formal statement
----------------
BCBS 279 (March 2014), Annex 4, carried into the consolidated framework as
CRE52:

    EAD        = alpha (RC + multiplier * AddOn_aggregate),   alpha = 1.4
    RC         = max(V - C, 0)                                  (unmargined)
    RC         = max(V - C, TH + MTA - NICA, 0)                 (margined)
    multiplier = min(1, f + (1 - f) exp((V - C) / (2 (1 - f) AddOn))),  f = 5 %
    AddOn      = sum over asset classes of the asset-class add-on.

Each trade i contributes delta_i * d_i * MF_i, where d_i is the adjusted
notional (notional times the supervisory duration
SD = (exp(-0.05 S) - exp(-0.05 E)) / 0.05 for interest-rate and credit trades),
delta_i the supervisory delta (+1/-1, or a Black delta at the supervisory
volatility for options) and MF_i the maturity factor. Aggregation per asset
class:

* interest rates: per currency, three maturity buckets (<1y, 1-5y, >5y)
  combined with correlations 1.4/2 = 0.7 (adjacent) and 0.6/2 = 0.3
  (bucket 1 with 3), i.e.
  EN = sqrt(D1^2 + D2^2 + D3^2 + 1.4 D1 D2 + 1.4 D2 D3 + 0.6 D1 D3),
  AddOn = sum over currencies of SF * EN (CRE52.57);
* FX: per currency pair, SF * |sum delta d MF| (CRE52.58);
* credit, equity, commodity: a single-factor model,
  AddOn = sqrt((sum rho_k A_k)^2 + sum (1 - rho_k^2) A_k^2) where A_k is the
  entity (or commodity-type) add-on (CRE52.61, 52.66; commodity per hedging
  set with rho = 0.4, then summed over hedging sets, CRE52.69).

Worked example
--------------
BCBS 279 Annex 4a, Example 1 (three interest-rate trades, unmargined; the
published result is EAD = 569 thousand, with intermediate values rounded):

>>> trades = [
...     IRTrade("USD", 10_000, 0.0, 10.0, delta=1.0),
...     IRTrade("USD", 10_000, 0.0, 4.0, delta=-1.0),
...     IRTrade("EUR", 5_000, 1.0, 11.0,
...             delta=supervisory_delta_option(0.06, 0.05, 0.5, 1.0, call=False, bought=True)),
... ]
>>> r = ead(NettingSet(mtm=30 - 20 + 50, ir=trades))
>>> round(r.addon), round(r.rc), round(r.ead)
(347, 60, 569)

Sources: BCBS 279 (2014) paras 128-187 and Annex 4a examples (the oracle for
the tests); BCBS CRE52 (2020) for the supervisory table (MAR/CRE text is
public on bis.org). Evidence class: regulatory text, reproduced to the
published rounding.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from qp_core.model import model

from ..errors import InputError
from ..normal import norm_cdf

__all__ = [
    "ALPHA",
    "SUPERVISORY",
    "CommodityTrade",
    "FXTrade",
    "IRTrade",
    "NettingSet",
    "SaccrResult",
    "SingleFactorTrade",
    "commodity_addon",
    "ead",
    "fx_addon",
    "ir_addon",
    "ir_bucket",
    "ir_effective_notional",
    "maturity_factor",
    "mpor_days",
    "multiplier",
    "replacement_cost",
    "single_factor_addon",
    "supervisory_delta_option",
    "supervisory_duration",
]

ALPHA = 1.4
MULTIPLIER_FLOOR = 0.05
BUSINESS_DAYS_PER_YEAR = 250.0

#: CRE52.72 supervisory parameters: (supervisory factor, correlation, option volatility).
#: Correlation is ``None`` where the asset class does not use one.
SUPERVISORY: dict[str, tuple[float, float | None, float]] = {
    "IR": (0.005, None, 0.50),
    "FX": (0.04, None, 0.15),
    "CR_AAA": (0.0038, 0.50, 1.00),
    "CR_AA": (0.0038, 0.50, 1.00),
    "CR_A": (0.0042, 0.50, 1.00),
    "CR_BBB": (0.0054, 0.50, 1.00),
    "CR_BB": (0.0106, 0.50, 1.00),
    "CR_B": (0.016, 0.50, 1.00),
    "CR_CCC": (0.06, 0.50, 1.00),
    "CR_IDX_IG": (0.0038, 0.80, 0.80),
    "CR_IDX_SG": (0.0106, 0.80, 0.80),
    "EQ_SINGLE": (0.32, 0.50, 1.20),
    "EQ_INDEX": (0.20, 0.80, 0.75),
    "CO_ELECTRICITY": (0.40, 0.40, 1.50),
    "CO_OIL_GAS": (0.18, 0.40, 0.70),
    "CO_METALS": (0.18, 0.40, 0.70),
    "CO_AGRICULTURAL": (0.18, 0.40, 0.70),
    "CO_OTHER": (0.18, 0.40, 0.70),
}


def _finite(x: float, what: str) -> float:
    x = float(x)
    if not math.isfinite(x):
        raise InputError(f"{what} must be finite")
    return x


def supervisory_duration(start: float, end: float) -> float:
    """SD = (exp(-0.05 S) - exp(-0.05 E)) / 0.05 (BCBS 279 para 157).

    ``start`` and ``end`` are in years and floored at ten business days, as
    CRE52.34 requires; ``end`` must not precede ``start``.

    >>> round(supervisory_duration(0.0, 10.0), 4)
    7.8694
    """
    s = max(_finite(start, "start"), 10.0 / BUSINESS_DAYS_PER_YEAR) if start > 0.0 else 0.0
    e = max(_finite(end, "end"), 10.0 / BUSINESS_DAYS_PER_YEAR)
    if e < s:
        raise InputError("end must not precede start")
    return (math.exp(-0.05 * s) - math.exp(-0.05 * e)) / 0.05


def supervisory_delta_option(
    price: float,
    strike: float,
    vol: float,
    expiry: float,
    *,
    call: bool,
    bought: bool,
    shift: float = 0.0,
) -> float:
    """Supervisory delta of an option (BCBS 279 para 159, CRE52.40).

    ``d1 = (ln((P + lambda) / (K + lambda)) + vol^2 T / 2) / (vol sqrt T)``;
    a bought call has delta N(d1), a bought put -N(-d1); selling flips the
    sign. ``shift`` is the CRE52.40 lambda for negative rates.

    >>> round(supervisory_delta_option(0.06, 0.05, 0.5, 1.0, call=False, bought=True), 4)
    -0.2694
    """
    p = _finite(price, "price") + shift
    k = _finite(strike, "strike") + shift
    if p <= 0.0 or k <= 0.0:
        raise InputError("shifted price and strike must be positive")
    if not vol > 0.0 or not expiry > 0.0:
        raise InputError("vol and expiry must be positive")
    srt = vol * math.sqrt(expiry)
    d1 = (math.log(p / k) + 0.5 * vol * vol * expiry) / srt
    d = norm_cdf(d1) if call else -norm_cdf(-d1)
    return d if bought else -d


def mpor_days(margin_frequency_days: int = 1, floor_days: int = 10) -> int:
    """MPOR = floor + N - 1 business days for re-margining every N days (CRE52.50).

    >>> mpor_days(5)
    14
    """
    if margin_frequency_days < 1:
        raise InputError("margin frequency must be at least one day")
    return floor_days + margin_frequency_days - 1


def maturity_factor(maturity: float, *, mpor: float | None = None) -> float:
    """Maturity factor (CRE52.48-52.52).

    Unmargined (``mpor`` None): sqrt(min(M, 1) / 1) with M in years, floored at
    ten business days. Margined: 1.5 sqrt(MPOR / 250) with MPOR in business days.

    >>> round(maturity_factor(0.75), 6)
    0.866025
    >>> round(maturity_factor(5.0, mpor=14), 6)
    0.354965
    """
    if mpor is not None:
        if not mpor > 0.0:
            raise InputError("mpor must be positive")
        return 1.5 * math.sqrt(mpor / BUSINESS_DAYS_PER_YEAR)
    m = max(_finite(maturity, "maturity"), 10.0 / BUSINESS_DAYS_PER_YEAR)
    return math.sqrt(min(m, 1.0))


def replacement_cost(
    v: float, c: float = 0.0, *, margined: bool = False, th: float = 0.0, mta: float = 0.0, nica: float = 0.0
) -> float:
    """RC = max(V - C, 0), or max(V - C, TH + MTA - NICA, 0) when margined (CRE52.10, 52.18)."""
    rc = max(v - c, 0.0)
    if margined:
        rc = max(rc, th + mta - nica)
    return rc


def multiplier(v: float, c: float, addon: float) -> float:
    """PFE multiplier min(1, f + (1-f) exp((V-C) / (2 (1-f) AddOn))), f = 0.05 (CRE52.23).

    >>> round(multiplier(-20.0, 0.0, 282.2), 3)
    0.965
    """
    if addon < 0.0:
        raise InputError("addon must be non-negative")
    if addon == 0.0:
        return 1.0
    f = MULTIPLIER_FLOOR
    return min(1.0, f + (1.0 - f) * math.exp((v - c) / (2.0 * (1.0 - f) * addon)))


# --------------------------------------------------------------------- trades


@dataclass(frozen=True)
class IRTrade:
    """Interest-rate trade: hedging set ``currency``; S, E in years; M defaults to E."""

    currency: str
    notional: float
    start: float
    end: float
    delta: float = 1.0
    maturity: float | None = None

    @property
    def adjusted_notional(self) -> float:
        return self.notional * supervisory_duration(self.start, self.end)


@dataclass(frozen=True)
class FXTrade:
    """FX trade: hedging set ``pair``; ``notional`` in the domestic currency."""

    pair: str
    notional: float
    maturity: float
    delta: float = 1.0


@dataclass(frozen=True)
class SingleFactorTrade:
    """Credit or equity trade. ``category`` is a key of :data:`SUPERVISORY`
    (``CR_*`` or ``EQ_*``); ``entity`` names the reference entity or index.
    Credit notionals are multiplied by the supervisory duration; equity
    notionals (units times price) are used as given."""

    category: str
    entity: str
    notional: float
    start: float
    end: float
    delta: float = 1.0
    maturity: float | None = None

    @property
    def adjusted_notional(self) -> float:
        if self.category.startswith("CR_"):
            return self.notional * supervisory_duration(self.start, self.end)
        return self.notional


@dataclass(frozen=True)
class CommodityTrade:
    """Commodity trade: ``category`` is one of the ``CO_*`` keys; the hedging set
    is energy (electricity, oil/gas), metals, agricultural or other;
    ``commodity_type`` is e.g. "crude oil"."""

    category: str
    commodity_type: str
    notional: float
    maturity: float
    delta: float = 1.0


def _mf(maturity: float, mpor: float | None) -> float:
    return maturity_factor(maturity, mpor=mpor)


def ir_bucket(end: float) -> int:
    """Maturity bucket by end date: 0 for E < 1y, 1 for 1-5y, 2 for > 5y (CRE52.55)."""
    return 0 if end < 1.0 else (1 if end <= 5.0 else 2)


def ir_effective_notional(d1: float, d2: float, d3: float) -> float:
    """EN = sqrt(D1^2 + D2^2 + D3^2 + 1.4 D1 D2 + 1.4 D2 D3 + 0.6 D1 D3) (CRE52.57).

    >>> round(ir_effective_notional(0.0, -36254.0, 78694.0))
    59270
    """
    q = d1 * d1 + d2 * d2 + d3 * d3 + 1.4 * d1 * d2 + 1.4 * d2 * d3 + 0.6 * d1 * d3
    return math.sqrt(max(q, 0.0))


def ir_addon(trades: Iterable[IRTrade], *, mpor: float | None = None) -> float:
    """Interest-rate add-on: per-currency bucket effective notionals times 0.5 %."""
    buckets: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for t in trades:
        m = t.end if t.maturity is None else t.maturity
        buckets[t.currency][ir_bucket(t.end)] += t.delta * t.adjusted_notional * _mf(m, mpor)
    sf = SUPERVISORY["IR"][0]
    return sum(sf * ir_effective_notional(*b) for b in buckets.values())


def fx_addon(trades: Iterable[FXTrade], *, mpor: float | None = None) -> float:
    """FX add-on: per currency pair, 4 % times |sum delta d MF| (CRE52.58)."""
    sets: dict[str, float] = defaultdict(float)
    for t in trades:
        sets[t.pair] += t.delta * t.notional * _mf(t.maturity, mpor)
    return sum(SUPERVISORY["FX"][0] * abs(v) for v in sets.values())


def single_factor_addon(addons: Sequence[float], rhos: Sequence[float]) -> float:
    """sqrt((sum rho_k A_k)^2 + sum (1 - rho_k^2) A_k^2) (CRE52.61).

    >>> round(single_factor_addon([105.86, -279.91, 168.11], [0.5, 0.5, 0.8]))
    282
    """
    if len(addons) != len(rhos):
        raise InputError("addons and rhos must have equal length")
    sys_ = sum(r * a for a, r in zip(addons, rhos, strict=True))
    idio = sum((1.0 - r * r) * a * a for a, r in zip(addons, rhos, strict=True))
    return math.sqrt(sys_ * sys_ + idio)


def _sf_rho(category: str) -> tuple[float, float]:
    try:
        sf, rho, _ = SUPERVISORY[category]
    except KeyError:
        raise InputError(f"unknown supervisory category {category!r}") from None
    if rho is None:
        raise InputError(f"{category} is not a single-factor category")
    return sf, rho


def _single_factor_class(trades: Sequence[SingleFactorTrade], mpor: float | None) -> float:
    ent: dict[str, float] = defaultdict(float)
    cat: dict[str, str] = {}
    for t in trades:
        _sf_rho(t.category)
        if cat.setdefault(t.entity, t.category) != t.category:
            raise InputError(f"entity {t.entity!r} has two categories")
        m = t.end if t.maturity is None else t.maturity
        ent[t.entity] += t.delta * t.adjusted_notional * _mf(m, mpor)
    names = sorted(ent)
    a = [_sf_rho(cat[n])[0] * ent[n] for n in names]
    r = [_sf_rho(cat[n])[1] for n in names]
    return single_factor_addon(a, r) if names else 0.0


_CO_HEDGING_SET = {
    "CO_ELECTRICITY": "energy",
    "CO_OIL_GAS": "energy",
    "CO_METALS": "metals",
    "CO_AGRICULTURAL": "agricultural",
    "CO_OTHER": "other",
}


def commodity_addon(trades: Iterable[CommodityTrade], *, mpor: float | None = None) -> float:
    """Commodity add-on: per hedging set a single-factor sum over commodity types
    with rho = 0.4, then a plain sum over hedging sets (CRE52.69)."""
    types: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cat: dict[tuple[str, str], str] = {}
    for t in trades:
        if t.category not in _CO_HEDGING_SET:
            raise InputError(f"unknown commodity category {t.category!r}")
        hs = _CO_HEDGING_SET[t.category]
        cat[(hs, t.commodity_type)] = t.category
        types[hs][t.commodity_type] += t.delta * t.notional * _mf(t.maturity, mpor)
    total = 0.0
    for hs, by_type in types.items():
        names = sorted(by_type)
        a = [_sf_rho(cat[(hs, n)])[0] * by_type[n] for n in names]
        total += single_factor_addon(a, [0.4] * len(a))
    return total


# ---------------------------------------------------------------- netting set


@dataclass(frozen=True)
class NettingSet:
    """One netting set. ``mtm`` is V; ``collateral`` is C (net, haircut).

    ``mpor`` (business days) switches on the margined formulas; ``th``,
    ``mta`` and ``nica`` are the CSA threshold, minimum transfer amount and
    net independent collateral amount.
    """

    mtm: float
    collateral: float = 0.0
    ir: Sequence[IRTrade] = field(default_factory=tuple)
    fx: Sequence[FXTrade] = field(default_factory=tuple)
    credit: Sequence[SingleFactorTrade] = field(default_factory=tuple)
    equity: Sequence[SingleFactorTrade] = field(default_factory=tuple)
    commodity: Sequence[CommodityTrade] = field(default_factory=tuple)
    mpor: float | None = None
    th: float = 0.0
    mta: float = 0.0
    nica: float = 0.0


@dataclass(frozen=True)
class SaccrResult:
    rc: float
    addon: float
    multiplier: float
    pfe: float
    ead: float
    addon_by_class: dict[str, float]


@model(id="CORE-CAP-SACCR")
def ead(ns: NettingSet) -> SaccrResult:
    """EAD = 1.4 (RC + multiplier AddOn) for one netting set (CRE52.1)."""
    v, c = _finite(ns.mtm, "mtm"), _finite(ns.collateral, "collateral")
    by = {
        "IR": ir_addon(ns.ir, mpor=ns.mpor),
        "FX": fx_addon(ns.fx, mpor=ns.mpor),
        "CR": _single_factor_class(list(ns.credit), ns.mpor),
        "EQ": _single_factor_class(list(ns.equity), ns.mpor),
        "CO": commodity_addon(ns.commodity, mpor=ns.mpor),
    }
    addon = sum(by.values())
    rc = replacement_cost(v, c, margined=ns.mpor is not None, th=ns.th, mta=ns.mta, nica=ns.nica)
    m = multiplier(v, c, addon)
    pfe = m * addon
    e = ALPHA * (rc + pfe)
    # CRE52.2: a margined EAD is capped at the unmargined EAD of the same set.
    if ns.mpor is not None:
        un = ead(NettingSet(ns.mtm, ns.collateral, ns.ir, ns.fx, ns.credit, ns.equity, ns.commodity))
        e = min(e, un.ead)
    return SaccrResult(rc, addon, m, pfe, e, by)
