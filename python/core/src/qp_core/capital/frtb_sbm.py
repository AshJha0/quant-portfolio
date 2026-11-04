"""FRTB standardised approach: sensitivities-based method (SBM), delta and vega
for general interest-rate risk (GIRR) and FX.

Plain idea
----------
The trading-book standard rule takes each sensitivity a desk already
reports (PV01 per curve tenor, FX delta, vega), multiplies it by a
regulator-set risk weight (how far that factor can move in a stress), and
adds the results up with regulator-set correlations, first inside a bucket
(one currency, or one currency pair) and then across buckets. Because the
correlations themselves are uncertain, the sum is done three times, with
medium, high and low correlations, and the worst one is kept.

Formal statement
----------------
MAR21 (BCBS d457, January 2019; consolidated Basel Framework MAR21):

* WS_k = RW_k s_k (MAR21.4(3)).
* K_b = sqrt(max(0, sum_k WS_k^2 + sum_{k != l} rho_kl WS_k WS_l)) (21.4(4)).
* Charge = sqrt(sum_b K_b^2 + sum_{b != c} gamma_bc S_b S_c) with
  S_b = sum_k WS_k; if the radicand is negative, S_b is clipped to
  [-K_b, K_b] (21.4(5)).
* Scenarios (21.6): medium as tabled; high min(1.25 rho, 1); low
  max(2 rho - 1, 0.75 rho); applied to rho and gamma alike.
* GIRR delta: tenors 0.25..30y with risk weights 1.7 % ... 1.1 % (Table 1,
  21.42); inflation and cross-currency basis 1.6 % (21.43); divided by
  sqrt 2 for specified currencies at the bank's discretion (21.44);
  rho = max(exp(-0.03 |T_k - T_l| / min(T_k, T_l)), 0.4) on one curve
  (21.46), times 0.999 across curves (21.45, 21.47); inflation to a yield
  tenor 0.4 (21.48); cross-currency basis to anything else 0 (21.49);
  gamma = 0.5 across currencies (21.50).
* FX delta: RW 15 %, divided by sqrt 2 for specified pairs (21.87-88);
  one risk factor per bucket; gamma = 0.6 (21.89).
* Vega: RW = min(0.55 sqrt(LH / 10), 1): 100 % for GIRR (LH 60) and FX
  (LH 40) (21.92). GIRR rho = min(e^{-0.01 |T_k-T_l|/min} for option
  maturity times the same for underlying maturity, 1) (21.93); FX rho = the
  option-maturity term alone (21.94, delta rho is 1 within an FX bucket);
  gamma as for delta (21.95). Vega sensitivity is vega times implied vol
  (21.25).

Worked example
--------------
One USD PV01 of +10,000 at the 5y tenor and -10,000 at 10y on the same
curve, no sqrt 2 relief: WS = +110 and -110; rho(5y, 10y) = e^{-0.03} =
0.97045; K = sqrt(110^2 + 110^2 - 2 (0.97045)(110^2)) = 26.74.

>>> r = girr_delta([GirrDelta("USD", "OIS", 5.0, 10_000), GirrDelta("USD", "OIS", 10.0, -10_000)],
...                specified_relief=False)
>>> round(r["medium"], 2)
26.74

Sources: BCBS d457 (2019) MAR21.4-21.95, text and Tables 1, 2 and 13
reproduced in the tests (Table 2 entry by entry). Evidence class:
regulatory text. Curvature, the other risk classes, the residual risk
add-on and default risk charge are out of scope (W14b covers FRTB IMA).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from qp_core.model import model

from ..errors import InputError

__all__ = [
    "FX_RW",
    "FX_SPECIFIED_PAIRS",
    "GIRR_RW",
    "GIRR_TENORS",
    "SCENARIOS",
    "SPECIFIED_CURRENCIES",
    "VEGA_LH",
    "FxDelta",
    "FxVega",
    "GirrDelta",
    "GirrVega",
    "across_buckets",
    "fx_delta",
    "fx_vega",
    "girr_delta",
    "girr_delta_rho",
    "girr_vega",
    "scenario_corr",
    "tenor_rho",
    "vega_rw",
    "within_bucket",
]

GIRR_TENORS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0)
GIRR_RW: tuple[float, ...] = (0.017, 0.017, 0.016, 0.013, 0.012, 0.011, 0.011, 0.011, 0.011, 0.011)
GIRR_RW_INFLATION_XCCY = 0.016
SPECIFIED_CURRENCIES = frozenset({"EUR", "USD", "GBP", "AUD", "JPY", "SEK", "CAD"})
FX_RW = 0.15
FX_SPECIFIED_PAIRS = frozenset(
    "USD" + c for c in ("EUR JPY GBP AUD CAD CHF MXN CNY NZD RUB HKD SGD TRY KRW SEK ZAR INR NOK BRL".split())
)
VEGA_LH = {
    "GIRR": 60.0,
    "CSR_NONSEC": 120.0,
    "EQ_LARGE": 20.0,
    "EQ_SMALL": 60.0,
    "COMMODITY": 120.0,
    "FX": 40.0,
}
RW_SIGMA = 0.55
SCENARIOS = ("medium", "high", "low")


def vega_rw(risk_class: str) -> float:
    """min(0.55 sqrt(LH / 10), 1) (MAR21.92 footnote 24).

    >>> vega_rw("GIRR"), round(vega_rw("EQ_LARGE"), 4)
    (1.0, 0.7778)
    """
    return min(RW_SIGMA * math.sqrt(VEGA_LH[risk_class] / 10.0), 1.0)


def scenario_corr(rho: float, scenario: str) -> float:
    """Correlation under a MAR21.6 scenario."""
    if scenario == "medium":
        return rho
    if scenario == "high":
        return min(1.25 * rho, 1.0)
    if scenario == "low":
        return max(2.0 * rho - 1.0, 0.75 * rho)
    raise InputError(f"unknown scenario {scenario!r}")


def tenor_rho(tk: float, tl: float, theta: float) -> float:
    """exp(-theta |T_k - T_l| / min(T_k, T_l))."""
    if not (tk > 0.0 and tl > 0.0):
        raise InputError("tenors must be positive")
    return math.exp(-theta * abs(tk - tl) / min(tk, tl))


def girr_delta_rho(tk: float, tl: float, *, same_curve: bool = True) -> float:
    """MAR21.46-47: max(exp(-3 % |dT| / min T), 40 %), times 99.9 % across curves.

    >>> round(girr_delta_rho(1.0, 5.0), 4), round(girr_delta_rho(1.0, 5.0, same_curve=False), 4)
    (0.8869, 0.886)
    """
    r = max(tenor_rho(tk, tl, 0.03), 0.4)
    return r if same_curve else 0.999 * r


def within_bucket(ws: Sequence[float], rho: np.ndarray) -> float:
    """K_b = sqrt(max(0, WS' R WS)) with unit diagonal (MAR21.4(4))."""
    w = np.asarray(ws, dtype=float)
    r = np.asarray(rho, dtype=float)
    if r.shape != (w.size, w.size):
        raise InputError("rho must be square and match ws")
    return math.sqrt(max(0.0, float(w @ r @ w)))


def across_buckets(k: Sequence[float], s: Sequence[float], gamma: np.ndarray) -> float:
    """sqrt(sum K_b^2 + sum_{b != c} gamma S_b S_c), with the clipped S_b
    fallback when the radicand is negative (MAR21.4(5))."""
    kk = np.asarray(k, dtype=float)
    ss = np.asarray(s, dtype=float)
    g = np.asarray(gamma, dtype=float).copy()
    if g.shape != (kk.size, kk.size) or ss.size != kk.size:
        raise InputError("gamma must be square and match k and s")
    np.fill_diagonal(g, 0.0)

    def rad(sv: np.ndarray) -> float:
        return float(kk @ kk + sv @ g @ sv)

    q = rad(ss)
    if q < 0.0:
        q = rad(np.clip(ss, -kk, kk))
    return math.sqrt(max(q, 0.0))


# -------------------------------------------------------------- positions


@dataclass(frozen=True)
class GirrDelta:
    """PV01-style sensitivity s_k (MAR21.19) to ``tenor`` of ``curve`` in
    ``currency``. ``curve`` "INFLATION" or a name starting "XCCY" marks the
    inflation and cross-currency basis factors (tenor ignored)."""

    currency: str
    curve: str
    tenor: float
    sensitivity: float


@dataclass(frozen=True)
class GirrVega:
    currency: str
    option_tenor: float
    underlying_tenor: float
    sensitivity: float


@dataclass(frozen=True)
class FxDelta:
    """Sensitivity to a 1 % move of the ``pair`` rate, scaled per MAR21.24."""

    pair: str
    sensitivity: float


@dataclass(frozen=True)
class FxVega:
    pair: str
    option_tenor: float
    sensitivity: float


def _kind(curve: str) -> str:
    if curve == "INFLATION":
        return "infl"
    if curve.startswith("XCCY"):
        return "xccy"
    return "yield"


def _girr_rw(d: GirrDelta) -> float:
    if _kind(d.curve) != "yield":
        return GIRR_RW_INFLATION_XCCY
    try:
        return GIRR_RW[GIRR_TENORS.index(float(d.tenor))]
    except ValueError:
        raise InputError(f"GIRR tenor {d.tenor} is not a MAR21.8 tenor") from None


def _girr_pair_rho(a: GirrDelta, b: GirrDelta) -> float:
    ka, kb = _kind(a.curve), _kind(b.curve)
    if "xccy" in (ka, kb):
        return 1.0 if a.curve == b.curve else 0.0
    if "infl" in (ka, kb):
        return 1.0 if ka == kb else 0.4
    return girr_delta_rho(a.tenor, b.tenor, same_curve=a.curve == b.curve)


def _net(
    items: Sequence[object], key: Callable[[object], tuple[object, ...]]
) -> dict[tuple[object, ...], float]:
    out: dict[tuple[object, ...], float] = defaultdict(float)
    for it in items:
        out[key(it)] += it.sensitivity  # type: ignore[attr-defined]
    return out


def _scenario_matrix(rho: np.ndarray, scenario: str) -> np.ndarray:
    out = np.array(rho, dtype=float, copy=True)
    for idx in np.ndindex(out.shape):
        out[idx] = scenario_corr(float(out[idx]), scenario)
    return out


def _aggregate(buckets: dict[str, tuple[list[float], np.ndarray]], gamma: float) -> dict[str, float]:
    names = sorted(buckets)
    res: dict[str, float] = {}
    for sc in SCENARIOS:
        ks, ss = [], []
        for n in names:
            ws, rho = buckets[n]
            r = _scenario_matrix(rho, sc)
            ks.append(within_bucket(ws, r))
            ss.append(sum(ws))
        g = np.full((len(names), len(names)), scenario_corr(gamma, sc))
        res[sc] = across_buckets(ks, ss, g) if names else 0.0
    res["charge"] = max(res[s] for s in SCENARIOS)
    return res


@model(id="CORE-FRTB-SBM-GIRR-FX")
def girr_delta(sens: Sequence[GirrDelta], *, specified_relief: bool = True) -> dict[str, float]:
    """GIRR delta charge per scenario plus ``charge`` = max (MAR21.4-6, 21.42-50).

    Sensitivities to the same (currency, curve, tenor) are netted first
    (21.4(2)). ``specified_relief`` applies the 21.44 sqrt 2 division to
    specified currencies.
    """
    net = _net(sens, lambda d: (d.currency, d.curve, float(d.tenor)))  # type: ignore[attr-defined]
    by_ccy: dict[str, list[GirrDelta]] = defaultdict(list)
    for (ccy, curve, tenor), s in net.items():
        by_ccy[str(ccy)].append(GirrDelta(str(ccy), str(curve), float(tenor), s))  # type: ignore[arg-type]
    buckets: dict[str, tuple[list[float], np.ndarray]] = {}
    for ccy, items in by_ccy.items():
        items.sort(key=lambda d: (d.curve, d.tenor))
        rel = math.sqrt(2.0) if specified_relief and ccy in SPECIFIED_CURRENCIES else 1.0
        ws = [_girr_rw(d) / rel * d.sensitivity for d in items]
        rho = np.array(
            [
                [1.0 if i == j else _girr_pair_rho(a, b) for j, b in enumerate(items)]
                for i, a in enumerate(items)
            ]
        )
        buckets[ccy] = (ws, rho)
    return _aggregate(buckets, 0.5)


@model(id="CORE-FRTB-SBM-GIRR-FX")
def girr_vega(sens: Sequence[GirrVega]) -> dict[str, float]:
    """GIRR vega charge per scenario (MAR21.92-95)."""
    net = _net(sens, lambda v: (v.currency, float(v.option_tenor), float(v.underlying_tenor)))  # type: ignore[attr-defined]
    by: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for (ccy, to, tu), s in net.items():
        by[str(ccy)].append((float(to), float(tu), s))  # type: ignore[arg-type]
    rw = vega_rw("GIRR")
    buckets: dict[str, tuple[list[float], np.ndarray]] = {}
    for ccy, items in by.items():
        items.sort()
        ws = [rw * s for _, _, s in items]
        rho = np.array(
            [
                [
                    1.0 if i == j else min(tenor_rho(a[0], b[0], 0.01) * tenor_rho(a[1], b[1], 0.01), 1.0)
                    for j, b in enumerate(items)
                ]
                for i, a in enumerate(items)
            ]
        )
        buckets[ccy] = (ws, rho)
    return _aggregate(buckets, 0.5)


def _fx_rw(pair: str, relief: bool) -> float:
    p = pair.replace("/", "").upper()
    spec = p in FX_SPECIFIED_PAIRS or (p[3:] + p[:3]) in FX_SPECIFIED_PAIRS
    return FX_RW / math.sqrt(2.0) if relief and spec else FX_RW


@model(id="CORE-FRTB-SBM-GIRR-FX")
def fx_delta(sens: Sequence[FxDelta], *, specified_relief: bool = True) -> dict[str, float]:
    """FX delta charge per scenario (MAR21.86-89). Each pair string is its own
    bucket, so quote every pair in one direction (the caller converts
    JPYUSD to USDJPY with the sign of the sensitivity flipped). First-order crosses
    (footnote 23) are not detected; pass ``specified_relief=False`` or list
    only USD pairs to stay conservative."""
    net = _net(sens, lambda d: (d.pair,))  # type: ignore[attr-defined]
    buckets = {
        str(p[0]): ([_fx_rw(str(p[0]), specified_relief) * s], np.ones((1, 1))) for p, s in net.items()
    }
    return _aggregate(buckets, 0.6)


@model(id="CORE-FRTB-SBM-GIRR-FX")
def fx_vega(sens: Sequence[FxVega]) -> dict[str, float]:
    """FX vega charge per scenario (MAR21.92-95)."""
    net = _net(sens, lambda v: (v.pair, float(v.option_tenor)))  # type: ignore[attr-defined]
    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (pair, to), s in net.items():
        by[str(pair)].append((float(to), s))  # type: ignore[arg-type]
    rw = vega_rw("FX")
    buckets: dict[str, tuple[list[float], np.ndarray]] = {}
    for pair, items in by.items():
        items.sort()
        ws = [rw * s for _, s in items]
        rho = np.array(
            [
                [1.0 if i == j else tenor_rho(a[0], b[0], 0.01) for j, b in enumerate(items)]
                for i, a in enumerate(items)
            ]
        )
        buckets[pair] = (ws, rho)
    return _aggregate(buckets, 0.6)
