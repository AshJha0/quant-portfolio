"""FRTB SBM delta and vega for CSR non-securitisation, equity and commodity.

Plain idea
----------
The same recipe as GIRR and FX (:mod:`.frtb_sbm`): weight each sensitivity
by a regulator's risk weight, add the weighted numbers inside a bucket with
the bucket's correlations, then across buckets, under three correlation
scenarios, and keep the worst. What changes by risk class is what a bucket
is (a credit rating and sector; a market cap, economy and sector; a group of
commodities) and the correlation tables. Two buckets are special: the
"other sector" bucket (CSR 16, equity 11) has no diversification at all, its
charge is the sum of the absolute weighted sensitivities.

Formal statement
----------------
BCBS d457 (2019), MAR21:

* CSR non-sec (21.51-57): WS = RW_b s, RW from Table 4. Within buckets 1-15,
  rho = rho_name rho_tenor rho_basis with 35 % / 65 % / 99.9 % when the names,
  tenors or curves (bond vs CDS) differ; buckets 17-18 the same with 80 %
  for names. Bucket 16: K = sum |WS| (21.56). Across buckets gamma =
  gamma_rating gamma_sector, gamma_rating = 50 % when both are in 1-15 with
  different rating classes, gamma_sector from Table 5 (21.57).
* Equity (21.72-80): spot and repo factors per name, RW from Table 10. Within
  a bucket: same name spot vs repo 99.9 %; different names rho_b (spot-spot
  and repo-repo) or 99.9 % rho_b (spot-repo). Bucket 11: sum |WS|. Across:
  15 % within 1-10, 0 % with 11, 75 % between 12 and 13, 45 % otherwise.
* Commodity (21.81-85): RW per bucket (Table 11); rho = rho_cty rho_tenor
  rho_basis with rho_cty from Table 12 when the commodities differ, 99 % when
  tenors differ, 99.9 % when delivery locations differ. Across: 20 %, 0 % with
  bucket 11.
* Vega (21.92-95): RW = min(0.55 sqrt(LH/10), 1) (Table 13); within a bucket
  rho = min(rho_delta rho_option_maturity, 1), where rho_delta is the delta
  correlation of the underlyings and rho_option_maturity = e^{-1% |dT|/min T};
  gamma as for delta.
* Scenarios (21.6) are applied to every rho and gamma, and the MAR21.4(5)(b)
  clipped-S fallback is used across buckets.

Interpretations (stated, tested, and listed in the handback): a CSR or
commodity vega risk factor has no underlying tenor, so its rho_delta is the
name/commodity times basis factor without the tenor factor (commodity vega
has no location dimension, MAR21.13(2)); for
repo-repo in the index buckets 12-13 we use the 80 % of 21.78(2)(e).

Worked example
--------------
Two IG financials (bucket 3, RW 5 %), 5y bond-curve CS01 +1,000 for issuer A
and 5y CDS CS01 -1,000 for issuer B: WS = +50 and -50, rho = 35 % x 99.9 %,
K = sqrt(2 x 50^2 (1 - 0.34965)) = 57.02.

>>> r = csr_delta([CsrDelta(3, "A", 5.0, 1_000.0), CsrDelta(3, "B", 5.0, -1_000.0, basis="CDS")])
>>> round(r["medium"], 2)
57.02

Sources: BCBS d457 (January 2019) MAR21.51-21.95; tables in
:mod:`.frtb_tables`. Evidence class: regulatory text.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import InputError
from ..model import model
from . import frtb_tables as T
from .frtb_sbm import SCENARIOS, across_buckets, scenario_corr, tenor_rho, vega_rw, within_bucket

__all__ = [
    "BucketInput",
    "ComDelta",
    "ComVega",
    "CsrDelta",
    "CsrVega",
    "EqDelta",
    "EqVega",
    "aggregate",
    "com_delta",
    "com_gamma",
    "com_vega",
    "csr_delta",
    "csr_gamma",
    "csr_vega",
    "eq_delta",
    "eq_gamma",
    "eq_vega",
    "eq_vega_rw",
]


# -------------------------------------------------------------- aggregation


@dataclass(frozen=True)
class BucketInput:
    """Weighted sensitivities of one bucket and their medium correlations.
    ``other`` buckets are aggregated as sum |WS| (MAR21.56, 21.79)."""

    ws: tuple[float, ...]
    rho: np.ndarray
    other: bool = False


def _scen(m: np.ndarray, sc: str) -> np.ndarray:
    out = np.array(m, dtype=float, copy=True)
    for idx in np.ndindex(out.shape):
        if idx[0] != idx[1]:
            out[idx] = scenario_corr(float(out[idx]), sc)
    return out


@model(id="CORE-FRTB-SBM")
def aggregate(buckets: dict[int, BucketInput], gamma: Callable[[int, int], float]) -> dict[str, float]:
    """Charge per MAR21.6 scenario plus ``charge`` = their max."""
    names = sorted(buckets)
    res: dict[str, float] = {}
    for sc in SCENARIOS:
        ks, ss = [], []
        for b in names:
            bi = buckets[b]
            ks.append(sum(abs(w) for w in bi.ws) if bi.other else within_bucket(bi.ws, _scen(bi.rho, sc)))
            ss.append(sum(bi.ws))
        g = np.zeros((len(names), len(names)))
        for i, b in enumerate(names):
            for j, c in enumerate(names):
                if i != j:
                    g[i, j] = scenario_corr(gamma(b, c), sc)
        res[sc] = across_buckets(ks, ss, g) if names else 0.0
    res["charge"] = max(res[s] for s in SCENARIOS)
    return res


Key = tuple[Any, ...]


def _buckets(
    rows: Sequence[tuple[int, Key, float]],
    rw: Callable[[int, Key], float],
    rho: Callable[[int, Key, Key], float],
    other: frozenset[int] = frozenset(),
) -> dict[int, BucketInput]:
    """Net ``(bucket, key, s)`` rows on (bucket, key) (MAR21.4(2)), weight
    them and build each bucket's medium correlation matrix."""
    net: dict[int, dict[Key, float]] = defaultdict(lambda: defaultdict(float))
    for b, k, s in rows:
        net[b][k] += s
    out: dict[int, BucketInput] = {}
    for b, d in net.items():
        keys = sorted(d, key=repr)
        m = np.eye(len(keys))
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                if i != j:
                    m[i, j] = rho(b, ki, kj)
        out[b] = BucketInput(tuple(rw(b, k) * d[k] for k in keys), m, other=b in other)
    return out


def _check_bucket(b: int, table: dict[int, float], what: str) -> None:
    if b not in table:
        raise InputError(f"{what} bucket {b} is not in {sorted(table)}")


def _opt_rho(t1: float, t2: float) -> float:
    return tenor_rho(t1, t2, 0.01)


# -------------------------------------------------------------- CSR non-sec


@dataclass(frozen=True)
class CsrDelta:
    """CS01 (MAR21.20) of ``issuer`` at ``tenor`` on curve ``basis`` ("BOND"
    or "CDS") in bucket 1-18. ``covered_aa`` selects the footnote 17 RW."""

    bucket: int
    issuer: str
    tenor: float
    sensitivity: float
    basis: str = "BOND"
    covered_aa: bool = False


@dataclass(frozen=True)
class CsrVega:
    bucket: int
    issuer: str
    option_tenor: float
    sensitivity: float


def csr_gamma(b: int, c: int) -> float:
    """MAR21.57 cross-bucket correlation.

    >>> csr_gamma(3, 11), csr_gamma(1, 2), round(csr_gamma(1, 10), 4)
    (0.5, 0.75, 0.375)
    """
    sb, sc = T.csr_sector(b), T.csr_sector(c)
    sector = 1.0 if sb == sc else T.CSR_SECTOR_GAMMA[(sb, sc)]
    both = b <= 15 and c <= 15
    rating = T.CSR_GAMMA_RATING if both and ((b <= 8) != (c <= 8)) else 1.0
    return rating * sector


def _csr_name_rho(b: int, n1: str, n2: str) -> float:
    if n1 == n2:
        return 1.0
    return T.CSR_RHO_NAME_INDEX if b in (17, 18) else T.CSR_RHO_NAME


def csr_delta(sens: Sequence[CsrDelta]) -> dict[str, float]:
    """CSR non-securitisation delta charge (MAR21.51-57)."""
    for d in sens:
        _check_bucket(d.bucket, T.CSR_RW, "CSR")
        if float(d.tenor) not in T.CSR_TENORS:
            raise InputError(f"CSR tenor {d.tenor} is not a MAR21.9 tenor")
    rows = [(d.bucket, (d.issuer, float(d.tenor), d.basis, d.covered_aa), d.sensitivity) for d in sens]

    def rw(b: int, k: Key) -> float:
        return T.CSR_RW_COVERED_AA if (k[3] and b == 8) else T.CSR_RW[b]

    def rho(b: int, p: Key, q: Key) -> float:
        return (
            _csr_name_rho(b, p[0], q[0])
            * (1.0 if p[1] == q[1] else T.CSR_RHO_TENOR)
            * (1.0 if p[2] == q[2] else T.CSR_RHO_BASIS)
        )

    return aggregate(_buckets(rows, rw, rho, frozenset({16})), csr_gamma)


def csr_vega(sens: Sequence[CsrVega]) -> dict[str, float]:
    """CSR non-securitisation vega charge (MAR21.92-95)."""
    for v in sens:
        _check_bucket(v.bucket, T.CSR_RW, "CSR")
    rows = [(v.bucket, (v.issuer, float(v.option_tenor)), v.sensitivity) for v in sens]
    w = vega_rw("CSR_NONSEC")

    def rho(b: int, p: Key, q: Key) -> float:
        return min(_csr_name_rho(b, p[0], q[0]) * _opt_rho(p[1], q[1]), 1.0)

    return aggregate(_buckets(rows, lambda b, k: w, rho, frozenset({16})), csr_gamma)


# -------------------------------------------------------------- equity


@dataclass(frozen=True)
class EqDelta:
    """Equity delta (MAR21.21-22): ``kind`` "SPOT" or "REPO"."""

    bucket: int
    issuer: str
    sensitivity: float
    kind: str = "SPOT"


@dataclass(frozen=True)
class EqVega:
    bucket: int
    issuer: str
    option_tenor: float
    sensitivity: float


def eq_gamma(b: int, c: int) -> float:
    """MAR21.80.

    >>> eq_gamma(1, 10), eq_gamma(11, 12), eq_gamma(12, 13), eq_gamma(5, 12)
    (0.15, 0.0, 0.75, 0.45)
    """
    if 11 in (b, c):
        return 0.0
    if b <= 10 and c <= 10:
        return T.EQ_GAMMA_SINGLE
    if {b, c} == {12, 13}:
        return T.EQ_GAMMA_INDEX
    return T.EQ_GAMMA_MIXED


def eq_vega_rw(bucket: int) -> float:
    """Table 13: 77.78 % for large cap and indices, 100 % for small cap and other.

    >>> round(eq_vega_rw(5), 4), eq_vega_rw(9)
    (0.7778, 1.0)
    """
    return vega_rw("EQ_SMALL" if bucket in (9, 10, 11) else "EQ_LARGE")


def _eq_rho(b: int, p: Key, q: Key) -> float:
    same_name, same_kind = p[0] == q[0], p[1] == q[1]
    if same_name:
        return 1.0 if same_kind else T.EQ_RHO_SPOT_REPO
    r = T.EQ_RHO.get(b, 0.0)
    return r if same_kind else r * T.EQ_RHO_SPOT_REPO


def eq_delta(sens: Sequence[EqDelta]) -> dict[str, float]:
    """Equity delta charge (MAR21.72-80)."""
    for d in sens:
        _check_bucket(d.bucket, T.EQ_RW_SPOT, "equity")
        if d.kind not in ("SPOT", "REPO"):
            raise InputError("equity kind must be SPOT or REPO")
    rows = [(d.bucket, (d.issuer, d.kind), d.sensitivity) for d in sens]

    def rw(b: int, k: Key) -> float:
        return (T.EQ_RW_SPOT if k[1] == "SPOT" else T.EQ_RW_REPO)[b]

    return aggregate(_buckets(rows, rw, _eq_rho, frozenset({11})), eq_gamma)


def eq_vega(sens: Sequence[EqVega]) -> dict[str, float]:
    """Equity vega charge (MAR21.92-95)."""
    for v in sens:
        _check_bucket(v.bucket, T.EQ_RW_SPOT, "equity")
    rows = [(v.bucket, (v.issuer, float(v.option_tenor)), v.sensitivity) for v in sens]

    def rho(b: int, p: Key, q: Key) -> float:
        d = 1.0 if p[0] == q[0] else T.EQ_RHO.get(b, 0.0)
        return min(d * _opt_rho(p[1], q[1]), 1.0)

    return aggregate(_buckets(rows, lambda b, k: eq_vega_rw(b), rho, frozenset({11})), eq_gamma)


# -------------------------------------------------------------- commodity


@dataclass(frozen=True)
class ComDelta:
    """Commodity delta (MAR21.23) to ``commodity`` at ``tenor`` delivered at
    ``location``."""

    bucket: int
    commodity: str
    tenor: float
    sensitivity: float
    location: str = ""


@dataclass(frozen=True)
class ComVega:
    """Commodity vega (MAR21.13(2)): one factor per commodity and option
    maturity. ``location`` is kept for the record and is not a risk factor:
    rows that differ only in location net."""

    bucket: int
    commodity: str
    option_tenor: float
    sensitivity: float
    location: str = ""


def com_gamma(b: int, c: int) -> float:
    """MAR21.85: 20 %, or 0 % with bucket 11."""
    return 0.0 if 11 in (b, c) else T.COM_GAMMA


def _com_cty_loc(b: int, p: Key, q: Key) -> float:
    return (1.0 if p[0] == q[0] else T.COM_RHO[b]) * (1.0 if p[2] == q[2] else T.COM_RHO_BASIS)


def com_delta(sens: Sequence[ComDelta]) -> dict[str, float]:
    """Commodity delta charge (MAR21.81-85)."""
    for d in sens:
        _check_bucket(d.bucket, T.COM_RW, "commodity")
    rows = [(d.bucket, (d.commodity, float(d.tenor), d.location), d.sensitivity) for d in sens]

    def rho(b: int, p: Key, q: Key) -> float:
        return _com_cty_loc(b, p, q) * (1.0 if p[1] == q[1] else T.COM_RHO_TENOR)

    return aggregate(_buckets(rows, lambda b, k: T.COM_RW[b], rho), com_gamma)


def com_vega(sens: Sequence[ComVega]) -> dict[str, float]:
    """Commodity vega charge (MAR21.92-95)."""
    for v in sens:
        _check_bucket(v.bucket, T.COM_RW, "commodity")
    rows = [(v.bucket, (v.commodity, float(v.option_tenor), ""), v.sensitivity) for v in sens]
    w = vega_rw("COMMODITY")

    def rho(b: int, p: Key, q: Key) -> float:
        return min(_com_cty_loc(b, p, q) * _opt_rho(p[1], q[1]), 1.0)

    return aggregate(_buckets(rows, lambda b, k: w, rho), com_gamma)
