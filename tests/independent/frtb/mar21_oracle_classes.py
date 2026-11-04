"""Independent FRTB SBM oracle for CSR, equity, commodity and curvature (test-only).

Second half of the MAR21 oracle: CSR non-securitisation, equity and commodity
delta and vega, and curvature for all five classes. It was written from the
text of BCBS d457 and the consolidated Basel Framework chapter MAR21,
including its FAQs, without reading the source or tests of
``qp_core.capital.frtb_classes``, ``frtb_curvature`` or ``frtb_tables``; only
their public names, signatures and dataclass fields were inspected. Every
table below was transcribed from the paragraph named beside it. Plain Python,
explicit loops, no numpy.

Paragraph map:
  MAR21.4    weighted sensitivity, K_b, across-bucket sum with the clipped
             S_b alternative when the sum is negative.
  MAR21.5    CVR+ and CVR-, K_b with psi, up/down selection and tie rule,
             S_b of the selected scenario, across-bucket sum with psi.
  MAR21.6    three correlation scenarios, applied to the final parameter.
  MAR21.9    CSR factors: delta (issuer curve, basis, tenor); vega (issuer,
             option maturity); curvature (issuer, bond and CDS one curve).
  MAR21.12   equity factors: delta spot and repo; vega and curvature spot.
  MAR21.13   commodity factors: delta (commodity, location, tenor); vega
             (commodity, option maturity, no location); curvature (commodity).
  MAR21.53   CSR delta risk weights, Table 4; footnote 19 covered bonds.
  MAR21.54   CSR correlation, buckets 1-15: name 35 %, tenor 65 %, basis 99.9 %.
  MAR21.55   CSR correlation, buckets 17-18: name 80 %.
  MAR21.56   bucket 16: sum of absolute values; curvature formula.
  MAR21.57   CSR gamma = rating times sector, Table 5.
  MAR21.77   equity risk weights, Table 10.
  MAR21.78   equity correlations within a bucket.
  MAR21.79   bucket 11: as MAR21.56.
  MAR21.80   equity gamma.
  MAR21.82   commodity risk weights, Table 11.
  MAR21.83   commodity correlation, Table 12; tenor 99 %, location 99.9 %.
  MAR21.85   commodity gamma.
  MAR21.92   vega risk weight, Table 13 with footnote 26 and its FAQ.
  MAR21.94   vega correlation; its FAQ limits rho_delta to name or commodity.
  MAR21.95   vega gamma = delta gamma.
  MAR21.98   FX and equity curvature weight; FX scalar 1.5.
  MAR21.99   GIRR, CSR, commodity curvature weight: highest bucket delta weight.
  MAR21.100  within-bucket curvature correlation = delta correlation squared,
             unused dimensions dropped.
  MAR21.101  across-bucket curvature correlation = delta gamma squared.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

from .mar21_oracle import SCENARIOS, k_bucket, scen

Result = dict[str, float]
Buckets = dict[int, tuple[list[float], list[list[float]]]]

# Counts which rarer rules ran, so a test can prove the random draws reach them.
TRACE: Counter[str] = Counter()

# MAR21.53, Table 4
CSR_RW = {
    1: 0.005, 2: 0.01, 3: 0.05, 4: 0.03, 5: 0.03, 6: 0.02, 7: 0.015, 8: 0.025, 9: 0.02,
    10: 0.04, 11: 0.12, 12: 0.07, 13: 0.085, 14: 0.055, 15: 0.05, 16: 0.12, 17: 0.015, 18: 0.05,
}  # fmt: skip
CSR_COVERED_AA_RW = 0.015  # MAR21.53 footnote 19
CSR_OTHER = 16  # MAR21.56
# MAR21.57, Table 5, upper triangle; labels 1-7 stand for 1/9 ... 7/15.
CSR_SECTOR = {
    1: {2: 0.75, 3: 0.10, 4: 0.20, 5: 0.25, 6: 0.20, 7: 0.15, 8: 0.10, 16: 0.0, 17: 0.45, 18: 0.45},
    2: {3: 0.05, 4: 0.15, 5: 0.20, 6: 0.15, 7: 0.10, 8: 0.10, 16: 0.0, 17: 0.45, 18: 0.45},
    3: {4: 0.05, 5: 0.15, 6: 0.20, 7: 0.05, 8: 0.20, 16: 0.0, 17: 0.45, 18: 0.45},
    4: {5: 0.20, 6: 0.25, 7: 0.05, 8: 0.05, 16: 0.0, 17: 0.45, 18: 0.45},
    5: {6: 0.25, 7: 0.05, 8: 0.15, 16: 0.0, 17: 0.45, 18: 0.45},
    6: {7: 0.05, 8: 0.20, 16: 0.0, 17: 0.45, 18: 0.45},
    7: {8: 0.05, 16: 0.0, 17: 0.45, 18: 0.45},
    8: {16: 0.0, 17: 0.45, 18: 0.45},
    16: {17: 0.0, 18: 0.0},
    17: {18: 0.75},
}

# MAR21.77, Table 10
EQ_RW_SPOT = {
    1: 0.55, 2: 0.60, 3: 0.45, 4: 0.55, 5: 0.30, 6: 0.35, 7: 0.40,
    8: 0.50, 9: 0.70, 10: 0.50, 11: 0.70, 12: 0.15, 13: 0.25,
}  # fmt: skip
EQ_RW_REPO = {
    1: 0.0055, 2: 0.0060, 3: 0.0045, 4: 0.0055, 5: 0.0030, 6: 0.0035, 7: 0.0040,
    8: 0.0050, 9: 0.0070, 10: 0.0050, 11: 0.0070, 12: 0.0015, 13: 0.0025,
}  # fmt: skip
EQ_OTHER = 11  # MAR21.79

# MAR21.82, Table 11
COM_RW = {
    1: 0.30, 2: 0.35, 3: 0.60, 4: 0.80, 5: 0.40, 6: 0.45,
    7: 0.20, 8: 0.35, 9: 0.25, 10: 0.35, 11: 0.50,
}  # fmt: skip
# MAR21.83, Table 12
COM_RHO = {
    1: 0.55, 2: 0.95, 3: 0.40, 4: 0.80, 5: 0.60, 6: 0.65,
    7: 0.55, 8: 0.45, 9: 0.15, 10: 0.40, 11: 0.15,
}  # fmt: skip

GIRR_CURVATURE_RW = 0.017  # MAR21.99: the 0.25-year weight of MAR21.42
FX_CURVATURE_RW = 0.15  # MAR21.98 with MAR21.87
FX_SCALAR = 1.5  # MAR21.98
GIRR_GAMMA = 0.50  # MAR21.50
FX_GAMMA = 0.60  # MAR21.89


def _vega_rw(lh: int) -> float:
    """MAR21.92 footnote 26: min(55 % * sqrt(LH / 10), 100 %)."""
    return min(0.55 * math.sqrt(lh / 10), 1.0)


def eq_vega_rw(bucket: int) -> float:
    """MAR21.92 FAQ: 20 days for buckets 1-8, 12, 13; 60 days for 9, 10, 11."""
    return _vega_rw(60 if bucket in (9, 10, 11) else 20)


def _rho_option(a: float, b: float) -> float:
    """MAR21.93(1) as used by MAR21.94(2)."""
    return math.exp(-0.01 * abs(a - b) / min(a, b))


def _sector(b: int) -> int:
    return b - 8 if 9 <= b <= 15 else b


def csr_gamma(b: int, c: int) -> float:
    """MAR21.57."""
    if b == c:
        return 1.0
    rating = 1.0
    if b <= 15 and c <= 15 and (b <= 8) != (c <= 8):
        rating = 0.5
    sb, sc = _sector(b), _sector(c)
    sector = 1.0 if sb == sc else CSR_SECTOR[min(sb, sc)][max(sb, sc)]
    return rating * sector


def eq_gamma(b: int, c: int) -> float:
    """MAR21.80."""
    if b == c:
        return 1.0
    if b == EQ_OTHER or c == EQ_OTHER:
        return 0.0
    if b <= 10 and c <= 10:
        return 0.15
    if {b, c} == {12, 13}:
        return 0.75
    return 0.45


def com_gamma(b: int, c: int) -> float:
    """MAR21.85."""
    if b == c:
        return 1.0
    return 0.0 if 11 in (b, c) else 0.20


def _csr_name(bucket: int) -> float:
    return 0.80 if bucket in (17, 18) else 0.35  # MAR21.54(1), MAR21.55(1)


def _eq_param(bucket: int) -> float:
    """MAR21.78(2)(a)-(e)."""
    if bucket <= 4:
        return 0.15
    if bucket <= 8:
        return 0.25
    if bucket == 9:
        return 0.075
    if bucket == 10:
        return 0.125
    return 0.80


def aggregate(buckets: Buckets, gamma: Callable[[int, int], float], other: Iterable[int] = ()) -> Result:
    """MAR21.4(4)-(5) and MAR21.6 for buckets given as (WS, medium rho)."""
    out: Result = {}
    names = sorted(buckets)
    other = set(other)
    for s in SCENARIOS:
        kb: list[float] = []
        sb: list[float] = []
        for b in names:
            ws, rho = buckets[b]
            if b in other:
                k = 0.0
                for w in ws:
                    k += abs(w)
            else:
                k = k_bucket(ws, [[scen(r, s) for r in row] for row in rho])
            kb.append(k)
            sb.append(sum(ws))

        def total(sv: list[float], s: str = s, kb: list[float] = kb) -> float:
            t = 0.0
            for i in range(len(names)):
                t += kb[i] * kb[i]
                for j in range(len(names)):
                    if i != j:
                        t += scen(gamma(names[i], names[j]), s) * sv[i] * sv[j]
            return t

        t = total(sb)
        if t < 0.0:
            TRACE["delta_clipped"] += 1
            t = total([max(min(x, k), -k) for x, k in zip(sb, kb, strict=True)])
        out[s] = math.sqrt(max(0.0, t))
    out["charge"] = max(out[s] for s in SCENARIOS)
    return out


def _net(rows: Iterable[tuple[tuple, float]]) -> dict[tuple, float]:
    """MAR21.4(2): net every sensitivity to the same risk factor."""
    acc: dict[tuple, float] = {}
    for key, v in rows:
        acc[key] = acc.get(key, 0.0) + v
    return acc


def _build(
    net: dict[tuple, float],
    rw: Callable[[tuple], float],
    rho: Callable[[int, tuple, tuple], float],
) -> Buckets:
    """Group netted factors by bucket (key[0]); WS and the rho matrix."""
    out: Buckets = {}
    for b in {k[0] for k in net}:
        keys = sorted(k for k in net if k[0] == b)
        ws = [rw(k) * net[k] for k in keys]
        out[b] = (ws, [[rho(b, x, y) for y in keys] for x in keys])
    return out


def csr_delta(rows: list[tuple]) -> Result:
    """rows: (bucket, issuer, tenor, sensitivity, basis, covered_aa).

    covered_aa must be a property of the issuer: the weight is read from a set.
    """
    covered = {(r[0], r[1]) for r in rows if r[5]}
    net = _net(((r[0], r[1], r[2], r[4]), r[3]) for r in rows)

    def rw(k: tuple) -> float:
        return CSR_COVERED_AA_RW if (k[0], k[1]) in covered else CSR_RW[k[0]]

    def rho(b: int, x: tuple, y: tuple) -> float:
        name = 1.0 if x[1] == y[1] else _csr_name(b)
        tenor = 1.0 if x[2] == y[2] else 0.65
        basis = 1.0 if x[3] == y[3] else 0.999
        return name * tenor * basis

    return aggregate(_build(net, rw, rho), csr_gamma, (CSR_OTHER,))


def csr_vega(rows: list[tuple]) -> Result:
    """rows: (bucket, issuer, option_tenor, sensitivity). MAR21.94 and its FAQ."""
    net = _net(((r[0], r[1], r[2]), r[3]) for r in rows)
    rw = _vega_rw(120)

    def rho(b: int, x: tuple, y: tuple) -> float:
        name = 1.0 if x[1] == y[1] else _csr_name(b)
        return min(name * _rho_option(x[2], y[2]), 1.0)

    return aggregate(_build(net, lambda _k: rw, rho), csr_gamma, (CSR_OTHER,))


def eq_delta(rows: list[tuple]) -> Result:
    """rows: (bucket, issuer, sensitivity, kind), kind SPOT or REPO."""
    net = _net(((r[0], r[1], r[3]), r[2]) for r in rows)

    def rw(k: tuple) -> float:
        return EQ_RW_SPOT[k[0]] if k[2] == "SPOT" else EQ_RW_REPO[k[0]]

    def rho(b: int, x: tuple, y: tuple) -> float:
        if x == y:
            return 1.0
        if x[1] == y[1]:
            return 0.999  # MAR21.78(1): same issuer, spot against repo
        if b == EQ_OTHER:
            return 0.0  # never read: bucket 11 is a sum of absolute values
        p = _eq_param(b)
        return p if x[2] == y[2] else p * 0.999  # MAR21.78(2)-(4)

    return aggregate(_build(net, rw, rho), eq_gamma, (EQ_OTHER,))


def eq_vega(rows: list[tuple]) -> Result:
    """rows: (bucket, issuer, option_tenor, sensitivity)."""
    net = _net(((r[0], r[1], r[2]), r[3]) for r in rows)

    def rho(b: int, x: tuple, y: tuple) -> float:
        if b == EQ_OTHER:
            return 0.0  # never read
        delta = 1.0 if x[1] == y[1] else _eq_param(b)
        return min(delta * _rho_option(x[2], y[2]), 1.0)

    return aggregate(_build(net, lambda k: eq_vega_rw(k[0]), rho), eq_gamma, (EQ_OTHER,))


def com_delta(rows: list[tuple]) -> Result:
    """rows: (bucket, commodity, tenor, sensitivity, location)."""
    net = _net(((r[0], r[1], r[2], r[4]), r[3]) for r in rows)

    def rho(b: int, x: tuple, y: tuple) -> float:
        name = 1.0 if x[1] == y[1] else COM_RHO[b]
        tenor = 1.0 if x[2] == y[2] else 0.99
        basis = 1.0 if x[3] == y[3] else 0.999
        return name * tenor * basis

    return aggregate(_build(net, lambda k: COM_RW[k[0]], rho), com_gamma)


def com_vega(rows: list[tuple]) -> Result:
    """rows: (bucket, commodity, option_tenor, sensitivity, location).

    MAR21.13(2): no differentiation by delivery location, so location is not
    part of the risk factor and rows differing only in it net.
    """
    net = _net(((r[0], r[1], r[2]), r[3]) for r in rows)
    rw = _vega_rw(120)

    def rho(b: int, x: tuple, y: tuple) -> float:
        name = 1.0 if x[1] == y[1] else COM_RHO[b]
        return min(name * _rho_option(x[2], y[2]), 1.0)

    return aggregate(_build(net, lambda _k: rw, rho), com_gamma)


def net_cvr(v0: float, v_up: float, v_down: float, rw: float, s: float) -> tuple[float, float]:
    """MAR21.5(2) for one instrument."""
    return -(v_up - v0 - rw * s), -(v_down - v0 + rw * s)


def psi(a: float, b: float) -> float:
    """MAR21.5(3)(b), MAR21.5(4)(b)."""
    return 0.0 if a < 0.0 and b < 0.0 else 1.0


def _k_curv(c: list[float], rho: list[list[float]]) -> float:
    """MAR21.5(3)."""
    t = 0.0
    for i in range(len(c)):
        t += max(c[i], 0.0) ** 2
        for j in range(len(c)):
            if i != j:
                t += rho[i][j] * c[i] * c[j] * psi(c[i], c[j])
    return math.sqrt(max(0.0, t))


def curvature(
    buckets: dict[Any, dict[Any, tuple[float, float]]],
    delta_rho: Callable[[Any, Any, Any], float],
    delta_gamma: Callable[[Any, Any], float],
    other: Iterable[Any] = (),
) -> Result:
    """MAR21.5(3)-(4), MAR21.6, MAR21.100, MAR21.101.

    buckets: bucket -> factor -> (CVR+, CVR-), already netted over instruments.
    Correlations are given as delta parameters; they are squared first and
    the scenario is applied to the square.
    """
    other = set(other)
    names = sorted(buckets, key=str)
    out: Result = {}
    for s in SCENARIOS:
        kb: list[float] = []
        sb: list[float] = []
        for b in names:
            fac = sorted(buckets[b], key=str)
            up = [buckets[b][f][0] for f in fac]
            dn = [buckets[b][f][1] for f in fac]
            if b in other:
                # MAR21.56(2), MAR21.79(2)
                kp = sum(max(x, 0.0) for x in up)
                km = sum(max(x, 0.0) for x in dn)
            else:
                rho = [[1.0 if x == y else scen(delta_rho(b, x, y) ** 2, s) for y in fac] for x in fac]
                kp, km = _k_curv(up, rho), _k_curv(dn, rho)
            if kp > km:
                pick_up = True
            elif kp < km:
                pick_up = False
                TRACE["curv_down"] += 1
            else:
                pick_up = sum(up) > sum(dn)  # MAR21.5(3)(a)(iii)
                TRACE["curv_tie_up" if pick_up else "curv_tie_down"] += 1
            kb.append(max(kp, km))
            sb.append(sum(up) if pick_up else sum(dn))
        t = 0.0
        for i in range(len(names)):
            t += kb[i] * kb[i]
            for j in range(len(names)):
                if i != j:
                    g = scen(delta_gamma(names[i], names[j]) ** 2, s)
                    t += g * sb[i] * sb[j] * psi(sb[i], sb[j])
        if t < 0.0:
            TRACE["curv_floor"] += 1
        out[s] = math.sqrt(max(0.0, t))
    out["charge"] = max(out[s] for s in SCENARIOS)
    return out


def _group(rows: Iterable[tuple]) -> dict[Any, dict[Any, tuple[float, float]]]:
    """Sum CVR over instruments (the sum over i in MAR21.5(2))."""
    out: dict[Any, dict[Any, tuple[float, float]]] = {}
    for b, f, up, dn in rows:
        u, d = out.setdefault(b, {}).get(f, (0.0, 0.0))
        out[b][f] = (u + up, d + dn)
    return out


def _one(_b: object, _x: object, _y: object) -> float:
    return 1.0


def girr_curvature(rows: list[tuple]) -> Result:
    """rows: (currency, curve, up, down). MAR21.8(5): one factor per currency."""
    g = _group((r[0], "", r[2], r[3]) for r in rows)
    return curvature(g, _one, lambda b, c: 1.0 if b == c else GIRR_GAMMA)


def fx_curvature(rows: list[tuple]) -> Result:
    """rows: (bucket, name, up, down, fx_scalar). One factor per FX bucket."""
    scaled = []
    for b, _n, up, dn, flag in rows:
        d = FX_SCALAR if flag else 1.0  # MAR21.98
        scaled.append((b, "", up / d, dn / d))
    return curvature(_group(scaled), _one, lambda b, c: 1.0 if b == c else FX_GAMMA)


def csr_curvature(rows: list[tuple]) -> Result:
    """rows: (bucket, issuer, up, down). MAR21.100: only rho_name, squared."""
    return curvature(_group(rows), lambda b, _x, _y: _csr_name(b), csr_gamma, (CSR_OTHER,))


def eq_curvature(rows: list[tuple]) -> Result:
    """rows: (bucket, issuer, up, down). Spot only (MAR21.12(3))."""
    return curvature(_group(rows), lambda b, _x, _y: _eq_param(b), eq_gamma, (EQ_OTHER,))


def com_curvature(rows: list[tuple]) -> Result:
    """rows: (bucket, commodity, up, down). MAR21.13(3), MAR21.100: rho_commodity."""
    return curvature(_group(rows), lambda b, _x, _y: COM_RHO[b], com_gamma)
