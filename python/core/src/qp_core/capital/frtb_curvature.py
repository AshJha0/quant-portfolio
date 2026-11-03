"""FRTB SBM curvature risk for all five non-securitisation risk classes.

Plain idea
----------
Delta charges a straight-line move. An option's value is curved, so after a
large move it can lose more than the straight line says. Curvature asks: if
the risk factor jumps up (or down) by its full risk weight, how much worse
is the full revaluation than the delta estimate? That shortfall, CVR, is
then added up across factors and buckets with correlations that are the
squares of the delta ones, and only losses are allowed to add up fully: two
gains never offset each other's charge.

Formal statement
----------------
BCBS d457 (2019) MAR21.5 and 21.96-101. For factor k and instruments i:

    CVR_k^+ = -sum_i [ V_i(x_k^{RW+}) - V_i(x_k) - RW_k s_ik ]
    CVR_k^- = -sum_i [ V_i(x_k^{RW-}) - V_i(x_k) + RW_k s_ik ]

Within a bucket, K_b^{+/-} = sqrt(max(0, sum_k max(CVR_k, 0)^2
+ sum_{k != l} rho_kl CVR_k CVR_l psi(CVR_k, CVR_l))), psi = 0 when both
are negative, else 1; K_b = max(K_b^+, K_b^-), the larger one "selected"
(ties: the side with the larger sum of CVR). Across buckets:

    curvature = sqrt(max(0, sum_b K_b^2 + sum_{b != c} gamma_bc S_b S_c psi(S_b, S_c)))

with S_b the sum of the selected side's CVR. Correlations: squares of the
delta rho and gamma (21.100-101), except CSR where only the name term is
kept before squaring (21.100(1)). The other-sector buckets (CSR 16, equity
11) use K_b = max(sum max(CVR^+, 0), sum max(CVR^-, 0)) (21.56(2), 21.79(2)).
MAR21.6 scenarios are applied to the squared correlations. Risk weights
(21.98-99): FX and equity, the delta RW as a relative shift; GIRR, CSR and
commodity, the largest delta RW of the class/bucket as a parallel shift of
the whole curve (GIRR 1.7 %). FX CVR may be divided by 1.5 for options that
do not reference the reporting currency (21.98).

Worked example
--------------
One equity factor in bucket 5 (RW 30 %), a long straddle-like position with
CVR^+ = 12 and CVR^- = 9 (both losses beyond delta): K = 12 (up selected),
and with a single bucket the charge is 12 in every scenario.

>>> curvature_charge({5: [("X", 12.0, 9.0)]}, lambda b, k, l: 0.0, lambda b, c: 0.0)["charge"]
12.0

Sources: BCBS d457 MAR21.5, 21.56, 21.79, 21.96-21.101. Evidence class:
regulatory text. The CVR inputs come from full revaluation, which this module
leaves to the caller (``net_cvr`` does the arithmetic per instrument).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..errors import InputError
from . import frtb_tables as T
from .frtb_classes import com_gamma, csr_gamma, eq_gamma
from .frtb_sbm import FX_RW, GIRR_RW, SCENARIOS, scenario_corr

__all__ = [
    "Cvr",
    "com_curvature",
    "csr_curvature",
    "curvature_charge",
    "curvature_rw",
    "eq_curvature",
    "fx_curvature",
    "girr_curvature",
    "net_cvr",
]

FX_CURVATURE_SCALAR = 1.5


def net_cvr(v0: float, v_up: float, v_down: float, rw: float, s: float) -> tuple[float, float]:
    """(CVR^+, CVR^-) of one instrument (MAR21.5(2)). ``rw * s`` is the delta
    P&L of the shock, so for a linear instrument both are 0.

    >>> net_cvr(10.0, 13.0, 7.0, 0.3, 10.0)
    (0.0, 0.0)
    """
    return v0 - v_up + rw * s, v0 - v_down - rw * s


def curvature_rw(risk_class: str, bucket: int | None = None, *, fx_relief: bool = False) -> float:
    """Curvature shock size (MAR21.98-99).

    >>> curvature_rw("GIRR"), curvature_rw("EQ", 12), curvature_rw("CSR", 11), curvature_rw("COM", 4)
    (0.017, 0.15, 0.12, 0.8)
    """
    if risk_class == "GIRR":
        return max(GIRR_RW)
    if risk_class == "FX":
        return FX_RW / math.sqrt(2.0) if fx_relief else FX_RW
    table = {"CSR": T.CSR_RW, "EQ": T.EQ_RW_SPOT, "COM": T.COM_RW}.get(risk_class)
    if table is None or bucket not in table:
        raise InputError(f"no curvature RW for {risk_class!r} bucket {bucket!r}")
    return table[bucket]


def _psi(a: float, b: float) -> float:
    return 0.0 if (a < 0.0 and b < 0.0) else 1.0


def _k_side(c: Sequence[float], rho: Callable[[int, int], float]) -> float:
    q = sum(max(x, 0.0) ** 2 for x in c)
    for i, a in enumerate(c):
        for j, b in enumerate(c):
            if i != j:
                q += rho(i, j) * a * b * _psi(a, b)
    return math.sqrt(max(q, 0.0))


def curvature_charge(
    buckets: dict[int, list[tuple[object, float, float]]],
    delta_rho: Callable[[int, object, object], float],
    delta_gamma: Callable[[int, int], float],
    *,
    other: frozenset[int] = frozenset(),
) -> dict[str, float]:
    """Curvature charge per scenario and ``charge`` = max.

    ``buckets[b]`` lists ``(factor key, CVR^+, CVR^-)`` (keys repeated are
    summed); ``delta_rho(b, k, l)`` and ``delta_gamma(b, c)`` are the medium
    *delta* correlations, squared here (21.100-101).
    """
    nets: dict[int, list[tuple[object, float, float]]] = {}
    for b, rows in buckets.items():
        acc: dict[object, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for key, up, dn in rows:
            acc[key][0] += up
            acc[key][1] += dn
        nets[b] = [(k, v[0], v[1]) for k, v in sorted(acc.items(), key=lambda kv: repr(kv[0]))]
    names = sorted(nets)
    res: dict[str, float] = {}
    for sc in SCENARIOS:
        ks: list[float] = []
        ws: list[float] = []
        for b in names:
            rows = nets[b]
            ups = [r[1] for r in rows]
            dns = [r[2] for r in rows]
            if b in other:
                kp = sum(max(x, 0.0) for x in ups)
                km = sum(max(x, 0.0) for x in dns)
            else:

                def rho(i: int, j: int, b: int = b, rows: list = rows, sc: str = sc) -> float:  # type: ignore[type-arg]
                    return scenario_corr(delta_rho(b, rows[i][0], rows[j][0]) ** 2, sc)

                kp, km = _k_side(ups, rho), _k_side(dns, rho)
            up_sel = kp > km or (kp == km and sum(ups) > sum(dns))
            ks.append(kp if up_sel else km)
            ws.append(sum(ups) if up_sel else sum(dns))
        q = sum(k * k for k in ks)
        for i, b in enumerate(names):
            for j, c in enumerate(names):
                if i != j:
                    q += scenario_corr(delta_gamma(b, c) ** 2, sc) * ws[i] * ws[j] * _psi(ws[i], ws[j])
        res[sc] = math.sqrt(max(q, 0.0))
    res["charge"] = max(res[s] for s in SCENARIOS)
    return res


# -------------------------------------------------------------- per class


@dataclass(frozen=True)
class Cvr:
    """Net CVR of one curvature risk factor. ``bucket`` is an int bucket
    (CSR, equity, commodity) or a currency/pair string (GIRR, FX); ``name``
    the issuer/commodity; ``location`` the commodity delivery location;
    ``fx_scalar`` divides an FX factor's CVR by 1.5 (21.98)."""

    bucket: int | str
    name: str
    up: float
    down: float
    location: str = ""
    fx_scalar: bool = False


def _string_buckets(
    rows: Sequence[Cvr], scale: Callable[[Cvr], float]
) -> dict[int, list[tuple[object, float, float]]]:
    labels = sorted({str(r.bucket) for r in rows})
    out: dict[int, list[tuple[object, float, float]]] = defaultdict(list)
    for r in rows:
        f = scale(r)
        out[labels.index(str(r.bucket))].append((r.name, r.up / f, r.down / f))
    return out


def girr_curvature(rows: Sequence[Cvr]) -> dict[str, float]:
    """GIRR: one curvature factor per currency (all its yield curves shifted
    together, 21.5(1)(a)); gamma = 50 %, squared."""
    b = _string_buckets([Cvr(r.bucket, "", r.up, r.down) for r in rows], lambda r: 1.0)
    return curvature_charge(b, lambda *_: 1.0, lambda *_: 0.5)


def fx_curvature(rows: Sequence[Cvr]) -> dict[str, float]:
    """FX: one factor per pair; gamma = 60 %, squared; optional 1.5 scalar."""
    b = _string_buckets(
        [Cvr(r.bucket, "", r.up, r.down, fx_scalar=r.fx_scalar) for r in rows],
        lambda r: FX_CURVATURE_SCALAR if r.fx_scalar else 1.0,
    )
    return curvature_charge(b, lambda *_: 1.0, lambda *_: 0.6)


def _int_buckets(
    rows: Sequence[Cvr], table: dict[int, float], what: str, key: Callable[[Cvr], object]
) -> dict[int, list[tuple[object, float, float]]]:
    out: dict[int, list[tuple[object, float, float]]] = defaultdict(list)
    for r in rows:
        if not isinstance(r.bucket, int) or r.bucket not in table:
            raise InputError(f"{what} bucket {r.bucket!r} is not in {sorted(table)}")
        out[r.bucket].append((key(r), r.up, r.down))
    return out


def csr_curvature(rows: Sequence[Cvr]) -> dict[str, float]:
    """CSR non-sec: rho = rho_name^2 only (21.100(1)); bucket 16 is other."""
    b = _int_buckets(rows, T.CSR_RW, "CSR", lambda r: r.name)

    def rho(bk: int, k: object, l: object) -> float:
        return 1.0 if k == l else (T.CSR_RHO_NAME_INDEX if bk in (17, 18) else T.CSR_RHO_NAME)

    return curvature_charge(b, rho, csr_gamma, other=frozenset({16}))


def eq_curvature(rows: Sequence[Cvr]) -> dict[str, float]:
    """Equity: one factor per issuer spot; rho_b^2; bucket 11 is other."""
    b = _int_buckets(rows, T.EQ_RW_SPOT, "equity", lambda r: r.name)
    return curvature_charge(
        b, lambda bk, k, l: 1.0 if k == l else T.EQ_RHO.get(bk, 0.0), eq_gamma, other=frozenset({11})
    )


def com_curvature(rows: Sequence[Cvr]) -> dict[str, float]:
    """Commodity: one factor per (commodity, location) curve;
    rho = (rho_cty rho_basis)^2 (the tenor term is 1 for a parallel shift)."""
    b = _int_buckets(rows, T.COM_RW, "commodity", lambda r: (r.name, r.location))

    def rho(bk: int, k: object, l: object) -> float:
        p, q = k, l
        return (1.0 if p[0] == q[0] else T.COM_RHO[bk]) * (1.0 if p[1] == q[1] else T.COM_RHO_BASIS)  # type: ignore[index]

    return curvature_charge(b, rho, com_gamma)
