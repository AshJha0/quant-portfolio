"""Write the Python reference values the Rust and C++ FRTB and collateral ports must reproduce.

Run from the repository root (root .venv, PYTHONPATH with qp_core):

    python python/equity/04-fixed-income/tools/gen_frtb_collateral_parity.py

Output: ``rust/core/tests/data/frtb_collateral_parity.txt``, shared by
``rust/core/tests/frtb_collateral_parity.rs`` and
``cpp/core/tests/test_frtb_collateral.cpp``. Floats are written with ``repr``
so they round-trip exactly (``inf`` included). Every expected value comes from
calling the real ``qp_core`` function; the seed makes the file reproducible.

Lines (tag first):
  ``AGG nb``                     an ``aggregate`` case with nb buckets, then per
                                 bucket ``B id other n``, ``WS``, ``RHO``
  ``GAM``                        row-major gamma over the sorted bucket ids
  ``OUT medium high low charge`` expected charges (closes AGG and CURV cases)
  ``CURV nb``                    a ``curvature_charge`` case, then per bucket
                                 ``CB id other nrows``, ``CK keys...``,
                                 ``CRHO`` (row-major over CK), ``nrows`` x
                                 ``CR key up down``; then ``GAM`` and ``OUT``
  ``CVR v0 vup vdown rw s plus minus``  a ``net_cvr`` case
  ``LG mpor t...`` + ``LGO ...`` a ``lagged_grid`` case and its output
  ``COL paths times``            a collateral case, then ``T``, ``V``,
                                 ``CSA hc hb mta mpor initial``, ``IMR``/``IMP``
                                 (``S x``, ``T ...``, ``P ...`` or ``F ...``:
                                 scalar, per time, per path, full), and the
                                 expected ``BAL``, ``LAG``, ``C``, ``E``, ``N``
"""

from __future__ import annotations

import pathlib

import numpy as np
from qp_core.capital.frtb_classes import BucketInput, aggregate, csr_gamma, eq_gamma
from qp_core.capital.frtb_curvature import Cvr, csr_curvature, curvature_charge, net_cvr
from qp_core.capital.frtb_sbm import SCENARIOS, within_bucket
from qp_core.xva.collateral import (
    Csa,
    _lag_index,
    collateral_balance,
    collateralised_exposure,
    lagged_grid,
)

OUT = pathlib.Path("rust/core/tests/data/frtb_collateral_parity.txt")
INF = float("inf")


def _row(tag: str, values) -> str:
    return " ".join([tag, *(repr(float(v)) for v in np.ravel(values))])


def _out(res: dict[str, float]) -> str:
    return _row("OUT", [res[s] for s in (*SCENARIOS, "charge")])


def _corr(rng: np.random.Generator, n: int) -> np.ndarray:
    """A random correlation matrix with a unit diagonal."""
    a = rng.normal(size=(n, n + 2))
    c = a @ a.T
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    np.fill_diagonal(c, 1.0)
    return c


def _agg_case(lines: list[str], buckets: dict[int, BucketInput], gam: np.ndarray) -> dict[str, float]:
    ids = sorted(buckets)
    res = aggregate(buckets, lambda b, c: float(gam[ids.index(b), ids.index(c)]))
    lines.append(f"AGG {len(ids)}")
    for b in ids:
        bi = buckets[b]
        lines.append(f"B {b} {int(bi.other)} {len(bi.ws)}")
        lines.append(_row("WS", bi.ws))
        lines.append(_row("RHO", bi.rho))
    lines.append(_row("GAM", gam))
    lines.append(_out(res))
    return res


def _aggregate_cases(lines: list[str], rng: np.random.Generator) -> None:
    # Random buckets; a multiple of five is an "other" bucket whose rho is ignored.
    for nb in (1, 2, 4, 7):
        buckets = {}
        for b in rng.choice(np.arange(1, 19), size=nb, replace=False):
            n = int(rng.integers(1, 7))
            other = bool(b % 5 == 0)
            buckets[int(b)] = BucketInput(
                tuple(float(x) for x in rng.normal(0.0, 100.0, n)),
                np.zeros((0, 0)) if other else _corr(rng, n),
                other,
            )
        _agg_case(lines, buckets, _corr(rng, nb) * 0.8)
    # Real equity gammas: 15 %, 0 % with bucket 11, 75 % between 12 and 13.
    ids = [1, 2, 11, 12, 13]
    gam = np.array([[0.0 if b == c else eq_gamma(b, c) for c in ids] for b in ids])
    buckets = {
        b: BucketInput(tuple(float(x) for x in rng.normal(0.0, 50.0, 3)), _corr(rng, 3), b == 11) for b in ids
    }
    _agg_case(lines, buckets, gam)
    # Asymmetric gamma and a bucket with no sensitivities.
    gam = np.array([[0.0, 0.3, 0.1], [0.5, 0.0, 0.2], [0.25, 0.4, 0.0]])
    _agg_case(
        lines,
        {
            2: BucketInput((30.0, -12.5), np.array([[1.0, 0.4], [0.4, 1.0]])),
            5: BucketInput((), np.zeros((0, 0))),
            9: BucketInput((-7.25,), np.eye(1)),
        },
        gam,
    )
    # Hedged buckets: the plain radicand is negative, so the clipped-S
    # fallback of MAR21.4(5)(b) must run.
    r = np.array([[1.0, -0.9], [-0.9, 1.0]])
    hedged = {1: BucketInput((10.0, 10.0), r), 2: BucketInput((-10.0, -10.0), r)}
    k = within_bucket((10.0, 10.0), r)
    assert 2.0 * k * k + 2.0 * 0.9 * 20.0 * -20.0 < 0.0
    _agg_case(lines, hedged, np.array([[0.0, 0.9], [0.9, 0.0]]))


def _curv_case(
    lines: list[str],
    buckets: dict[int, list[tuple[str, float, float]]],
    rho: dict[int, tuple[list[str], np.ndarray]],
    gam: np.ndarray,
    other: frozenset[int] = frozenset(),
) -> dict[str, float]:
    ids = sorted(buckets)

    def delta_rho(b: int, k: object, l: object) -> float:
        keys, m = rho[b]
        return float(m[keys.index(k), keys.index(l)])

    res = curvature_charge(
        buckets, delta_rho, lambda b, c: float(gam[ids.index(b), ids.index(c)]), other=other
    )
    lines.append(f"CURV {len(ids)}")
    for b in ids:
        lines.append(f"CB {b} {int(b in other)} {len(buckets[b])}")
        lines.append(" ".join(["CK", *rho[b][0]]))
        lines.append(_row("CRHO", rho[b][1]))
        for key, up, dn in buckets[b]:
            lines.append(f"CR {key} {up!r} {dn!r}")
    lines.append(_row("GAM", gam))
    lines.append(_out(res))
    return res


def _curvature_cases(lines: list[str], rng: np.random.Generator) -> None:
    # Random rows with repeated keys (they are summed), mixed signs, and an
    # "other" bucket.
    for nb in (1, 3, 6):
        ids = [int(b) for b in rng.choice(np.arange(1, 19), size=nb, replace=False)]
        buckets, rho = {}, {}
        for b in ids:
            keys = [f"K{i}" for i in range(int(rng.integers(1, 6)))]
            buckets[b] = [
                (str(rng.choice(keys)), float(rng.normal(0.0, 40.0)), float(rng.normal(0.0, 40.0)))
                for _ in range(len(keys) + 2)
            ]
            rho[b] = (keys, _corr(rng, len(keys)))
        other = frozenset(ids[:1]) if nb > 1 else frozenset()
        _curv_case(lines, buckets, rho, _corr(rng, nb) * 0.7, other)
    # Exact tie K+ == K-: the larger sum picks the side (here the down side,
    # which changes S_b and so the cross-bucket term).
    eye = (["A", "B"], np.eye(2))
    gam = np.array([[0.0, 0.8], [0.8, 0.0]])
    _curv_case(
        lines,
        {1: [("A", 3.0, 3.0), ("B", -2.0, -1.0)], 2: [("A", 4.0, 1.0), ("B", 1.0, 1.0)]},
        {1: eye, 2: eye},
        gam,
    )
    # All-negative CVRs: psi switches the cross terms off.
    corr = (["A", "B"], np.array([[1.0, 0.6], [0.6, 1.0]]))
    _curv_case(
        lines,
        {1: [("A", -3.0, -5.0), ("B", -2.0, -1.0)], 2: [("A", -4.0, -1.0), ("B", -1.5, -1.0)]},
        {1: corr, 2: corr},
        gam,
    )
    # The real CSR path: the fixture case must equal csr_curvature, with bucket
    # 16 as the "other" bucket and real gammas.
    cvrs = [
        Cvr(3, "A", 12.0, 9.0), Cvr(3, "B", -4.0, 6.5), Cvr(3, "A", 1.5, -2.0),
        Cvr(8, "C", 7.0, 11.0), Cvr(8, "D", 2.0, -3.0),
        Cvr(16, "E", 5.0, 1.0), Cvr(16, "F", -8.0, 2.5),
        Cvr(17, "G", 3.0, 4.0), Cvr(17, "H", 6.0, 0.5),
    ]  # fmt: skip
    ids = [3, 8, 16, 17]
    buckets = {b: [(r.name, r.up, r.down) for r in cvrs if r.bucket == b] for b in ids}
    rho = {}
    for b in ids:
        keys = sorted({r.name for r in cvrs if r.bucket == b})
        off = 0.8 if b in (17, 18) else 0.35
        rho[b] = (keys, np.where(np.eye(len(keys)) > 0, 1.0, off))
    gam = np.array([[0.0 if b == c else csr_gamma(b, c) for c in ids] for b in ids])
    res = _curv_case(lines, buckets, rho, gam, frozenset({16}))
    assert res == csr_curvature(cvrs), "fixture case differs from csr_curvature"


def _cvr_cases(lines: list[str], rng: np.random.Generator) -> None:
    for _ in range(6):
        v0, vu, vd = (float(x) for x in rng.normal(100.0, 10.0, 3))
        rw, s = float(rng.uniform(0.01, 0.8)), float(rng.normal(0.0, 20.0))
        p, m = net_cvr(v0, vu, vd, rw, s)
        lines.append(_row("CVR", [v0, vu, vd, rw, s, p, m]))


def _im(tag: str, im: object, kind: str) -> str:
    return " ".join([tag, kind, *(repr(float(x)) for x in np.ravel(im))])


def _col_case(lines: list[str], t: np.ndarray, v: np.ndarray, csa: Csa, kr: str = "S", kp: str = "S") -> None:
    ce = collateralised_exposure(t, v, csa)
    lines.append(f"COL {v.shape[0]} {v.shape[1]}")
    lines.append(_row("T", t))
    lines.append(_row("V", v))
    lines.append(_row("CSA", [csa.threshold_cpty, csa.threshold_bank, csa.mta, csa.mpor, csa.initial]))
    lines.append(_im("IMR", csa.im_received, kr))
    lines.append(_im("IMP", csa.im_posted, kp))
    lines.append(_row("BAL", collateral_balance(v, csa)))
    lines.append(" ".join(["LAG", *(str(int(i)) for i in _lag_index(t, csa.mpor))]))
    lines.append(_row("C", ce.collateral))
    lines.append(_row("E", ce.exposure))
    lines.append(_row("N", ce.negative_exposure))


def _collateral_cases(lines: list[str], rng: np.random.Generator) -> None:
    for mpor, t in [
        (0.25, [0.0, 1.0, 2.0]),
        (10.0 / 250.0, [0.5, 0.54, 1.0, 1.04, 3.0]),
        (0.0, [0.0, 1.0]),
        (5.0, [0.0, 1.0, 2.0]),
        (0.1 + 0.2, [0.3, 0.6, 0.9]),
    ]:
        lines.append(_row("LG", [mpor, *t]))
        lines.append(_row("LGO", lagged_grid(t, mpor)))

    # The docstring path, then every CSA term on a random Brownian book.
    _col_case(lines, np.array([0.0, 1.0, 2.0]), np.array([[0.0, 10.0, 4.0]]), Csa(threshold_cpty=3.0))
    t = lagged_grid(np.linspace(0.0, 2.0, 9), 0.25)
    n = t.size
    steps = rng.normal(0.0, 3.0, (6, n - 1)) * np.sqrt(np.diff(t))[None, :]
    v = np.concatenate([np.zeros((6, 1)), np.cumsum(steps, axis=1)], axis=1) * 5.0
    _col_case(lines, t, v, Csa())
    _col_case(lines, t, v, Csa(threshold_cpty=1.0, threshold_bank=2.0, mta=0.75))
    _col_case(lines, t, v, Csa(threshold_cpty=1.0, threshold_bank=0.5, mta=0.5, mpor=0.25, initial=1.25))
    # MPOR off the grid, and longer than the whole horizon.
    _col_case(lines, t, v, Csa(mpor=0.3, initial=-0.5))
    _col_case(lines, t, v, Csa(mpor=50.0, initial=2.0))
    # Infinite thresholds: the uncollateralised limit.
    _col_case(lines, t, v, Csa(threshold_cpty=INF, threshold_bank=INF))
    # Initial margin in each shape the ports accept.
    _col_case(lines, t, v, Csa(mpor=0.25, im_received=1.5, im_posted=0.75))
    _col_case(lines, t, v, Csa(mpor=0.25, im_received=rng.uniform(0.0, 2.0, n)), "T")
    _col_case(lines, t, v, Csa(im_posted=rng.uniform(0.0, 2.0, (6, 1))), "S", "P")
    full = Csa(mta=0.25, im_received=rng.uniform(0.0, 2.0, (6, n)), im_posted=rng.uniform(0.0, 1.0, (6, n)))
    _col_case(lines, t, v, full, "F", "F")


def main() -> None:
    rng = np.random.default_rng(20261104)
    lines = ["# Written by python/equity/04-fixed-income/tools/gen_frtb_collateral_parity.py"]
    _aggregate_cases(lines, rng)
    _curvature_cases(lines, rng)
    _cvr_cases(lines, rng)
    _collateral_cases(lines, rng)
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
