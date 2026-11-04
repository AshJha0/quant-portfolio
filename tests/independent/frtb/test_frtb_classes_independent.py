"""Compare frtb_classes and frtb_curvature with the independent MAR21 oracle.

250 seeded random portfolios per measure, every correlation scenario plus the
final charge, to 1e-10 relative. Portfolios mix signs, repeat risk factors so
netting runs, and draw every bucket including the other-sector buckets.
``test_branches_are_reached`` proves the draws reach the rarer rules.

Open disagreement, kept as strict xfails so that either side changing shows
up: MAR21.13(2)-(3), the MAR21.94 FAQ and MAR21.100 leave delivery location
out of commodity vega and curvature risk factors; qp_core keeps it. The
comparisons below therefore hold location constant, and the two
``*_location_disagreement`` tests vary it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import numpy as np
import pytest
from qp_core.capital import frtb_classes as impl
from qp_core.capital import frtb_curvature as curv

from . import mar21_oracle_classes as ref

N = 250
KEYS = ("medium", "high", "low", "charge")
CSR_T = (0.5, 1.0, 3.0, 5.0, 10.0)
COM_T = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0)
NAMES = ("A", "B", "C", "D", "E")
LOCS = ("", "ROTTERDAM", "CUSHING")
LOCATION_DISAGREEMENT = (
    "qp_core treats location as a commodity vega and curvature dimension; "
    "MAR21.13(2)-(3), the MAR21.94 FAQ and MAR21.100 do not"
)
CCYS = ("USD", "EUR", "GBP", "JPY", "CHF")
PAIRS = ("EURUSD", "USDJPY", "GBPUSD", "USDTRY", "USDTHB")


def _close(a: dict[str, float], b: dict[str, float]) -> None:
    for k in KEYS:
        assert a[k] == pytest.approx(b[k], rel=1e-10, abs=1e-12), (k, a, b)


def _amt(r: random.Random) -> float:
    return r.choice((-1, 1)) * 10 ** r.uniform(2, 7)


def _pick(r: random.Random, n: int) -> list[int]:
    return r.sample(range(1, n + 1), r.randint(1, min(n, 6)))


def _csr_delta_rows(r: random.Random) -> list[tuple]:
    bs = _pick(r, 18)
    covered = r.random() < 0.5
    rows = []
    for _ in range(r.randint(1, 40)):
        b, name = r.choice(bs), r.choice(NAMES)
        flag = covered and b == 8 and name in ("A", "B")
        rows.append((b, name, r.choice(CSR_T), _amt(r), r.choice(("BOND", "CDS")), flag))
    return rows


def _vega_rows(r: random.Random, n: int) -> list[tuple]:
    bs = _pick(r, n)
    return [(r.choice(bs), r.choice(NAMES), r.choice(CSR_T), _amt(r)) for _ in range(r.randint(1, 40))]


def _eq_delta_rows(r: random.Random) -> list[tuple]:
    bs = _pick(r, 13)
    return [
        (r.choice(bs), r.choice(NAMES), _amt(r), r.choice(("SPOT", "REPO"))) for _ in range(r.randint(1, 40))
    ]


def _com_rows(r: random.Random, tenors: tuple[float, ...]) -> list[tuple]:
    bs = _pick(r, 11)
    return [
        (r.choice(bs), r.choice(NAMES), r.choice(tenors), _amt(r), r.choice(LOCS))
        for _ in range(r.randint(1, 40))
    ]


def _cvr_pair(r: random.Random) -> tuple[float, float]:
    """A quarter of the rows are loss-free in both directions."""
    if r.random() < 0.25:
        return -abs(_amt(r)), -abs(_amt(r))
    return _amt(r), _amt(r)


def _cvr_rows(r: random.Random, buckets: list) -> list[tuple]:
    return [(r.choice(buckets), r.choice(NAMES), *_cvr_pair(r)) for _ in range(r.randint(1, 30))]


@pytest.mark.parametrize("seed", range(N))
def test_csr_delta(seed: int) -> None:
    rows = _csr_delta_rows(random.Random(10000 + seed))
    _close(impl.csr_delta([impl.CsrDelta(*x) for x in rows]), ref.csr_delta(rows))


@pytest.mark.parametrize("seed", range(N))
def test_csr_vega(seed: int) -> None:
    rows = _vega_rows(random.Random(11000 + seed), 18)
    _close(impl.csr_vega([impl.CsrVega(*x) for x in rows]), ref.csr_vega(rows))


@pytest.mark.parametrize("seed", range(N))
def test_eq_delta(seed: int) -> None:
    rows = _eq_delta_rows(random.Random(12000 + seed))
    _close(impl.eq_delta([impl.EqDelta(*x) for x in rows]), ref.eq_delta(rows))


@pytest.mark.parametrize("seed", range(N))
def test_eq_vega(seed: int) -> None:
    rows = _vega_rows(random.Random(13000 + seed), 13)
    _close(impl.eq_vega([impl.EqVega(*x) for x in rows]), ref.eq_vega(rows))


@pytest.mark.parametrize("seed", range(N))
def test_com_delta(seed: int) -> None:
    rows = _com_rows(random.Random(14000 + seed), COM_T)
    _close(impl.com_delta([impl.ComDelta(*x) for x in rows]), ref.com_delta(rows))


@pytest.mark.parametrize("seed", range(N))
def test_com_vega(seed: int) -> None:
    rows = [(*x[:4], "") for x in _com_rows(random.Random(15000 + seed), CSR_T)]
    _close(impl.com_vega([impl.ComVega(*x) for x in rows]), ref.com_vega(rows))


@pytest.mark.xfail(strict=True, reason=LOCATION_DISAGREEMENT)
def test_com_vega_location_disagreement() -> None:
    """Same commodity and maturity, two locations: one factor, so it nets to 0."""
    rows = [(1, "A", 1.0, 100.0, "X"), (1, "A", 1.0, -100.0, "Y")]
    _close(impl.com_vega([impl.ComVega(*x) for x in rows]), ref.com_vega(rows))


@pytest.mark.parametrize("seed", range(N))
def test_girr_curvature(seed: int) -> None:
    rows = _cvr_rows(random.Random(16000 + seed), list(CCYS))
    _close(curv.girr_curvature([curv.Cvr(*x) for x in rows]), ref.girr_curvature(rows))


@pytest.mark.parametrize("seed", range(N))
def test_fx_curvature(seed: int) -> None:
    r = random.Random(17000 + seed)
    rows = [(*x, r.random() < 0.5) for x in _cvr_rows(r, list(PAIRS))]
    got = curv.fx_curvature([curv.Cvr(b, n, u, d, "", f) for b, n, u, d, f in rows])
    _close(got, ref.fx_curvature(rows))


@pytest.mark.parametrize("seed", range(N))
def test_csr_curvature(seed: int) -> None:
    r = random.Random(18000 + seed)
    rows = _cvr_rows(r, _pick(r, 18))
    _close(curv.csr_curvature([curv.Cvr(*x) for x in rows]), ref.csr_curvature(rows))


@pytest.mark.parametrize("seed", range(N))
def test_eq_curvature(seed: int) -> None:
    r = random.Random(19000 + seed)
    rows = _cvr_rows(r, _pick(r, 13))
    _close(curv.eq_curvature([curv.Cvr(*x) for x in rows]), ref.eq_curvature(rows))


@pytest.mark.parametrize("seed", range(N))
def test_com_curvature(seed: int) -> None:
    r = random.Random(20000 + seed)
    rows = _cvr_rows(r, _pick(r, 11))
    _close(curv.com_curvature([curv.Cvr(*x) for x in rows]), ref.com_curvature(rows))


@pytest.mark.xfail(strict=True, reason=LOCATION_DISAGREEMENT)
def test_com_curvature_location_disagreement() -> None:
    """MAR21.13(3): one curve per commodity, so 10 and -4 net to 6."""
    got = curv.com_curvature([curv.Cvr(1, "A", 10.0, 0.0, "X"), curv.Cvr(1, "A", -4.0, 0.0, "Y")])
    _close(got, ref.com_curvature([(1, "A", 10.0, 0.0), (1, "A", -4.0, 0.0)]))


def _generic_case(seed: int) -> tuple:
    r = random.Random(22000 + seed)
    nb = r.randint(1, 5)
    rho, gam = r.uniform(0.0, 1.0), r.uniform(0.0, 1.0)
    buckets = {b: {n: _cvr_pair(r) for n in r.sample(NAMES, r.randint(1, 5))} for b in range(1, nb + 1)}
    other = frozenset({1}) if r.random() < 0.5 else frozenset()

    def rho_f(_b: object, x: object, y: object) -> float:
        return 1.0 if x == y else rho

    def gam_f(b: object, c: object) -> float:
        return 1.0 if b == c else gam

    return buckets, rho_f, gam_f, other


def _tie_case(seed: int) -> tuple:
    r = random.Random(23000 + seed)
    buckets = {
        b: {n: (-float(r.randint(1, 9)), -float(r.randint(1, 9))) for n in NAMES[:3]} for b in (1, 2, 3)
    }
    buckets[4] = {"A": (float(r.randint(1, 9)), float(r.randint(1, 9)))}

    def rho_f(_b: object, _x: object, _y: object) -> float:
        return 0.5

    def gam_f(b: object, c: object) -> float:
        return 1.0 if b == c else 0.6

    return buckets, rho_f, gam_f, frozenset()


def _arg(buckets: dict) -> dict:
    return {b: [(n, u, d) for n, (u, d) in f.items()] for b, f in buckets.items()}


@pytest.mark.parametrize("seed", range(N))
def test_curvature_charge_generic(seed: int) -> None:
    """The MAR21.5 engine with arbitrary delta parameters and an other bucket."""
    buckets, rho_f, gam_f, other = _generic_case(seed)
    got = curv.curvature_charge(_arg(buckets), rho_f, gam_f, other=other)
    _close(got, ref.curvature(buckets, rho_f, gam_f, other))


@pytest.mark.parametrize("seed", range(N))
def test_tie_rule(seed: int) -> None:
    """MAR21.5(3)(a)(iii): K+ = K- = 0, so the sums alone pick S_b."""
    buckets, rho_f, gam_f, other = _tie_case(seed)
    got = curv.curvature_charge(_arg(buckets), rho_f, gam_f, other=other)
    _close(got, ref.curvature(buckets, rho_f, gam_f, other))


@pytest.mark.parametrize("seed", range(N))
def test_aggregate_generic(seed: int) -> None:
    """MAR21.4(4)-(5) on arbitrary inputs, with and without an other bucket."""
    r = random.Random(24000 + seed)
    nb = r.randint(2, 5)
    gam = r.uniform(0.0, 1.0)
    raw: ref.Buckets = {}
    for b in range(1, nb + 1):
        n = r.randint(1, 4)
        rho = r.uniform(0.0, 1.0)
        raw[b] = (
            [_amt(r) for _ in range(n)],
            [[1.0 if i == j else rho for j in range(n)] for i in range(n)],
        )
    other = {1} if r.random() < 0.5 else set()
    arg = {b: impl.BucketInput(tuple(ws), np.array(rho), b in other) for b, (ws, rho) in raw.items()}

    def gam_f(b: int, c: int) -> float:
        return 1.0 if b == c else gam

    _close(impl.aggregate(arg, gam_f), ref.aggregate(raw, gam_f, other))


def _clipped_eq(seed: int) -> list[tuple]:
    """Many same-sized names, opposite signs, in the two low-rho small-cap buckets."""
    r = random.Random(26000 + seed)
    size = 10 ** r.uniform(2, 6)
    rows = []
    for i in range(r.randint(30, 60)):
        rows.append((9, f"N{i}", size * r.uniform(0.9, 1.1), "SPOT"))
        rows.append((10, f"M{i}", -size * r.uniform(1.2, 1.5), "SPOT"))
    return rows


@pytest.mark.parametrize("seed", range(N))
def test_eq_delta_clipped_branch(seed: int) -> None:
    """MAR21.4(5)(b) on real equity parameters, not synthetic K_b and S_b."""
    rows = _clipped_eq(seed)
    _close(impl.eq_delta([impl.EqDelta(*x) for x in rows]), ref.eq_delta(rows))


@pytest.mark.parametrize("seed", range(N))
def test_net_cvr(seed: int) -> None:
    r = random.Random(25000 + seed)
    args = (_amt(r), _amt(r), _amt(r), r.uniform(0.0, 1.0), _amt(r))
    assert curv.net_cvr(*args) == pytest.approx(ref.net_cvr(*args), rel=1e-12)


@pytest.mark.parametrize(
    ("got", "want", "n"),
    [
        (impl.csr_gamma, ref.csr_gamma, 18),
        (impl.eq_gamma, ref.eq_gamma, 13),
        (impl.com_gamma, ref.com_gamma, 11),
    ],
    ids=["csr", "eq", "com"],
)
def test_gamma_tables_every_pair(
    got: Callable[[int, int], float], want: Callable[[int, int], float], n: int
) -> None:
    for b in range(1, n + 1):
        for c in range(1, n + 1):
            if b != c:  # MAR21.4(5) uses gamma only between different buckets
                assert got(b, c) == pytest.approx(want(b, c), abs=1e-14), (b, c)


def test_risk_weights_every_bucket() -> None:
    """One unit of sensitivity isolates each transcribed weight."""
    for b, rw in ref.CSR_RW.items():
        got = impl.csr_delta([impl.CsrDelta(b, "A", 1.0, 1.0)])["charge"]
        assert got == pytest.approx(rw, rel=1e-12), b
        assert curv.curvature_rw("CSR", b) == pytest.approx(rw, rel=1e-12), b
    for b, rw in ref.EQ_RW_SPOT.items():
        assert impl.eq_delta([impl.EqDelta(b, "A", 1.0)])["charge"] == pytest.approx(rw, rel=1e-12)
        repo = impl.eq_delta([impl.EqDelta(b, "A", 1.0, "REPO")])["charge"]
        assert repo == pytest.approx(ref.EQ_RW_REPO[b], rel=1e-12), b
        assert curv.curvature_rw("EQ", b) == pytest.approx(rw, rel=1e-12), b
        assert impl.eq_vega_rw(b) == pytest.approx(ref.eq_vega_rw(b), rel=1e-12), b
    for b, rw in ref.COM_RW.items():
        got = impl.com_delta([impl.ComDelta(b, "A", 1.0, 1.0)])["charge"]
        assert got == pytest.approx(rw, rel=1e-12), b
        assert curv.curvature_rw("COM", b) == pytest.approx(rw, rel=1e-12), b
    assert curv.curvature_rw("GIRR") == pytest.approx(ref.GIRR_CURVATURE_RW)
    assert curv.curvature_rw("FX") == pytest.approx(ref.FX_CURVATURE_RW)
    assert curv.curvature_rw("FX", fx_relief=True) == pytest.approx(0.15 / math.sqrt(2.0))
    assert curv.FX_CURVATURE_SCALAR == ref.FX_SCALAR


def test_covered_bond_weight() -> None:
    """MAR21.53 footnote 19."""
    got = impl.csr_delta([impl.CsrDelta(8, "A", 1.0, 1.0, "BOND", True)])["charge"]
    assert got == pytest.approx(ref.CSR_COVERED_AA_RW, rel=1e-12)


def test_hand_values() -> None:
    """Oracle checks worked by hand from the text, independent of qp_core."""
    # MAR21.56: bucket 16 is the sum of absolute weighted sensitivities.
    rows = [(16, "A", 1.0, 100.0, "BOND", False), (16, "B", 1.0, -100.0, "BOND", False)]
    assert ref.csr_delta(rows)["medium"] == pytest.approx(24.0)
    # MAR21.57: sectors 1 and 2, different rating: 0.5 * 0.75.
    assert ref.csr_gamma(1, 10) == pytest.approx(0.375)
    # MAR21.5(3): a loss-free factor gives K = 0.
    one = ref.curvature({1: {"A": (-1.0, -2.0)}}, lambda *_: 1.0, lambda *_: 1.0)
    assert one["medium"] == 0.0
    # MAR21.101: gamma 0.5 squared = 0.25, so 1 + 1 + 2 * 0.25.
    got = ref.girr_curvature([("USD", "", 1.0, 0.0), ("EUR", "", 1.0, 0.0)])
    assert got["medium"] == pytest.approx(math.sqrt(2.5))


def test_branches_are_reached() -> None:
    """The seeded draws must reach the rarer rules, or the comparison proves less."""
    ref.TRACE.clear()
    for seed in range(N):
        ref.eq_delta(_eq_delta_rows(random.Random(12000 + seed)))
        ref.com_delta(_com_rows(random.Random(14000 + seed), COM_T))
        ref.csr_delta(_csr_delta_rows(random.Random(10000 + seed)))
        ref.eq_delta(_clipped_eq(seed))
        for case in (_generic_case(seed), _tie_case(seed)):
            ref.curvature(*case)
    for name in ("delta_clipped", "curv_down", "curv_tie_up", "curv_tie_down", "curv_floor"):
        assert ref.TRACE[name] > 0, (name, ref.TRACE)
