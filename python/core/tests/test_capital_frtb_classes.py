"""FRTB SBM CSR non-sec, equity, commodity (delta, vega) and curvature for all
classes, against BCBS d457 (January 2019) MAR21.

The tables below are transcribed a second time, directly from the PDF text,
independently of ``frtb_tables``; the digests pin that transcription so any
later edit has to be deliberate. Hand fixtures are written as plain loops
from the paragraph text, not by calling the module's helpers.
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest
from qp_core.capital import frtb_classes as C
from qp_core.capital import frtb_curvature as V
from qp_core.capital import frtb_tables as T
from qp_core.errors import InputError

SCEN = {
    "medium": lambda r: r,
    "high": lambda r: min(1.25 * r, 1.0),
    "low": lambda r: max(2 * r - 1, 0.75 * r),
}

# Table 4 (% ), buckets 1..18.
TABLE4 = [0.5, 1.0, 5.0, 3.0, 3.0, 2.0, 1.5, 2.5, 2.0, 4.0, 12.0, 7.0, 8.5, 5.5, 5.0, 12.0, 1.5, 5.0]
# Table 5 (%), full upper triangle in the printed column order.
T5_COLS = ["1", "2", "3", "4", "5", "6", "7", "8", "16", "17", "18"]
TABLE5 = {
    "1": [75, 10, 20, 25, 20, 15, 10, 0, 45, 45],
    "2": [5, 15, 20, 15, 10, 10, 0, 45, 45],
    "3": [5, 15, 20, 5, 20, 0, 45, 45],
    "4": [20, 25, 5, 5, 0, 45, 45],
    "5": [25, 5, 15, 0, 45, 45],
    "6": [5, 20, 0, 45, 45],
    "7": [5, 0, 45, 45],
    "8": [0, 45, 45],
    "16": [0, 0],
    "17": [75],
}
# Table 10 (% spot), buckets 1..13; repo is spot / 100.
TABLE10 = [55, 60, 45, 55, 30, 35, 40, 50, 70, 50, 70, 15, 25]
# Tables 11 and 12 (%), buckets 1..11.
TABLE11 = [30, 35, 60, 80, 40, 45, 20, 35, 25, 35, 50]
TABLE12 = [55, 95, 40, 80, 60, 65, 55, 45, 15, 40, 15]

_DIGESTS = {
    "table4": "445e0c73021a21ef",
    "table5": "3cd96799f7ebba64",
    "table10": "0a167aa7fddbd604",
    "table11": "93d95a1bd5819609",
    "table12": "e064e7e5016489e8",
}


def _digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def test_transcription_digests() -> None:
    # Any edit to a transcribed table must be deliberate: update the digest
    # after re-reading the named MAR21 table.
    got = {
        "table4": _digest(TABLE4),
        "table5": _digest(TABLE5),
        "table10": _digest(TABLE10),
        "table11": _digest(TABLE11),
        "table12": _digest(TABLE12),
    }
    assert got == _DIGESTS


def test_module_tables_match_transcription() -> None:
    assert [round(100 * T.CSR_RW[b], 6) for b in range(1, 19)] == TABLE4
    assert T.CSR_RW_COVERED_AA == 0.015
    for row, vals in TABLE5.items():
        start = T5_COLS.index(row) + 1
        for col, v in zip(T5_COLS[start:], vals, strict=True):
            assert round(100 * T.CSR_SECTOR_GAMMA[(row, col)], 6) == v
            assert T.CSR_SECTOR_GAMMA[(col, row)] == T.CSR_SECTOR_GAMMA[(row, col)]
    assert len(T.CSR_SECTOR_GAMMA) == 11 * 10
    assert [round(100 * T.EQ_RW_SPOT[b], 6) for b in range(1, 14)] == TABLE10
    assert [round(10000 * T.EQ_RW_REPO[b], 6) for b in range(1, 14)] == TABLE10
    assert [round(100 * T.COM_RW[b], 6) for b in range(1, 12)] == TABLE11
    assert [round(100 * T.COM_RHO[b], 6) for b in range(1, 12)] == TABLE12
    assert T.CSR_TENORS == (0.5, 1.0, 3.0, 5.0, 10.0)
    assert T.EQ_RHO == {
        **dict.fromkeys((1, 2, 3, 4), 0.15),
        **dict.fromkeys((5, 6, 7, 8), 0.25),
        9: 0.075,
        10: 0.125,
        12: 0.80,
        13: 0.80,
    }


def test_footnote_examples() -> None:
    # Footnote 18: 35% x 65% x 99.90% = 22.73%.
    assert round(100 * 0.35 * 0.65 * 0.999, 2) == 22.73
    two = C.csr_delta([C.CsrDelta(4, "APPLE", 5.0, 1.0), C.CsrDelta(4, "GOOGLE", 10.0, 1.0, basis="CDS")])
    ws = 0.03
    assert two["medium"] == pytest.approx(math.sqrt(2 * ws**2 * (1 + 0.35 * 0.65 * 0.999)), rel=1e-14)
    # Footnote 21: 95% x 99.00% x 99.90% = 93.96%.
    assert round(100 * T.COM_RHO[2] * T.COM_RHO_TENOR * T.COM_RHO_BASIS, 2) == 93.96
    # MAR21.101 example: GIRR curvature gamma = 50%^2 = 25%.
    r = V.girr_curvature([V.Cvr("EUR", "", 3.0, 0.0), V.Cvr("USD", "", 4.0, 0.0)])
    assert r["medium"] == pytest.approx(math.sqrt(9 + 16 + 2 * 0.25 * 12), rel=1e-14)


def _quad(ws: list[float], rho: list[list[float]]) -> float:
    return math.sqrt(max(sum(ws[i] * ws[j] * rho[i][j] for i in range(len(ws)) for j in range(len(ws))), 0.0))


def _across(ks: list[float], ss: list[float], g: list[list[float]]) -> float:
    q = sum(k * k for k in ks) + sum(
        g[i][j] * ss[i] * ss[j] for i in range(len(ks)) for j in range(len(ks)) if i != j
    )
    if q < 0:
        ss = [max(-k, min(s, k)) for s, k in zip(ss, ks, strict=True)]
        q = sum(k * k for k in ks) + sum(
            g[i][j] * ss[i] * ss[j] for i in range(len(ks)) for j in range(len(ks)) if i != j
        )
    return math.sqrt(max(q, 0.0))


def test_csr_delta_hand_fixture() -> None:
    sens = [
        C.CsrDelta(3, "BANKA", 5.0, 600.0),
        C.CsrDelta(3, "BANKA", 5.0, 400.0),  # nets to 1,000
        C.CsrDelta(3, "BANKA", 10.0, -500.0, basis="CDS"),
        C.CsrDelta(3, "BANKB", 1.0, 300.0),
        C.CsrDelta(11, "BANKC", 3.0, 200.0),  # HY financials: 12%
        C.CsrDelta(16, "X", 1.0, 100.0),
        C.CsrDelta(16, "Y", 1.0, -100.0),  # other sector: sum |WS|
        C.CsrDelta(17, "CDX", 5.0, 1000.0),
    ]
    b3 = [("BANKA", 5, "B", 0.05 * 1000), ("BANKA", 10, "C", 0.05 * -500), ("BANKB", 1, "B", 0.05 * 300)]
    out = {}
    for sc, up in SCEN.items():

        def r(a, b):  # type: ignore[no-untyped-def]
            return (
                (1 if a[0] == b[0] else 0.35) * (1 if a[1] == b[1] else 0.65) * (1 if a[2] == b[2] else 0.999)
            )

        rho = [[1.0 if i == j else up(r(b3[i], b3[j])) for j in range(3)] for i in range(3)]
        k3 = _quad([x[3] for x in b3], rho)
        k11, k16, k17 = 0.12 * 200, 0.12 * 200, 0.015 * 1000
        ks = [k3, k11, k16, k17]
        ss = [sum(x[3] for x in b3), 24.0, 0.0, 15.0]
        # gamma: (3,11) same sector, diff rating 0.5; (3,16) 0; (3,17) 45%; (11,16) 0;
        # (11,17) 45% (17 is not in 1-15, so no rating factor); (16,17) 0.
        g0: list[list[float]] = [[0, 0.5, 0, 0.45], [0.5, 0, 0, 0.45], [0, 0, 0, 0], [0.45, 0.45, 0, 0]]
        g = [[up(x) if i != j else 0.0 for j, x in enumerate(row)] for i, row in enumerate(g0)]
        out[sc] = _across(ks, ss, g)
    got = C.csr_delta(sens)
    for sc in SCEN:
        assert got[sc] == pytest.approx(out[sc], rel=1e-13), sc
    assert got["charge"] == pytest.approx(max(out.values()), rel=1e-13)


def test_csr_gamma_rules() -> None:
    assert C.csr_gamma(1, 9) == 0.5  # same sector, IG vs HY
    assert C.csr_gamma(1, 2) == 0.75
    assert C.csr_gamma(8, 9) == pytest.approx(0.5 * 0.10)  # covered (IG) vs HY sovereign
    assert C.csr_gamma(17, 18) == 0.75
    assert C.csr_gamma(16, 1) == 0.0
    assert C.csr_gamma(2, 17) == 0.45


def test_csr_covered_bond_footnote17() -> None:
    a = C.csr_delta([C.CsrDelta(8, "CB", 5.0, 1000.0)])["medium"]
    b = C.csr_delta([C.CsrDelta(8, "CB", 5.0, 1000.0, covered_aa=True)])["medium"]
    assert (a, b) == (pytest.approx(25.0), pytest.approx(15.0))


def test_csr_bad_inputs() -> None:
    with pytest.raises(InputError):
        C.csr_delta([C.CsrDelta(19, "X", 5.0, 1.0)])
    with pytest.raises(InputError):
        C.csr_delta([C.CsrDelta(1, "X", 2.0, 1.0)])


def test_csr_vega_hand_fixture() -> None:
    sens = [C.CsrVega(5, "A", 1.0, 100.0), C.CsrVega(5, "B", 3.0, 50.0)]
    rho = 0.35 * math.exp(-0.01 * 2 / 1)
    want = math.sqrt(100**2 + 50**2 + 2 * rho * 5000)
    assert C.csr_vega(sens)["medium"] == pytest.approx(want, rel=1e-14)


def test_equity_delta_hand_fixture() -> None:
    sens = [
        C.EqDelta(5, "ACME", 1000.0),
        C.EqDelta(5, "ACME", -2000.0, kind="REPO"),
        C.EqDelta(5, "BETA", -400.0),
        C.EqDelta(9, "SMALL", 100.0),
        C.EqDelta(11, "ODD", 50.0),
        C.EqDelta(11, "ODD2", -50.0),
        C.EqDelta(12, "SPX", 2000.0),
        C.EqDelta(13, "EM", -1000.0),
    ]
    # bucket 5: rows ACME spot, ACME repo, BETA spot (sorted name, kind: REPO < SPOT irrelevant)
    w5 = [0.30 * 1000, 0.003 * -2000, 0.30 * -400]
    r5 = [[1, 0.999, 0.25], [0.999, 1, 0.25 * 0.999], [0.25, 0.25 * 0.999, 1]]
    out = {}
    for sc, up in SCEN.items():
        rho = [[1.0 if i == j else up(r5[i][j]) for j in range(3)] for i in range(3)]
        ks = [_quad(w5, rho), 70.0, 0.70 * 100, 0.15 * 2000, 0.25 * 1000]
        ss = [sum(w5), 70.0, 0.0, 300.0, -250.0]
        g0: list[list[float]] = [
            [0, 0.15, 0, 0.45, 0.45],
            [0.15, 0, 0, 0.45, 0.45],
            [0, 0, 0, 0, 0],
            [0.45, 0.45, 0, 0, 0.75],
            [0.45, 0.45, 0, 0.75, 0],
        ]
        g = [[up(x) if i != j else 0.0 for j, x in enumerate(row)] for i, row in enumerate(g0)]
        out[sc] = _across(ks, ss, g)
    got = C.eq_delta(sens)
    for sc in SCEN:
        assert got[sc] == pytest.approx(out[sc], rel=1e-13), sc


def test_equity_vega() -> None:
    sens = [C.EqVega(6, "A", 0.5, 200.0), C.EqVega(6, "B", 2.0, 100.0), C.EqVega(10, "S", 1.0, 30.0)]
    rw = min(0.55 * math.sqrt(20 / 10), 1.0)
    rho = 0.25 * math.exp(-0.01 * 1.5 / 0.5)
    k6 = rw * math.sqrt(200**2 + 100**2 + 2 * rho * 200 * 100)
    k10 = 30.0
    want = math.sqrt(k6**2 + k10**2 + 2 * 0.15 * rw * 300 * 30)
    assert C.eq_vega(sens)["medium"] == pytest.approx(want, rel=1e-14)
    assert round(100 * rw, 2) == 77.78


def test_commodity_delta_hand_fixture() -> None:
    sens = [
        C.ComDelta(2, "BRENT", 1.0, 1000.0, "LEHAVRE"),
        C.ComDelta(2, "WTI", 5.0, -800.0, "OKLAHOMA"),
        C.ComDelta(2, "WTI", 1.0, 300.0, "OKLAHOMA"),
        C.ComDelta(7, "GOLD", 0.5, 2000.0),
        C.ComDelta(11, "POTASH", 1.0, 100.0),
    ]
    rows = [("BRENT", 1, "L", 350.0), ("WTI", 1, "O", 0.35 * 300), ("WTI", 5, "O", 0.35 * -800)]

    def r(a, b):  # type: ignore[no-untyped-def]
        return (1 if a[0] == b[0] else 0.95) * (1 if a[1] == b[1] else 0.99) * (1 if a[2] == b[2] else 0.999)

    out = {}
    for sc, up in SCEN.items():
        rho = [[1.0 if i == j else up(r(rows[i], rows[j])) for j in range(3)] for i in range(3)]
        ks = [_quad([x[3] for x in rows], rho), 400.0, 50.0]
        ss = [sum(x[3] for x in rows), 400.0, 50.0]
        g = [[0, up(0.2), 0], [up(0.2), 0, 0], [0, 0, 0]]
        out[sc] = _across(ks, ss, g)
    got = C.com_delta(sens)
    for sc in SCEN:
        assert got[sc] == pytest.approx(out[sc], rel=1e-13), sc


def test_commodity_vega() -> None:
    sens = [C.ComVega(6, "NG", 1.0, 100.0, "HH"), C.ComVega(6, "LNG", 2.0, -60.0, "JKM")]
    rho = 0.65 * 0.999 * math.exp(-0.01)
    want = math.sqrt(100**2 + 60**2 - 2 * rho * 6000)
    assert C.com_vega(sens)["medium"] == pytest.approx(want, rel=1e-14)


# -------------------------------------------------------------- curvature


def _bs_call(s: float, k: float, t: float, sig: float) -> float:
    n = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    d1 = (math.log(s / k) + 0.5 * sig * sig * t) / (sig * math.sqrt(t))
    return s * n(d1) - k * n(d1 - sig * math.sqrt(t))


def test_net_cvr_linear_is_zero_and_option_positive() -> None:
    # A forward (linear): CVR = 0 both ways.
    assert V.net_cvr(0.0, 30.0, -30.0, 0.30, 100.0) == (0.0, 0.0)
    # Long call, S=K=100, T=1, vol 20%, bucket 5 (RW 30%). Delta sensitivity per
    # MAR21.21: (V(1.01 S) - V(S)) / 0.01.
    s0, rw = 100.0, 0.30
    v0 = _bs_call(s0, 100, 1, 0.2)
    s_k = (_bs_call(1.01 * s0, 100, 1, 0.2) - v0) / 0.01
    up, dn = V.net_cvr(v0, _bs_call(s0 * 1.3, 100, 1, 0.2), _bs_call(s0 * 0.7, 100, 1, 0.2), rw, s_k)
    # Convexity: a long option loses less than delta predicts on the way down and
    # gains more on the way up, so both CVRs are negative (no curvature charge).
    assert up < 0 and dn < 0
    r = V.eq_curvature([V.Cvr(5, "ACME", up, dn)])
    assert r["charge"] == 0.0
    # The short call is the one that pays: CVR = -(long CVR) > 0 both ways.
    r = V.eq_curvature([V.Cvr(5, "ACME", -up, -dn)])
    assert r["charge"] == pytest.approx(max(-up, -dn), rel=1e-14)
    assert V.curvature_rw("EQ", 5) == rw


def test_curvature_within_bucket_psi_and_selection() -> None:
    # Bucket 5 equities A, B, C: CVR+ = (10, -4, -6), CVR- = (3, 8, -2).
    rows = [V.Cvr(5, "A", 10.0, 3.0), V.Cvr(5, "B", -4.0, 8.0), V.Cvr(5, "C", -6.0, -2.0)]
    out = {}
    for sc, up in SCEN.items():
        rho = up(0.25**2)

        def k(c: list[float], rho: float = rho) -> float:
            q = sum(max(x, 0) ** 2 for x in c)
            for i in range(3):
                for j in range(3):
                    if i != j and not (c[i] < 0 and c[j] < 0):
                        q += rho * c[i] * c[j]
            return math.sqrt(max(q, 0))

        out[sc] = max(k([10, -4, -6]), k([3, 8, -2]))
    got = V.eq_curvature(rows)
    for sc in SCEN:
        assert got[sc] == pytest.approx(out[sc], rel=1e-14)


def test_curvature_across_buckets_uses_selected_side() -> None:
    # Two equity buckets (1, 6): b1 selects up (S=+20), b6 selects down (S=-9),
    # gamma = 15%^2; psi = 1 (not both negative).
    rows = [V.Cvr(1, "A", 20.0, 5.0), V.Cvr(6, "B", -1.0, -9.0), V.Cvr(6, "C", 3.0, 0.0)]
    got = V.eq_curvature(rows)
    for sc, up in SCEN.items():
        r = up(0.0625)

        def q6(c: list[float], r: float = r) -> float:
            q = sum(max(x, 0) ** 2 for x in c)
            q += sum(
                r * c[i] * c[j] for i in range(2) for j in range(2) if i != j and not (c[i] < 0 and c[j] < 0)
            )
            return math.sqrt(max(q, 0))

        kp6, km6 = q6([-1.0, 3.0]), q6([-9.0, 0.0])
        sel_up = kp6 > km6 or (kp6 == km6 and 2.0 > -9.0)
        k6, w6 = (kp6, 2.0) if sel_up else (km6, -9.0)
        g = up(0.15**2)
        want = math.sqrt(max(400 + k6**2 + 2 * g * 20 * w6 * (0 if (w6 < 0 and 20 < 0) else 1), 0))
        assert got[sc] == pytest.approx(want, rel=1e-14), sc


def test_curvature_two_negative_buckets_do_not_offset() -> None:
    # Both W_b negative: psi = 0, so the cross term vanishes.
    rows = [V.Cvr(5, "A", -3.0, -4.0), V.Cvr(6, "B", 2.0, -5.0)]
    got = V.eq_curvature(rows)
    # b5: K+ = K- = 0, tie broken by sums: -3 > -4 so up selected, W=-3.
    # b6: K+ = 2, K- = 0, up selected, W=+2. psi(-3, 2) = 1.
    assert got["medium"] == pytest.approx(math.sqrt(4 + 2 * 0.0225 * -3 * 2), rel=1e-14)
    rows = [V.Cvr(5, "A", -3.0, -4.0), V.Cvr(6, "B", -1.0, -5.0)]
    assert V.eq_curvature(rows)["medium"] == 0.0


def test_curvature_other_buckets() -> None:
    rows = [V.Cvr(16, "X", 5.0, 1.0), V.Cvr(16, "Y", 4.0, 7.0), V.Cvr(16, "Z", -3.0, 2.0)]
    assert V.csr_curvature(rows)["charge"] == pytest.approx(max(9.0, 10.0))
    rows = [V.Cvr(11, "X", 5.0, 1.0), V.Cvr(11, "Y", -5.0, 0.0)]
    assert V.eq_curvature(rows)["charge"] == pytest.approx(5.0)


def test_csr_curvature_name_only_correlation() -> None:
    rows = [V.Cvr(4, "A", 10.0, 0.0), V.Cvr(4, "B", 10.0, 0.0), V.Cvr(17, "IDX", 6.0, 0.0)]
    k4 = math.sqrt(200 + 2 * 0.35**2 * 100)
    want = math.sqrt(k4**2 + 36 + 2 * 0.45**2 * 20 * 6)
    assert V.csr_curvature(rows)["medium"] == pytest.approx(want, rel=1e-14)


def test_commodity_curvature() -> None:
    rows = [V.Cvr(2, "BRENT", 10.0, 0.0, "LH"), V.Cvr(2, "WTI", 10.0, 0.0, "OK"), V.Cvr(7, "GOLD", 4.0, 0.0)]
    r = (0.95 * 0.999) ** 2
    k2 = math.sqrt(200 + 2 * r * 100)
    want = math.sqrt(k2**2 + 16 + 2 * 0.04 * 20 * 4)
    assert V.com_curvature(rows)["medium"] == pytest.approx(want, rel=1e-14)


def test_fx_curvature_scalar_and_gamma() -> None:
    rows = [V.Cvr("EURUSD", "", 15.0, 0.0), V.Cvr("EURJPY", "", 9.0, 0.0, fx_scalar=True)]
    want = math.sqrt(225 + 36 + 2 * 0.36 * 15 * 6)
    assert V.fx_curvature(rows)["medium"] == pytest.approx(want, rel=1e-14)


def test_curvature_rw_table() -> None:
    assert V.curvature_rw("GIRR") == 0.017
    assert V.curvature_rw("FX") == 0.15
    assert V.curvature_rw("FX", fx_relief=True) == pytest.approx(0.15 / math.sqrt(2))
    assert V.curvature_rw("CSR", 16) == 0.12
    assert V.curvature_rw("COM", 3) == 0.60
    with pytest.raises(InputError):
        V.curvature_rw("CSR", 99)
