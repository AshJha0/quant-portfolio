"""FRTB SBM (GIRR, FX; delta and vega) against MAR21 text and hand fixtures.

Table 2 below is copied entry by entry from BCBS d457 (January 2019),
MAR21.46; the footnote examples from footnotes 13 and 14. The hand
fixtures are computed in this file with plain loops written from the MAR21
text, independently of the vectorised module code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qp_core.capital import frtb_sbm as f
from qp_core.errors import InputError

TABLE2 = [
    [100.0, 97.0, 91.4, 81.1, 71.9, 56.6, 40.0, 40.0, 40.0, 40.0],
    [97.0, 100.0, 97.0, 91.4, 86.1, 76.3, 56.6, 41.9, 40.0, 40.0],
    [91.4, 97.0, 100.0, 97.0, 94.2, 88.7, 76.3, 65.7, 56.6, 41.9],
    [81.1, 91.4, 97.0, 100.0, 98.5, 95.6, 88.7, 82.3, 76.3, 65.7],
    [71.9, 86.1, 94.2, 98.5, 100.0, 98.0, 93.2, 88.7, 84.4, 76.3],
    [56.6, 76.3, 88.7, 95.6, 98.0, 100.0, 97.0, 94.2, 91.4, 86.1],
    [40.0, 56.6, 76.3, 88.7, 93.2, 97.0, 100.0, 98.5, 97.0, 94.2],
    [40.0, 41.9, 65.7, 82.3, 88.7, 94.2, 98.5, 100.0, 99.0, 97.0],
    [40.0, 40.0, 56.6, 76.3, 84.4, 91.4, 97.0, 99.0, 100.0, 98.5],
    [40.0, 40.0, 41.9, 65.7, 76.3, 86.1, 94.2, 97.0, 98.5, 100.0],
]


def test_table1_risk_weights() -> None:
    assert f.GIRR_TENORS == (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0)
    assert f.GIRR_RW == (0.017, 0.017, 0.016, 0.013, 0.012, 0.011, 0.011, 0.011, 0.011, 0.011)
    assert f.GIRR_RW_INFLATION_XCCY == 0.016
    assert f.FX_RW == 0.15
    assert len(f.FX_SPECIFIED_PAIRS) == 19
    assert f.SPECIFIED_CURRENCIES == {"EUR", "USD", "GBP", "AUD", "JPY", "SEK", "CAD"}


def test_table2_every_entry() -> None:
    for i, ti in enumerate(f.GIRR_TENORS):
        for j, tj in enumerate(f.GIRR_TENORS):
            assert round(100 * f.girr_delta_rho(ti, tj), 1) == TABLE2[i][j], (ti, tj)


def test_footnote_examples() -> None:
    assert round(100 * f.girr_delta_rho(1.0, 5.0), 2) == 88.69
    assert round(100 * f.girr_delta_rho(1.0, 5.0, same_curve=False), 2) == 88.60


def test_table13_vega_risk_weights() -> None:
    assert f.vega_rw("GIRR") == 1.0
    assert f.vega_rw("FX") == 1.0
    assert round(100 * f.vega_rw("EQ_LARGE"), 2) == 77.78


def test_scenarios() -> None:
    assert f.scenario_corr(0.5, "high") == 0.625
    assert f.scenario_corr(0.5, "low") == 0.375  # max(0, 0.375)
    assert f.scenario_corr(0.9, "low") == pytest.approx(0.8)
    assert f.scenario_corr(0.9, "high") == 1.0
    with pytest.raises(InputError):
        f.scenario_corr(0.5, "stress")


def _hand_girr() -> dict[str, float]:
    """USD (specified, sqrt 2 relief): OIS 2y +5,000, OIS 10y -3,000,
    SOFR3M 10y +2,000, inflation +1,000. BRL: 1y +4,000. Plain loops."""
    usd = [
        ("OIS", 2.0, 5000.0, 0.013),
        ("OIS", 10.0, -3000.0, 0.011),
        ("SOFR3M", 10.0, 2000.0, 0.011),
        ("INFLATION", 0.0, 1000.0, 0.016),
    ]
    ws_usd = [rw / math.sqrt(2) * s for _, _, s, rw in usd]
    brl_ws = 0.016 * 4000.0
    out = {}
    for sc, up in (
        ("medium", lambda r: r),
        ("high", lambda r: min(1.25 * r, 1.0)),
        ("low", lambda r: max(2 * r - 1, 0.75 * r)),
    ):

        def rho(a: tuple, b: tuple) -> float:
            if a[0] == "INFLATION" or b[0] == "INFLATION":
                return 0.4
            base = max(math.exp(-0.03 * abs(a[1] - b[1]) / min(a[1], b[1])), 0.4)
            return base if a[0] == b[0] else 0.999 * base

        q = 0.0
        for i in range(4):
            for j in range(4):
                q += ws_usd[i] * ws_usd[j] * (1.0 if i == j else up(rho(usd[i], usd[j])))
        k_usd = math.sqrt(max(q, 0.0))
        k_brl = abs(brl_ws)
        s_usd, s_brl = sum(ws_usd), brl_ws
        g = up(0.5)
        out[sc] = math.sqrt(k_usd**2 + k_brl**2 + 2 * g * s_usd * s_brl)
    return out


def test_girr_delta_hand_fixture() -> None:
    sens = [
        f.GirrDelta("USD", "OIS", 2.0, 3000.0),
        f.GirrDelta("USD", "OIS", 2.0, 2000.0),  # nets with the line above
        f.GirrDelta("USD", "OIS", 10.0, -3000.0),
        f.GirrDelta("USD", "SOFR3M", 10.0, 2000.0),
        f.GirrDelta("USD", "INFLATION", 0.0, 1000.0),
        f.GirrDelta("BRL", "CDI", 1.0, 4000.0),
    ]
    got = f.girr_delta(sens)
    want = _hand_girr()
    for sc in f.SCENARIOS:
        assert got[sc] == pytest.approx(want[sc], rel=1e-13)
    assert got["charge"] == max(want.values())


def test_docstring_example_by_hand() -> None:
    k = math.sqrt(110**2 + 110**2 - 2 * math.exp(-0.03) * 110**2)
    r = f.girr_delta(
        [f.GirrDelta("USD", "OIS", 5.0, 1e4), f.GirrDelta("USD", "OIS", 10.0, -1e4)], specified_relief=False
    )
    assert r["medium"] == pytest.approx(k, rel=1e-14)


def test_xccy_basis_zero_correlation() -> None:
    a = f.GirrDelta("EUR", "XCCY_EURUSD", 0.0, 1000.0)
    b = f.GirrDelta("EUR", "ESTR", 5.0, 1000.0)
    r = f.girr_delta([a, b], specified_relief=False)
    assert r["medium"] == pytest.approx(math.hypot(16.0, 11.0), rel=1e-14)


def test_across_bucket_alternative_specification() -> None:
    # Radicand 49 + 49 - 2 * 0.5 * 100 = -2 < 0, so S is clipped to +-K.
    assert f.across_buckets([7.0, 7.0], [10.0, -10.0], np.full((2, 2), 0.5)) == pytest.approx(7.0)
    assert f.across_buckets([7.0, 7.0], [5.0, -5.0], np.full((2, 2), 0.5)) == pytest.approx(
        math.sqrt(98 - 25)
    )


def test_fx_delta_hand_fixture() -> None:
    sens = [
        f.FxDelta("USDJPY", 1e6),
        f.FxDelta("JPYUSD", -2e5),
        f.FxDelta("EURAUD", 5e5),
        f.FxDelta("USDCHF", -3e5),
    ]
    got = f.fx_delta(sens)
    ws = {"USDJPY": 0.15 / math.sqrt(2) * 8e5, "EURAUD": 0.15 * 5e5, "USDCHF": 0.15 / math.sqrt(2) * -3e5}
    # JPYUSD is its own bucket (pairs are taken as quoted) but gets the specified-pair RW.
    ws_all = [0.15 / math.sqrt(2) * 1e6, 0.15 / math.sqrt(2) * -2e5, ws["EURAUD"], ws["USDCHF"]]
    for sc, g in (("medium", 0.6), ("high", 0.75), ("low", 0.45)):
        q = sum(w * w for w in ws_all) + sum(
            g * a * b for i, a in enumerate(ws_all) for j, b in enumerate(ws_all) if i != j
        )
        assert got[sc] == pytest.approx(math.sqrt(q), rel=1e-13)


def test_girr_vega_hand_fixture() -> None:
    sens = [
        f.GirrVega("EUR", 1.0, 5.0, 2000.0),
        f.GirrVega("EUR", 5.0, 10.0, -1500.0),
        f.GirrVega("GBP", 0.5, 1.0, 800.0),
    ]
    rho = min(math.exp(-0.01 * 4 / 1) * math.exp(-0.01 * 5 / 5), 1.0)
    k_eur = math.sqrt(2000**2 + 1500**2 - 2 * rho * 2000 * 1500)
    want = math.sqrt(k_eur**2 + 800**2 + 2 * 0.5 * 500 * 800)
    assert f.girr_vega(sens)["medium"] == pytest.approx(want, rel=1e-14)


def test_fx_vega_single_bucket() -> None:
    sens = [f.FxVega("EURUSD", 0.5, 300.0), f.FxVega("EURUSD", 3.0, 200.0)]
    rho = math.exp(-0.01 * 2.5 / 0.5)
    want_low = math.sqrt(300**2 + 200**2 + 2 * max(2 * rho - 1, 0.75 * rho) * 300 * 200)
    assert f.fx_vega(sens)["low"] == pytest.approx(want_low, rel=1e-14)


def test_bad_tenor_rejected() -> None:
    with pytest.raises(InputError):
        f.girr_delta([f.GirrDelta("USD", "OIS", 7.0, 1.0)])
