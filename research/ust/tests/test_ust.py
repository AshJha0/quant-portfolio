"""Tests for research/ust: GSW curve and DV01-neutral slope and fly (W23 D)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "research" / "ust")]

import curve_trades as ct  # noqa: E402
import gsw  # noqa: E402


def P(**kw):
    base = {"BETA0": 4.0, "BETA1": 0.0, "BETA2": 0.0, "BETA3": 0.0, "TAU1": 1.0, "TAU2": 5.0}
    base.update(kw)
    return pd.Series(base)


def test_docstring_yield():
    y = gsw.zero_yield(P(BETA0=5.0, BETA1=-2.0), [1.0, 1e4])
    assert y[0] == pytest.approx((5 - 2 * (1 - math.exp(-1))) / 100, abs=1e-15)
    assert y[0] == pytest.approx(0.03735758882342885, abs=1e-15)
    assert y[1] == pytest.approx(0.05, abs=1e-5)


def test_svensson_term_and_nelson_siegel_fallback():
    p = P(BETA3=1.0, TAU2=2.0)
    n = 3.0
    l2 = (1 - math.exp(-n / 2)) / (n / 2)
    want = (4.0 + (l2 - math.exp(-n / 2))) / 100
    assert gsw.zero_yield(p, n)[0] == pytest.approx(want, abs=1e-15)
    ns = P(BETA3=float("nan"), TAU2=float("nan"))
    assert gsw.zero_yield(ns, n)[0] == pytest.approx(0.04, abs=1e-15)
    with pytest.raises(ValueError):
        gsw.zero_yield(p, 0.0)


def test_flat_curve_zero_carry_and_parallel_shift():
    h = 1 / 12
    assert ct.trade_carry(P(), "slope", (2, 10), h) == pytest.approx(0.0, abs=1e-15)
    assert ct.trade_carry(P(), "fly", (2, 5, 10), h) == pytest.approx(0.0, abs=1e-15)
    r = ct.trade_return(P(), P(BETA0=4.5), "slope", (2, 10), h)
    assert r == pytest.approx(-0.002 * h, abs=1e-15)
    assert r == pytest.approx(-0.0049583333333333 + 0.0047916666666667, abs=1e-13)


def test_dv01_neutral_legs():
    for kind, ten in (("slope", (2, 10)), ("fly", (2, 5, 10))):
        assert sum(w * n for n, w in ct.legs(kind, ten)) == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        ct.legs("box", (1, 2))


def test_upward_curve_carry_by_hand():
    """Per unit DV01 the 2y rolls down a steeper part of this concave curve than
    the 10y, so the DV01-neutral steepener has negative carry: the rule is
    flattener here, a known feature of concave curves, not a sign error."""
    up = P(BETA0=5.0, BETA1=-3.0, TAU1=2.0)
    h = 1 / 12

    def y(n):
        x = n / 2.0
        return (5.0 - 3.0 * (1 - math.exp(-x)) / x) / 100

    def cr(n):
        return n * y(n) - (n - h) * y(n - h) - h * y(h)

    want = cr(10) / 10 - cr(2) / 2
    assert ct.trade_carry(up, "slope", (2, 10), h) == pytest.approx(want, abs=1e-15)
    assert want < 0
    assert ct.trade_return(up, up, "slope", (2, 10), h) == pytest.approx(
        ct.trade_carry(up, "slope", (2, 10), h), abs=1e-16)


def test_monthly_study_and_load(tmp_path):
    days = pd.bdate_range("2020-01-01", "2020-06-30")
    rows = [{"BETA0": 5.0, "BETA1": -3.0 + 0.01 * i, "BETA2": 0.0, "BETA3": 0.0,
             "TAU1": 2.0, "TAU2": 5.0} for i in range(len(days))]
    curve = pd.DataFrame(rows, index=days)
    out = ct.monthly_study(curve, "slope", (2, 10), cost_per_flip=1e-4)
    assert len(out) == 5 and (out["position"] == -1.0).all()
    assert out["net"].iloc[0] == pytest.approx(-out["rx"].iloc[0] - 1e-4, abs=1e-16)
    np.testing.assert_allclose(out["net"].iloc[1:], -out["rx"].iloc[1:])
    lines = [f"note {i}" for i in range(9)]
    lines += ["Date,BETA0,BETA1,BETA2,BETA3,SVENY01,TAU1,TAU2",
              "2020-01-02,5,-3,0,0,3.1,2,5", "2020-01-03,NA,NA,NA,NA,NA,NA,NA"]
    f = tmp_path / "feds200628.csv"
    f.write_text("\n".join(lines) + "\n")
    df, sha = gsw.load(f)
    assert list(df.columns) == list(gsw.PARAMS) and len(df) == 1 and len(sha) == 64
