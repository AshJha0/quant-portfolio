"""Hand examples for research/tsmom/tsmom.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "research" / "tsmom"), str(REPO / "research" / "panel"),
                str(REPO / "research" / "tools")]

import tsmom as ts  # noqa: E402
from holdout_lock import HoldoutLocked  # noqa: E402

M = pd.date_range("2001-01-31", periods=13, freq="ME")


@pytest.mark.parametrize(("last", "earned"), [(0.02, 0.08), (-0.02, -0.08)])
def test_docstring_worked_example(last, earned):
    rx = pd.DataFrame({"A": [0.01] * 12 + [last]}, index=M)
    vol = pd.DataFrame({"A": 0.10}, index=M)
    res = ts.tsmom(rx, vol)
    assert res["weights"]["A"].iloc[11] == pytest.approx(4.0)
    assert len(res["returns"]) == 1
    assert res["returns"]["A"].iloc[0] == pytest.approx(earned, abs=1e-15)
    assert res["portfolio"].iloc[0] == pytest.approx(earned, abs=1e-15)
    assert res["label"] == "exploratory"


def test_no_lookahead_weights_use_past_only():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    rx = pd.DataFrame({"A": rng.normal(0, 0.03, 60)}, index=idx)
    base = ts.weights(rx)
    bumped = rx.copy()
    bumped.iloc[40:] += 1.0  # later data must not move weights set before month 40
    pd.testing.assert_frame_equal(ts.weights(bumped).iloc[:40], base.iloc[:40])


def test_short_when_trend_negative_and_portfolio_equal_weight():
    rx = pd.DataFrame({"A": [0.01] * 12 + [0.03], "B": [-0.01] * 12 + [-0.03]}, index=M)
    vol = pd.DataFrame({"A": 0.20, "B": 0.40}, index=M)
    res = ts.tsmom(rx, vol)
    assert res["returns"]["A"].iloc[0] == pytest.approx(2.0 * 0.03)
    assert res["returns"]["B"].iloc[0] == pytest.approx(-1.0 * -0.03)
    assert res["portfolio"].iloc[0] == pytest.approx((0.06 + 0.03) / 2)


def test_ewma_vol_constant_returns():
    rx = pd.DataFrame({"A": [0.02] * 24}, index=pd.date_range("2000-01-31", periods=24, freq="ME"))
    v = ts.ewma_vol(rx)
    assert v["A"].iloc[:11].isna().all()
    assert v["A"].iloc[-1] == pytest.approx(0.02 * np.sqrt(12), rel=1e-12)


def test_ewma_vol_matches_hand_recursion():
    x = np.array([0.01, -0.03, 0.02, 0.05, -0.01])
    rx = pd.DataFrame({"A": x})
    com = 2.0
    a = 1 / (1 + com)
    num = den = 0.0
    for xi in x:
        num = (1 - a) * num + xi**2
        den = (1 - a) * den + 1
    got = ts.ewma_vol(rx, com=com, min_periods=1)["A"].iloc[-1]
    assert got == pytest.approx(np.sqrt(12 * num / den), rel=1e-12)


def test_turnover_counts_first_build_and_flips():
    w = pd.DataFrame({"A": [np.nan, 1.0, 1.0, -1.0]})
    assert ts.turnover(w).tolist() == [1.0, 0.0, 2.0]


def test_assemble_assets_prefixes_and_aligns():
    fx = pd.DataFrame({"AUD": [0.01]}, index=pd.DatetimeIndex(["2001-01-31"]))
    eq = pd.Series([0.02], index=pd.DatetimeIndex(["2001-01-30"]), name="Mkt-RF")
    out = ts.assemble_assets(fx=fx, eq=eq)
    assert list(out.columns) == ["fx:AUD", "eq:Mkt-RF"]
    assert list(out.index) == [pd.Timestamp("2001-01-31")]


def test_lock_refuses_2016():
    idx = pd.date_range("2014-01-31", "2016-01-31", freq="ME")
    with pytest.raises(HoldoutLocked):
        ts.tsmom(pd.DataFrame({"A": 0.01}, index=idx))
