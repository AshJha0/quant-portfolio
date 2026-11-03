"""Tests for research/capacity: known-answer capacity fixture (W23 acceptance)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "research" / "capacity")]

import capacity as cap  # noqa: E402


def _fixture(t: int = 240):
    idx = pd.RangeIndex(t + 1)
    w = pd.DataFrame({"X": [float(i % 2) for i in range(t + 1)]}, index=idx)  # 0,1,0,1,...
    z = np.tile([1.0, -1.0], t // 2)
    g = pd.Series(0.01 + 0.02 * z, index=idx[1:])
    model = cap.CapacityModel(half_spread={"X": 0.0}, sigma={"X": 0.01}, adv={"X": 1e9})
    return g, w, model


def test_known_answer_capacity():
    g, w, model = _fixture()
    c = cap.cost_series(w, 1e8, model).iloc[1:]
    np.testing.assert_allclose(c, 0.01 * math.sqrt(0.1), rtol=1e-14)
    a = cap.capacity(g, w, model)
    assert a == pytest.approx(2.5e8, rel=1e-8)
    s0 = cap.net_sharpe(g, w, 0.0, model)
    assert cap.net_sharpe(g, w, a, model) == pytest.approx(s0 / 2, rel=1e-8)


def test_curve_is_decreasing_and_spread_lowers_s0():
    g, w, model = _fixture()
    cv = cap.curve(g, w, model, np.array([0, 1e6, 1e7, 1e8, 1e9]))
    assert (np.diff(cv.to_numpy()) < 0).all()
    m2 = cap.CapacityModel(half_spread={"X": 0.002}, sigma={"X": 0.01}, adv={"X": 1e9})
    assert cap.net_sharpe(g, w, 0.0, m2) < cap.net_sharpe(g, w, 0.0, model)


def test_nan_and_inf_edges():
    g, w, model = _fixture()
    assert math.isnan(cap.capacity(g - 0.02, w, model))
    tiny = cap.CapacityModel(half_spread={"X": 0.0}, sigma={"X": 1e-12}, adv={"X": 1e9})
    assert cap.capacity(g, w, tiny) == math.inf


def test_turnover_and_holding_period():
    w = pd.DataFrame({"A": [1.0, -1.0, 1.0, -1.0], "B": 0.0})
    assert cap.holding_period(w) == pytest.approx(1.0)
    np.testing.assert_allclose(cap.turnover(w), [0.5, 1.0, 1.0, 1.0])
    rows = [[1.0 - (t // 4) % 2, float((t // 4) % 2)] for t in range(40)]
    # 9 switches of 2 units over 39 diffs; gross 1 -> 2 / (18/39) = 4.333
    assert cap.holding_period(pd.DataFrame(rows, columns=["A", "B"])) == pytest.approx(
        2.0 / (18 / 39), rel=1e-12)


def test_bis_adv():
    assert cap.adv_bis("EUR") == pytest.approx(7.5e12 * 30.5 / 200)
    assert sum(cap.BIS_SHARE_2022.values()) < 200
