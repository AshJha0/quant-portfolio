"""Hand examples for research/vix/vix_rv.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "research" / "vix"), str(REPO / "research" / "panel"),
                str(REPO / "research" / "tools")]

import vix_rv  # noqa: E402
from holdout_lock import HoldoutLocked  # noqa: E402

DAYS = pd.bdate_range("2010-01-04", periods=120)


def test_docstring_worked_example():
    vix = pd.Series(20.0, index=DAYS)
    r = pd.Series(np.where(np.arange(len(DAYS)) % 2 == 0, 0.01, -0.01), index=DAYS)
    res = vix_rv.vix_study(vix, r)
    assert res["mean_iv"] == pytest.approx(0.04, abs=1e-15)
    assert res["mean_rv"] == pytest.approx(0.0252, abs=1e-15)
    assert res["mean_vrp"] == pytest.approx(0.0148, abs=1e-15)
    assert res["share_positive"] == 1.0
    assert res["days"] == len(DAYS) - vix_rv.HORIZON
    assert "no option prices" in res["label"] and "exploratory" in res["label"]


def test_forward_window_excludes_today_and_ends_nan():
    r = pd.Series(0.0, index=DAYS)
    r.iloc[0] = 0.10  # today's return must not count in today's forward RV
    rv = vix_rv.forward_realised_variance(r, horizon=3)
    assert rv.iloc[0] == 0.0
    r2 = pd.Series(0.0, index=DAYS)
    r2.iloc[3] = 0.10
    rv2 = vix_rv.forward_realised_variance(r2, horizon=3)
    assert rv2.iloc[0] == pytest.approx(0.01 * 252 / 3)
    assert rv2.iloc[2] == pytest.approx(0.01 * 252 / 3)
    assert rv2.iloc[3] == 0.0
    assert rv2.iloc[-3:].isna().all() and not math.isnan(rv2.iloc[-4])


def test_mincer_zarnowitz_recovers_known_line():
    x = pd.Series(np.linspace(0.01, 0.09, 50))
    y = 0.002 + 0.7 * x
    mz = vix_rv.mincer_zarnowitz(y, x)
    assert mz["intercept"] == pytest.approx(0.002, abs=1e-12)
    assert mz["slope"] == pytest.approx(0.7, abs=1e-12)
    assert mz["r2"] == pytest.approx(1.0, abs=1e-12)


def test_mz_matches_numpy_polyfit():
    rng = np.random.default_rng(7)
    x = pd.Series(rng.uniform(0.01, 0.1, 300))
    y = 0.01 + 0.6 * x + rng.normal(0, 0.01, 300)
    b, a = np.polyfit(x, y, 1)
    mz = vix_rv.mincer_zarnowitz(y, x)
    assert mz["slope"] == pytest.approx(b, rel=1e-10)
    assert mz["intercept"] == pytest.approx(a, rel=1e-10)


def test_market_log_returns_adds_rf():
    ff = pd.DataFrame({"Mkt-RF": [0.01], "RF": [0.0002]}, index=DAYS[:1])
    assert vix_rv.market_log_returns(ff).iloc[0] == pytest.approx(math.log(1.0102))


def test_lock_refuses_2016():
    d = pd.bdate_range("2015-12-01", "2016-01-15")
    with pytest.raises(HoldoutLocked):
        vix_rv.vix_study(pd.Series(20.0, index=d), pd.Series(0.01, index=d))
