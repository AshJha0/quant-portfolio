"""Signal combination, vol targeting, session and carry filters."""

import numpy as np
import pandas as pd
import pytest

from fx_algo import (
    carry_gate,
    combine_signals,
    rolling_zscore,
    session_filter,
    vol_target_positions,
)


def test_rolling_zscore_basic():
    x = pd.Series(np.arange(10.0))
    z = rolling_zscore(x, 4, min_periods=4)
    # window [0,1,2,3]: mean 1.5, std(ddof=0) sqrt(1.25)
    assert z.iloc[3] == pytest.approx((3 - 1.5) / np.sqrt(1.25))
    assert z.iloc[:3].tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        rolling_zscore(x, 1)


def test_rolling_zscore_zero_variance_gives_zero():
    z = rolling_zscore(pd.Series(np.ones(20)), 5)
    assert (z == 0.0).all()


def test_combine_signals_weights_and_clip():
    idx = range(50)
    f = pd.DataFrame({"a": np.random.default_rng(0).standard_normal(50)}, index=idx)
    s1 = combine_signals(f, {"a": 1.0}, zscore_window=10, clip=3.0)
    s2 = combine_signals(f, {"a": 2.0}, zscore_window=10, clip=1e9)
    za = rolling_zscore(f["a"], 10)
    assert np.allclose(s2, 2.0 * za)
    assert (s1.abs() <= 3.0).all()


def test_combine_signals_unknown_feature_raises():
    f = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        combine_signals(f, {"nope": 1.0})


def test_vol_target_scaling_exact_on_constant_vol():
    n = 300
    ret = pd.Series(0.001 * (-1.0) ** np.arange(n))  # alternating, std exactly 10bp
    sig = pd.Series(np.ones(n))
    pos = vol_target_positions(
        sig, ret, target_ann_vol=0.10, bars_per_year=6264.0, vol_window=100, max_leverage=50.0
    )
    expected = 0.10 / (0.001 * np.sqrt(6264.0))
    assert pos.iloc[-1] == pytest.approx(expected, rel=1e-6)


def test_vol_target_cap_and_nan_handling():
    ret = pd.Series(np.zeros(50))  # zero vol -> unbounded scale -> capped? -> NaN -> 0
    pos = vol_target_positions(pd.Series(np.ones(50)), ret, max_leverage=2.0)
    assert (pos == 0.0).all()
    r2 = pd.Series(1e-9 * np.random.default_rng(0).standard_normal(200))
    pos2 = vol_target_positions(pd.Series(np.ones(200)), r2, max_leverage=2.0)
    assert (pos2.abs() <= 2.0).all()
    assert pos2.iloc[-1] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        vol_target_positions(pd.Series([1.0]), pd.Series([0.0]), target_ann_vol=0.0)


def test_session_filter_zeroes_disallowed_sessions():
    hours = np.array([3.0, 8.0, 13.0, 18.0, 22.0])
    pos = pd.Series(np.ones(5))
    out = session_filter(pos, hours, allowed_sessions=("london", "overlap", "ny"))
    assert out.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]


def test_carry_gate_flattens_disagreeing_overnight_position():
    hours = np.array([19.0, 20.0, 21.0, 22.0])
    pos = pd.Series([1.0, 1.0, 1.0, 1.0])
    carry = np.full(4, -0.02)  # negative carry, long position
    out = carry_gate(pos, hours, carry, rollover_hour=21.0, bar_hours=1.0)
    # only the bar whose holding interval (20,21] crosses the rollover is cut
    assert out.tolist() == [1.0, 0.0, 1.0, 1.0]


def test_carry_gate_keeps_agreeing_position():
    hours = np.array([20.0, 21.0])
    pos = pd.Series([-1.0, -1.0])
    out = carry_gate(pos, hours, np.full(2, -0.02), rollover_hour=21.0)
    assert out.tolist() == [-1.0, -1.0]
