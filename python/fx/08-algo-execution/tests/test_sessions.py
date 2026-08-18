"""Sessions, profiles and time-grid utilities."""

import numpy as np
import pytest

from fx_algo import (
    EURUSD,
    GBPUSD,
    USDMXN,
    SESSION_BOUNDS,
    SESSION_NAMES,
    fix_window_mask,
    make_time_grid,
    session_of_hour,
    weekend_mask,
)


def test_session_bounds_cover_full_day():
    starts = sorted(lo for lo, _ in SESSION_BOUNDS.values())
    ends = sorted(hi for _, hi in SESSION_BOUNDS.values())
    assert starts[0] == 0.0 and ends[-1] == 24.0
    # contiguous, non-overlapping
    assert ends[:-1] == starts[1:]


def test_session_of_hour_values():
    assert session_of_hour(3.0)[0] == "asia"
    assert session_of_hour(8.5)[0] == "london"
    assert session_of_hour(13.0)[0] == "overlap"
    assert session_of_hour(16.0)[0] == "overlap"  # 4pm fix sits in the overlap
    assert session_of_hour(18.0)[0] == "ny"
    assert session_of_hour(22.5)[0] == "late"


def test_session_of_hour_wraps_absolute_hours():
    assert session_of_hour(24.0 + 3.0)[0] == "asia"
    assert session_of_hour(48.0 + 13.0)[0] == "overlap"


def test_profiles_have_all_sessions():
    for prof in (EURUSD, GBPUSD, USDMXN):
        for m in (prof.spread_pips, prof.depth_mm_per_min, prof.vol_pips_per_sqrt_min):
            assert set(m) == set(SESSION_NAMES)


def test_eurusd_overlap_is_tightest_and_deepest():
    sp = EURUSD.spread_pips
    # spread ordering: overlap < london < ny < asia < late
    assert sp["overlap"] < sp["london"] < sp["ny"] < sp["asia"] < sp["late"]
    d = EURUSD.depth_mm_per_min
    assert d["overlap"] == max(d.values())
    assert d["late"] == min(d.values())


def test_em_pair_much_wider_than_major():
    for s in SESSION_NAMES:
        assert USDMXN.spread_pips[s] > 10 * EURUSD.spread_pips[s]


def test_profile_lookup_vectorised():
    hours = np.array([3.0, 8.0, 13.0, 18.0, 22.0])
    sp = EURUSD.spread_pips_at(hours)
    assert np.allclose(sp, [0.6, 0.35, 0.2, 0.4, 1.0])
    assert np.allclose(EURUSD.depth_at(13.0), [70.0])


def test_make_time_grid_lengths_and_spacing():
    t = make_time_grid(7.0, 9.0, 5.0)
    assert len(t) == 9 * 12
    assert t[0] == 7.0
    assert np.allclose(np.diff(t), 5.0 / 60.0)


def test_make_time_grid_invalid():
    with pytest.raises(ValueError):
        make_time_grid(0.0, 24.0, 0.0)
    with pytest.raises(ValueError):
        make_time_grid(0.0, -1.0, 5.0)


def test_weekend_mask_half_open():
    t = np.array([23.0, 24.0, 30.0, 39.9, 40.0])
    m = weekend_mask(t, 24.0, 40.0)
    assert m.tolist() == [True, False, False, False, True]


def test_fix_window_mask_one_minute_grid():
    t = make_time_grid(14.0, 3.0, 1.0)
    m = fix_window_mask(t, 1.0)
    assert m.sum() == 5
    minutes = np.round(t[m] * 60).astype(int) % 1440
    assert minutes.tolist() == [957, 958, 959, 960, 961]  # 15:57..16:01


def test_fix_window_mask_off_grid_empty():
    t = make_time_grid(0.0, 6.0, 5.0)  # Asia only
    assert not fix_window_mask(t, 5.0).any()
