"""Schedulers: TWAP, liquidity-weighted, POV-analog, fix-targeting."""

import numpy as np
import pytest

from fx_algo import (
    EURUSD,
    MarketSimulator,
    fix_schedule,
    fix_window_mask,
    liquidity_weighted_schedule,
    make_time_grid,
    pov_schedule,
    twap_schedule,
    weekend_mask,
)


def test_twap_equal_slices_and_exact_sum():
    q = twap_schedule(500.0, 288)
    assert np.allclose(q, 500.0 / 288)
    assert q.sum() == pytest.approx(500.0, abs=1e-9)


def test_twap_respects_weekend_mask():
    mask = np.array([True, False, False, True])
    q = twap_schedule(10.0, 4, tradeable=mask)
    assert q.tolist() == [5.0, 0.0, 0.0, 5.0]


def test_twap_single_bucket():
    assert twap_schedule(7.0, 1).tolist() == [7.0]


def test_twap_invalid_inputs():
    with pytest.raises(ValueError):
        twap_schedule(0.0, 10)
    with pytest.raises(ValueError):
        twap_schedule(1.0, 0)
    with pytest.raises(ValueError):
        twap_schedule(1.0, 3, tradeable=np.zeros(3, dtype=bool))


def test_liquidity_weighted_exactly_proportional_to_depth():
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    q = liquidity_weighted_schedule(500.0, sim.depth_bucket)
    assert q.sum() == pytest.approx(500.0, abs=1e-9)
    expected = 500.0 * sim.depth_bucket / sim.depth_bucket.sum()
    assert np.allclose(q, expected, rtol=1e-12)
    # constant participation rate across buckets
    assert np.allclose(q / sim.depth_bucket, q[0] / sim.depth_bucket[0])


def test_liquidity_weighted_concentrates_in_overlap():
    sim = MarketSimulator(EURUSD, dt_minutes=60.0)
    q = liquidity_weighted_schedule(240.0, sim.depth_bucket)
    h = sim.times_hours + 0.5
    per_hour_overlap = q[(h >= 12) & (h < 17)].mean()
    per_hour_late = q[h >= 21].mean()
    assert per_hour_overlap == pytest.approx(per_hour_late * 70.0 / 8.0)


def test_liquidity_weighted_zero_depth_raises():
    with pytest.raises(ValueError):
        liquidity_weighted_schedule(10.0, np.array([1.0, 0.0, 2.0]))


def test_pov_cap_respected_and_completes():
    vols = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    q = pov_schedule(120.0, vols, participation=0.3)
    assert q.sum() == pytest.approx(120.0)
    assert (q <= 0.3 * vols + 1e-12).all()
    assert q.tolist() == [30.0, 30.0, 30.0, 30.0, 0.0]


def test_pov_infeasible_raises():
    with pytest.raises(ValueError, match="incomplete"):
        pov_schedule(1000.0, np.full(5, 100.0), participation=0.1)
    with pytest.raises(ValueError):
        pov_schedule(10.0, np.full(5, 100.0), participation=0.0)


def test_pov_skips_weekend_buckets():
    vols = np.full(6, 100.0)
    mask = np.array([True, False, False, True, True, True])
    q = pov_schedule(90.0, vols, participation=0.3, tradeable=mask)
    assert q[1] == 0.0 and q[2] == 0.0
    assert q.sum() == pytest.approx(90.0)


def test_pov_sell_side():
    q = pov_schedule(-50.0, np.full(4, 100.0), participation=0.25)
    assert (q <= 0).all()
    assert q.sum() == pytest.approx(-50.0)


def test_fix_schedule_concentrates_in_window():
    times = make_time_grid(14.0, 3.0, 1.0)
    q = fix_schedule(100.0, times, 1.0)
    mask = fix_window_mask(times, 1.0)
    assert mask.sum() == 5
    assert np.allclose(q[mask], 20.0)
    assert np.allclose(q[~mask], 0.0)
    assert q.sum() == pytest.approx(100.0)


def test_fix_schedule_outside_window_raises():
    times = make_time_grid(0.0, 6.0, 5.0)
    with pytest.raises(ValueError, match="fix window"):
        fix_schedule(100.0, times, 5.0)


def test_schedules_feed_simulator_and_sum_to_parent():
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    twap = twap_schedule(500.0, sim.n_buckets)
    lw = liquidity_weighted_schedule(500.0, sim.depth_bucket)
    pov = pov_schedule(500.0, sim.depth_bucket, participation=0.05)
    for sched in (twap, lw, pov):
        r = sim.execute(sched, seed=0)
        assert r.total_qty == pytest.approx(500.0, abs=1e-9)


def test_weekend_gap_end_to_end():
    times = make_time_grid(0.0, 48.0, 30.0)
    wk = weekend_mask(times, 24.0, 40.0)
    sim = MarketSimulator(EURUSD, 0.0, 48.0, 30.0, tradeable=wk)
    q = twap_schedule(100.0, len(times), tradeable=wk)
    r = sim.execute(q, seed=0)
    assert np.all(r.qty[~wk] == 0.0)
    assert r.total_qty == pytest.approx(100.0)
