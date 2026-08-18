"""Market simulator: session profiles, impact behaviour, determinism."""

import numpy as np
import pytest

from fx_algo import (
    EURUSD,
    FirmVenue,
    MarketSimulator,
    liquidity_weighted_schedule,
    make_time_grid,
    twap_schedule,
    weekend_mask,
)


def make_sim(**kw):
    return MarketSimulator(EURUSD, **kw)


def test_grid_and_profile_arrays_match_configuration():
    sim = make_sim(dt_minutes=5.0)
    assert sim.n_buckets == 288
    # bucket midpoints in each session carry the configured spread/depth
    hours = sim.times_hours + 0.5 * 5.0 / 60.0
    assert sim.spread_pips[(hours >= 12) & (hours < 17)].max() == pytest.approx(0.2)
    assert sim.spread_pips[(hours >= 21)].min() == pytest.approx(1.0)
    assert sim.depth_bucket[(hours >= 12) & (hours < 17)].max() == pytest.approx(70.0 * 5.0)


def test_overlap_is_tightest_and_deepest_in_simulator():
    sim = make_sim(dt_minutes=30.0)
    sess_hours = {"asia": 3.0, "london": 9.0, "overlap": 13.0, "ny": 18.0, "late": 22.0}
    spread = {}
    depth = {}
    for name, h in sess_hours.items():
        i = int(np.argmin(np.abs(sim.times_hours - h)))
        spread[name] = sim.spread_pips[i]
        depth[name] = sim.depth_bucket[i]
    assert spread["overlap"] == min(spread.values())
    assert depth["overlap"] == max(depth.values())
    assert spread["late"] == max(spread.values())


def test_zero_trade_zero_impact_and_path_equals_no_trade_path():
    sim = make_sim(dt_minutes=5.0)
    r = sim.execute(np.zeros(sim.n_buckets), FirmVenue(), seed=42)
    base = sim.simulate_mids(42)
    assert np.allclose(r.mids_path, base, atol=0.0)
    assert (r.temp_pips == 0.0).all()
    assert (r.perm_cum_pips == 0.0).all()
    assert r.is_pips == 0.0


def test_temporary_impact_hits_fill_not_mid():
    sim = make_sim(dt_minutes=5.0)
    sched = np.zeros(sim.n_buckets)
    sched[150] = 50.0
    r = sim.execute(sched, FirmVenue(), seed=7)
    base = sim.simulate_mids(7)
    # pre-trade mid of the traded bucket is untouched by the trade
    assert r.mids_pre[150] == pytest.approx(base[150])
    # the fill is above mid by half-spread + temporary impact
    fill_premium_pips = (r.fills[150] - r.mids_pre[150]) / EURUSD.pip_size
    assert fill_premium_pips == pytest.approx(0.5 * r.spread_pips[150] + r.temp_pips[150])
    assert r.temp_pips[150] > 0


def test_permanent_impact_persists_temporary_decays():
    sim = make_sim(dt_minutes=5.0)
    sched = np.zeros(sim.n_buckets)
    sched[100] = 50.0
    r = sim.execute(sched, FirmVenue(), seed=7)
    base = sim.simulate_mids(7)
    diff_pips = (r.mids_path - base) / EURUSD.pip_size
    assert np.allclose(diff_pips[: 101], 0.0)
    # after the trade every mid is shifted by the same permanent amount
    part = 50.0 / sim.depth_bucket[100]
    expected_perm = sim.k_perm * sim.sigma_bucket_pips[100] * part
    assert np.allclose(diff_pips[101:], expected_perm)
    assert expected_perm > 0


def test_sqrt_impact_scaling_with_size():
    sim = make_sim(dt_minutes=5.0)
    s1 = np.zeros(sim.n_buckets)
    s4 = np.zeros(sim.n_buckets)
    s1[150], s4[150] = 15.0, 60.0  # 4x size (overlap depth 350: both under cap)
    r1 = sim.execute(s1, FirmVenue(), seed=0)
    r4 = sim.execute(s4, FirmVenue(), seed=0)
    assert r4.temp_pips[150] / r1.temp_pips[150] == pytest.approx(2.0, rel=1e-12)


def test_impact_scaled_by_session_depth():
    sim = make_sim(dt_minutes=5.0)
    q = 10.0
    s_thin = np.zeros(sim.n_buckets)
    s_deep = np.zeros(sim.n_buckets)
    i_late = int(np.argmin(np.abs(sim.times_hours - 22.0)))
    i_ovl = int(np.argmin(np.abs(sim.times_hours - 13.0)))
    s_thin[i_late] = q
    s_deep[i_ovl] = q
    r_thin = sim.execute(s_thin, FirmVenue(), seed=0)
    r_deep = sim.execute(s_deep, FirmVenue(), seed=0)
    # same size, per unit of session vol the thin bucket costs sqrt(D2/D1)
    # times more: temp/sigma = k*sqrt(q/D)
    ratio = (r_thin.temp_pips[i_late] / sim.sigma_bucket_pips[i_late]) / (
        r_deep.temp_pips[i_ovl] / sim.sigma_bucket_pips[i_ovl]
    )
    expected = np.sqrt(sim.depth_bucket[i_ovl] / sim.depth_bucket[i_late])
    assert ratio == pytest.approx(expected, rel=1e-12)
    assert ratio > 1.0


def test_seeded_reproducibility():
    sim = make_sim(dt_minutes=5.0)
    sched = twap_schedule(200.0, sim.n_buckets)
    a = sim.execute(sched, FirmVenue(), seed=9)
    b = sim.execute(sched, FirmVenue(), seed=9)
    assert np.array_equal(a.fills, b.fills)
    assert np.array_equal(a.mids_pre, b.mids_pre)
    c = sim.execute(sched, FirmVenue(), seed=10)
    assert not np.allclose(a.fills, c.fills)


def test_zero_vol_path_deterministic_cost():
    sim = make_sim(dt_minutes=5.0, vol_scale=0.0)
    lw = liquidity_weighted_schedule(500.0, sim.depth_bucket)
    r = sim.execute(lw, FirmVenue(), seed=3)
    # flat mids, cost is exactly the qty-weighted half-spread (sigma=0
    # kills impact too in this parameterisation)
    assert np.allclose(r.mids_pre, EURUSD.s0)
    q = np.abs(r.qty)
    expected = np.sum(q * 0.5 * r.spread_pips) / q.sum()
    assert r.is_pips == pytest.approx(expected, abs=1e-9)


def test_depth_cap_exceeded_raises_multi_session_hint():
    sim = make_sim(dt_minutes=5.0)
    sched = np.zeros(sim.n_buckets)
    sched[150] = 500.0  # 500mm into one 5-min bucket
    with pytest.raises(ValueError, match="depth cap"):
        sim.execute(sched, FirmVenue(), seed=0)


def test_mixed_sign_schedule_raises():
    sim = make_sim(dt_minutes=60.0)
    sched = np.zeros(sim.n_buckets)
    sched[0], sched[1] = 10.0, -10.0
    with pytest.raises(ValueError, match="mixes"):
        sim.execute(sched, FirmVenue(), seed=0)


def test_sell_side_symmetric():
    sim = make_sim(dt_minutes=5.0)
    lw = liquidity_weighted_schedule(-300.0, sim.depth_bucket)
    r = sim.execute(lw, FirmVenue(), seed=1)
    assert r.side == -1
    assert r.total_qty == pytest.approx(-300.0)
    # sells fill below mid
    active = np.abs(r.qty) > 0
    assert (r.fills[active] < r.mids_pre[active]).all()
    assert r.is_pips > 0  # cost is still positive in pips


def test_weekend_bucket_trading_raises_and_market_frozen():
    times = make_time_grid(0.0, 48.0, 60.0)
    wk = weekend_mask(times, 24.0, 40.0)
    sim = MarketSimulator(EURUSD, 0.0, 48.0, 60.0, tradeable=wk)
    bad = np.zeros(len(times))
    bad[30] = 1.0  # inside the weekend
    with pytest.raises(ValueError, match="non-tradeable"):
        sim.execute(bad, FirmVenue(), seed=0)
    # no diffusion during the gap
    mids = sim.simulate_mids(0)
    assert np.allclose(np.diff(mids)[~wk], 0.0)


def test_schedule_length_mismatch_raises():
    sim = make_sim(dt_minutes=60.0)
    with pytest.raises(ValueError, match="length"):
        sim.execute(np.zeros(5), FirmVenue(), seed=0)


def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        MarketSimulator(EURUSD, k_temp=-0.1)
    with pytest.raises(ValueError):
        MarketSimulator(EURUSD, max_participation=0.0)
    with pytest.raises(ValueError):
        MarketSimulator(EURUSD, tradeable=np.array([True, False]))
