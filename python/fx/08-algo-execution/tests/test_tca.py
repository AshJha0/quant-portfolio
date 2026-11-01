"""TCA: exact IS decomposition, benchmarks, venue attribution."""

import numpy as np
import pytest

from fx_algo import (
    EURUSD,
    FirmVenue,
    LastLookVenue,
    MarketSimulator,
    decompose_implementation_shortfall,
    fix_benchmark,
    fix_print,
    fix_self_impact_pips,
    fix_schedule,
    liquidity_weighted_schedule,
    make_time_grid,
    rejection_cost_pips,
    slippage_vs_benchmark,
    twap_benchmark,
    twap_schedule,
    venue_comparison,
)


def run_lw(seed=0, venue=None, alpha=0.0, dt=5.0):
    sim = MarketSimulator(EURUSD, dt_minutes=dt)
    lw = liquidity_weighted_schedule(500.0, sim.depth_bucket)
    return sim.execute(lw, venue or FirmVenue(), seed=seed, alpha_pips_per_bucket=alpha)


def test_is_components_sum_exactly():
    for seed in range(5):
        for venue in (FirmVenue(), LastLookVenue()):
            r = run_lw(seed=seed, venue=venue, alpha=0.3)
            d = decompose_implementation_shortfall(r)
            resid = d["total"] - (
                d["spread_temporary"] + d["permanent_impact"] + d["market_drift"]
            )
            assert resid == pytest.approx(0.0, abs=1e-10)
            assert d["total"] == pytest.approx(r.is_pips, abs=1e-10)


def test_is_decomposition_zero_vol_isolates_spread():
    sim = MarketSimulator(EURUSD, dt_minutes=5.0, vol_scale=0.0)
    lw = liquidity_weighted_schedule(500.0, sim.depth_bucket)
    d = decompose_implementation_shortfall(sim.execute(lw, seed=0))
    assert d["market_drift"] == pytest.approx(0.0, abs=1e-12)
    assert d["permanent_impact"] == pytest.approx(0.0, abs=1e-12)
    assert d["total"] == pytest.approx(d["spread_temporary"])
    assert d["spread_temporary"] > 0


def test_is_decomposition_empty_execution():
    sim = MarketSimulator(EURUSD, dt_minutes=60.0)
    r = sim.execute(np.zeros(sim.n_buckets), seed=0)
    d = decompose_implementation_shortfall(r)
    assert d == {
        "total": 0.0,
        "spread_temporary": 0.0,
        "permanent_impact": 0.0,
        "market_drift": 0.0,
    }


def test_permanent_component_positive_and_schedule_deterministic():
    # permanent impact of buy pressure raises later fills; the component
    # is a deterministic function of the schedule (path noise cancels)
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    lw = liquidity_weighted_schedule(500.0, sim.depth_bucket)
    perms = [
        decompose_implementation_shortfall(sim.execute(lw, seed=s))["permanent_impact"]
        for s in range(20)
    ]
    assert np.mean(perms) > 0
    assert np.std(perms) < 1e-12  # exactly schedule-determined in this model


def test_twap_benchmark_is_mean_of_mids():
    r = run_lw(seed=1)
    assert twap_benchmark(r) == pytest.approx(float(r.mids_pre.mean()))


def test_slippage_sign_convention():
    r = run_lw(seed=2)
    # buying above the benchmark costs money
    below = r.avg_fill - 10 * EURUSD.pip_size
    assert slippage_vs_benchmark(r, below) == pytest.approx(10.0, abs=1e-9)


def test_fix_benchmark_equals_window_twap():
    sim = MarketSimulator(EURUSD, start_hour=14.0, horizon_hours=3.0, dt_minutes=1.0)
    q = fix_schedule(100.0, sim.times_hours, 1.0)
    r = sim.execute(q, seed=0)
    from fx_algo import fix_window_mask

    mask = fix_window_mask(r.times_hours, 1.0)
    assert fix_benchmark(r) == pytest.approx(float(r.mids_pre[mask].mean()))


def test_fix_benchmark_outside_grid_raises():
    sim = MarketSimulator(EURUSD, start_hour=0.0, horizon_hours=6.0, dt_minutes=5.0)
    r = sim.execute(np.zeros(sim.n_buckets), seed=0)
    with pytest.raises(ValueError):
        fix_benchmark(r)


def test_fix_benchmark_from_own_mids_is_a_tautology():
    # Documents why fix_benchmark is not used for tracking error: built from
    # the same mids the schedule trades on, the fix schedule's tracking
    # error is half-spread plus temporary impact with zero variance.
    sim = MarketSimulator(EURUSD, start_hour=14.0, horizon_hours=3.0, dt_minutes=1.0)
    q_fix = fix_schedule(100.0, sim.times_hours, 1.0)
    te = [slippage_vs_benchmark(r, fix_benchmark(r)) for r in (sim.execute(q_fix, seed=s) for s in range(20))]
    assert np.std(te) < 1e-9


def test_fix_print_tracking_error_has_real_variance():
    # Against an independent print (intra-bucket market observations from a
    # separate RNG) the fix schedule's tracking error is random, and the
    # TWAP schedule's tracking error is still much larger.
    sim = MarketSimulator(EURUSD, start_hour=14.0, horizon_hours=3.0, dt_minutes=1.0)
    q_fix = fix_schedule(100.0, sim.times_hours, 1.0)
    q_twap = twap_schedule(100.0, sim.n_buckets)
    te_fix, te_twap = [], []
    for s in range(60):
        rf = sim.execute(q_fix, seed=s)
        rt = sim.execute(q_twap, seed=s)
        te_fix.append(slippage_vs_benchmark(rf, fix_print(rf, seed=1000 + s)))
        te_twap.append(slippage_vs_benchmark(rt, fix_print(rt, seed=1000 + s)))
    assert np.std(te_fix) > 0.1
    assert np.std(te_twap) > 5 * np.std(te_fix)


def test_fix_print_zero_vol_is_linear_interpolation_mean():
    sim = MarketSimulator(EURUSD, start_hour=14.0, horizon_hours=3.0, dt_minutes=1.0, vol_scale=0.0)
    r = sim.execute(fix_schedule(100.0, sim.times_hours, 1.0), seed=0)
    mask = make_mask(r)
    path = r.mids_path
    m = 10
    k = np.arange(m) / m
    expected = np.mean([path[t] + k * (path[t + 1] - path[t]) for t in np.flatnonzero(mask)])
    assert fix_print(r, seed=5, obs_per_bucket=m) == pytest.approx(expected, abs=1e-12)


def test_self_impact_on_print_is_visible_and_vanishes_without_permanent_impact():
    kw = dict(start_hour=14.0, horizon_hours=3.0, dt_minutes=1.0)
    sim = MarketSimulator(EURUSD, **kw)
    r = sim.execute(fix_schedule(100.0, sim.times_hours, 1.0), seed=4)
    si = fix_self_impact_pips(r, seed=9)
    assert si > 0.01
    # unimpacted print is cheaper for a buyer by exactly the self-impact
    assert slippage_vs_benchmark(r, fix_print(r, 9, include_own_impact=False)) == pytest.approx(
        slippage_vs_benchmark(r, fix_print(r, 9)) + si, abs=1e-9
    )
    sim0 = MarketSimulator(EURUSD, k_perm=0.0, **kw)
    r0 = sim0.execute(fix_schedule(100.0, sim0.times_hours, 1.0), seed=4)
    assert fix_self_impact_pips(r0, seed=9) == pytest.approx(0.0, abs=1e-12)


def test_fix_print_rejects_bad_obs_count():
    sim = MarketSimulator(EURUSD, start_hour=14.0, horizon_hours=3.0, dt_minutes=1.0)
    r = sim.execute(fix_schedule(100.0, sim.times_hours, 1.0), seed=0)
    with pytest.raises(ValueError):
        fix_print(r, obs_per_bucket=0)


def make_mask(r):
    from fx_algo.sessions import fix_window_mask

    return fix_window_mask(r.times_hours, r.dt_minutes)


def test_venue_attribution_consistent_totals():
    rl = run_lw(seed=3, venue=LastLookVenue(), alpha=0.5)
    d = decompose_implementation_shortfall(rl)
    vc = venue_comparison({"ll": rl})["ll"]
    # venue-attributable cost == the spread+temporary IS component
    assert vc["effective_cost_pips"] == pytest.approx(d["spread_temporary"], abs=1e-10)
    assert vc["rejection_cost_pips"] == pytest.approx(rejection_cost_pips(rl), abs=1e-12)


def test_venue_comparison_empty_raises():
    sim = MarketSimulator(EURUSD, dt_minutes=60.0)
    r = sim.execute(np.zeros(sim.n_buckets), seed=0)
    with pytest.raises(ValueError):
        venue_comparison({"v": r})
