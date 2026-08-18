"""Desk scenarios & edge cases: EM costs, scheduler savings, flash vol."""

import numpy as np
import pandas as pd
import pytest

from fx_algo import (
    BacktestConfig,
    EURUSD,
    FirmVenue,
    IntradayBacktester,
    MarketSimulator,
    USDMXN,
    build_bars,
    decompose_implementation_shortfall,
    generate_ticks,
    liquidity_weighted_schedule,
    momentum,
    pov_schedule,
    twap_schedule,
)

PIP = 1e-4
EM_SPREADS = {"asia": 30.0, "london": 12.0, "overlap": 10.0, "ny": 8.0, "late": 40.0}


def momentum_positions(bars):
    sig = np.sign(momentum(bars, 1).fillna(0.0))
    return pd.Series(sig, index=bars.index)


def test_em_wide_spread_flips_strategy_profitability():
    """The same planted momentum alpha nets positive under EURUSD session
    spreads and negative under EM-style spreads (cost sensitivity)."""
    ticks = generate_ticks(n_days=30, phi=0.25, seed=0)
    bars = build_bars(ticks, 1.0)
    pos = momentum_positions(bars)

    major_cfg = BacktestConfig(pip_size=PIP, spread_pips_by_session=EURUSD.spread_pips)
    em_cfg = BacktestConfig(pip_size=PIP, spread_pips_by_session=EM_SPREADS)
    _, s_major = IntradayBacktester(major_cfg).run(bars, pos)
    _, s_em = IntradayBacktester(em_cfg).run(bars, pos)

    assert s_major["gross_pips"] == pytest.approx(s_em["gross_pips"])  # same alpha
    assert s_major["net_pips"] > 0
    assert s_em["net_pips"] < 0


def test_liquidity_weighted_beats_twap_controllable_cost():
    """The session-aware schedule exploits the depth/spread profile: the
    controllable (spread + impact) cost is strictly below naive TWAP.
    This component is deterministic given the schedule, so one seed
    suffices; drift noise is common to both."""
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    lw = liquidity_weighted_schedule(500.0, sim.depth_bucket)
    tw = twap_schedule(500.0, sim.n_buckets)

    def ctrl(sched):
        d = decompose_implementation_shortfall(sim.execute(sched, FirmVenue(), seed=0))
        return d["spread_temporary"] + d["permanent_impact"]

    assert ctrl(lw) < 0.95 * ctrl(tw)


def test_500mm_parent_requires_multi_session_split():
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    # all 500mm in the single deepest bucket -> depth-cap error
    sched = np.zeros(sim.n_buckets)
    sched[int(np.argmax(sim.depth_bucket))] = 500.0
    with pytest.raises(ValueError, match="depth cap"):
        sim.execute(sched, seed=0)
    # spread across the day it executes fine
    r = sim.execute(liquidity_weighted_schedule(500.0, sim.depth_bucket), seed=0)
    assert r.total_qty == pytest.approx(500.0)


def test_pov_completes_500mm_within_day_at_5pct():
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    q = pov_schedule(500.0, sim.depth_bucket, participation=0.05)
    assert q.sum() == pytest.approx(500.0)
    assert (q <= 0.05 * sim.depth_bucket + 1e-9).all()
    r = sim.execute(q, seed=0)
    assert r.total_qty == pytest.approx(500.0)


def test_em_pair_execution_costs_dwarf_major():
    dt = 5.0
    sim_maj = MarketSimulator(EURUSD, dt_minutes=dt)
    sim_em = MarketSimulator(USDMXN, dt_minutes=dt)
    q_maj = liquidity_weighted_schedule(100.0, sim_maj.depth_bucket)
    q_em = liquidity_weighted_schedule(20.0, sim_em.depth_bucket)

    def ctrl(sim, q):
        d = decompose_implementation_shortfall(sim.execute(q, seed=0))
        return d["spread_temporary"] + d["permanent_impact"]

    assert ctrl(sim_em, q_em) > 20 * ctrl(sim_maj, q_maj)


def test_flash_vol_regime_multiplies_execution_cost():
    """GBP-flash-crash-style regime: a 5x session-vol shock multiplies
    both temporary impact and execution-risk variance."""
    from fx_algo import GBPUSD

    calm = MarketSimulator(GBPUSD, dt_minutes=5.0, vol_scale=1.0)
    flash = MarketSimulator(GBPUSD, dt_minutes=5.0, vol_scale=5.0)
    q = liquidity_weighted_schedule(200.0, calm.depth_bucket)

    def ctrl(sim):
        d = decompose_implementation_shortfall(sim.execute(q, seed=0))
        return d["spread_temporary"] + d["permanent_impact"]

    assert ctrl(flash) > 2 * ctrl(calm)
    is_calm = np.std([calm.execute(q, seed=s).is_pips for s in range(30)])
    is_flash = np.std([flash.execute(q, seed=s).is_pips for s in range(30)])
    assert is_flash > 3 * is_calm


def test_zero_vol_execution_fully_deterministic():
    sim = MarketSimulator(EURUSD, dt_minutes=5.0, vol_scale=0.0)
    q = twap_schedule(100.0, sim.n_buckets)
    a = sim.execute(q, seed=0)
    b = sim.execute(q, seed=999)
    assert np.array_equal(a.fills, b.fills)  # seed-independent when vol=0


def test_session_filtered_strategy_cuts_costs():
    """Restricting trading to liquid sessions reduces cost per unit of
    turnover (spread paid is lower where we trade)."""
    from fx_algo import session_filter

    ticks = generate_ticks(n_days=30, phi=0.25, seed=0)
    bars = build_bars(ticks, 1.0)
    pos = momentum_positions(bars)
    pos_f = session_filter(pos, bars["hour"], ("london", "overlap", "ny"))
    cfg = BacktestConfig(pip_size=PIP, spread_pips_by_session=EURUSD.spread_pips)
    bt = IntradayBacktester(cfg)
    _, s_all = bt.run(bars, pos)
    _, s_filt = bt.run(bars, pos_f)
    assert s_filt["cost_pips"] / max(s_filt["turnover"], 1e-9) < s_all["cost_pips"] / s_all["turnover"]
