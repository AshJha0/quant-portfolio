"""Last-look dealer model: monotonicity, the spread trap, attribution."""

import numpy as np
import pytest

from fx_algo import (
    EURUSD,
    FirmVenue,
    LastLookVenue,
    MarketSimulator,
    last_look_reject_prob,
    liquidity_weighted_schedule,
    rejection_cost_pips,
    venue_comparison,
)


def paired_costs(alpha, n_rep=60, seed0=0):
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    lw = liquidity_weighted_schedule(500.0, sim.depth_bucket)
    ll, firm, rej = [], [], []
    for s in range(seed0, seed0 + n_rep):
        rl = sim.execute(lw, LastLookVenue(), seed=s, alpha_pips_per_bucket=alpha)
        rf = sim.execute(lw, FirmVenue(), seed=s, alpha_pips_per_bucket=alpha)
        vc = venue_comparison({"ll": rl, "firm": rf})
        ll.append(vc["ll"]["effective_cost_pips"])
        firm.append(vc["firm"]["effective_cost_pips"])
        rej.append(rl.rejection_rate)
    return np.array(ll), np.array(firm), np.array(rej)


def test_reject_prob_monotone_increasing_in_adverse_move():
    moves = np.linspace(-3.0, 3.0, 201)
    p = last_look_reject_prob(moves)
    assert np.all(np.diff(p) > 0)
    assert p[0] < 0.01 and p[-1] > 0.99
    assert last_look_reject_prob(0.6)[()] == pytest.approx(0.5)  # threshold


def test_reject_prob_invalid_sharpness():
    with pytest.raises(ValueError):
        last_look_reject_prob(0.0, sharpness_pips=0.0)


def test_firm_venue_never_rejects():
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    lw = liquidity_weighted_schedule(500.0, sim.depth_bucket)
    for s in range(5):
        r = sim.execute(lw, FirmVenue(), seed=s, alpha_pips_per_bucket=1.0)
        assert not r.rejected.any()
        assert r.rejection_rate == 0.0
        assert rejection_cost_pips(r) == 0.0


def test_last_look_quoted_spread_is_tighter():
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    lw = liquidity_weighted_schedule(500.0, sim.depth_bucket)
    rl = sim.execute(lw, LastLookVenue(spread_mult=0.6), seed=0)
    rf = sim.execute(lw, FirmVenue(), seed=0)
    vc = venue_comparison({"ll": rl, "firm": rf})
    assert vc["ll"]["quoted_half_spread_pips"] == pytest.approx(
        0.6 * vc["firm"]["quoted_half_spread_pips"], rel=1e-12
    )


def test_the_trap_effective_cost_higher_with_adverse_selection():
    """Both sides of the last-look trap, statistically over replications:
    tighter quote LOOKS cheaper, and IS cheaper for uninformed flow, but
    for informed (alpha) flow the rejections convert the spread saving
    into a net cost."""
    ll0, firm0, rej0 = paired_costs(alpha=0.0)
    d0 = ll0 - firm0
    se0 = d0.std(ddof=1) / np.sqrt(len(d0))
    assert d0.mean() < 0  # uninformed flow: last-look genuinely cheaper
    assert d0.mean() < -3 * se0

    ll1, firm1, rej1 = paired_costs(alpha=0.5)
    d1 = ll1 - firm1
    se1 = d1.std(ddof=1) / np.sqrt(len(d1))
    assert d1.mean() > 3 * se1  # informed flow: last-look strictly worse
    assert rej1.mean() > rej0.mean() + 0.1  # driven by rejections


def test_rejection_rate_increases_with_alpha():
    rates = [paired_costs(alpha=a, n_rep=20)[2].mean() for a in (0.0, 0.5, 1.0)]
    assert rates[0] < rates[1] < rates[2]


def test_rejected_fills_worse_than_original_quote():
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    lw = liquidity_weighted_schedule(500.0, sim.depth_bucket)
    r = sim.execute(lw, LastLookVenue(), seed=4, alpha_pips_per_bucket=0.5)
    assert r.rejected.any()
    worse = r.side * (r.fills[r.rejected] - r.quoted[r.rejected])
    assert (worse > 0).all()
    assert rejection_cost_pips(r) > 0


def test_venue_scorecard_identity_exact():
    sim = MarketSimulator(EURUSD, dt_minutes=5.0)
    lw = liquidity_weighted_schedule(500.0, sim.depth_bucket)
    for venue in (LastLookVenue(), FirmVenue()):
        r = sim.execute(lw, venue, seed=6, alpha_pips_per_bucket=0.5)
        vc = venue_comparison({"v": r})["v"]
        resid = vc["effective_cost_pips"] - (
            vc["quoted_half_spread_pips"]
            + vc["temp_impact_pips"]
            + vc["rejection_cost_pips"]
        )
        assert resid == pytest.approx(0.0, abs=1e-10)


def test_hold_window_scales_with_dt():
    # same hold_seconds on a coarser grid -> smaller share of the bucket
    # move seen in the hold -> fewer rejects for pure-diffusion flow
    sim_fine = MarketSimulator(EURUSD, dt_minutes=1.0)
    sim_coarse = MarketSimulator(EURUSD, dt_minutes=30.0)
    q_fine = liquidity_weighted_schedule(200.0, sim_fine.depth_bucket)
    q_coarse = liquidity_weighted_schedule(200.0, sim_coarse.depth_bucket)
    venue = LastLookVenue(hold_seconds=30.0)
    rej_fine = np.mean(
        [sim_fine.execute(q_fine, venue, seed=s, **{}).rejection_rate for s in range(5)]
    )
    rej_coarse = np.mean(
        [sim_coarse.execute(q_coarse, venue, seed=s, **{}).rejection_rate for s in range(5)]
    )
    assert rej_fine > rej_coarse
