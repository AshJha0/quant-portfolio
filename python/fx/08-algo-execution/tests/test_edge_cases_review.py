"""Edge-case and property tests added in the review pass (project 08).

Focus, per the FX execution domain:

* illiquid crosses (USDMXN in Asia: 150-pip spreads, 0.5mm/min depth),
* session-time effects (overlap vs late, fix window, weekend gap),
* one-sided books / one-sided execution (no sell-back on a buy parent),
* degenerate parents (zero quantity, single bucket) and NaN/Inf rejection.
"""

import numpy as np
import pytest

from fx_algo.execution.optimal import (
    ac_closed_form_schedule,
    ac_expected_cost,
    eta_from_depth,
    piecewise_ac_schedule,
)
from fx_algo.execution.schedulers import (
    fix_schedule,
    liquidity_weighted_schedule,
    pov_schedule,
    twap_schedule,
)
from fx_algo.sessions import (
    EURUSD,
    GBPUSD,
    SESSION_BOUNDS,
    USDMXN,
    fix_window_mask,
    make_time_grid,
    session_of_hour,
    weekend_mask,
)


# ---------------------------------------------------------------------------
# Illiquid crosses
# ---------------------------------------------------------------------------

def test_em_cross_is_an_order_of_magnitude_worse_than_major():
    """USDMXN in Asia vs EURUSD in the overlap: the illiquidity gap."""
    assert USDMXN.spread_pips["asia"] / EURUSD.spread_pips["overlap"] > 100
    assert EURUSD.depth_mm_per_min["overlap"] / USDMXN.depth_mm_per_min["asia"] > 100


def test_illiquid_pair_session_ordering_differs_from_major():
    """EURUSD is deepest in the overlap; USDMXN is deepest in its NY home session."""
    assert max(EURUSD.depth_mm_per_min, key=EURUSD.depth_mm_per_min.get) == "overlap"
    assert max(USDMXN.depth_mm_per_min, key=USDMXN.depth_mm_per_min.get) == "ny"


def test_every_pair_is_thinnest_and_widest_in_the_late_session():
    for pair in (EURUSD, GBPUSD, USDMXN):
        assert pair.depth_mm_per_min["late"] == min(pair.depth_mm_per_min.values())
        assert pair.spread_pips["late"] == max(pair.spread_pips.values())


def test_liquidity_weighted_schedule_avoids_the_asian_desert_on_em_cross():
    """On USDMXN the schedule must concentrate away from the Asia session."""
    grid = make_time_grid(0.0, 24.0, 60.0)
    depths = USDMXN.depth_at(grid)
    q = liquidity_weighted_schedule(100.0, depths)
    sess = session_of_hour(grid)
    asia_share = q[sess == "asia"].sum() / q.sum()
    ny_share = q[sess == "ny"].sum() / q.sum()
    # Asia is 7/24 of the clock but must take far less than its time share
    assert asia_share < 7.0 / 24.0 / 2
    assert ny_share > asia_share


def test_ac_schedule_backloads_into_deep_buckets_when_risk_neutral():
    """lambda -> 0: n_j proportional to 1/eta_j = depth_j."""
    depths = np.array([0.5, 3.0, 5.0, 6.0, 0.3])
    eta = eta_from_depth(depths)
    n = piecewise_ac_schedule(100.0, eta, np.ones(5), risk_aversion=0.0)
    assert np.allclose(n / n.sum(), depths / depths.sum(), atol=1e-8)


def test_illiquid_bucket_gets_least_quantity():
    depths = np.array([0.5, 3.0, 5.0, 6.0, 0.3])
    eta = eta_from_depth(depths)
    n = piecewise_ac_schedule(100.0, eta, np.ones(5), risk_aversion=0.0)
    assert int(np.argmin(n)) == int(np.argmin(depths))


# ---------------------------------------------------------------------------
# Session-time effects
# ---------------------------------------------------------------------------

def test_session_bounds_partition_the_day_without_gaps_or_overlaps():
    edges = sorted(SESSION_BOUNDS.values())
    assert edges[0][0] == 0.0
    assert edges[-1][1] == 24.0
    for (_, prev_hi), (next_lo, _) in zip(edges, edges[1:]):
        assert prev_hi == next_lo


def test_every_hour_of_the_day_maps_to_exactly_one_session():
    hours = np.arange(0.0, 24.0, 0.25)
    sess = session_of_hour(hours)
    assert set(sess) == set(SESSION_BOUNDS)
    assert not np.any(sess == "None")


def test_session_boundaries_are_start_inclusive_end_exclusive():
    assert session_of_hour(7.0)[0] == "london"
    assert session_of_hour(6.999)[0] == "asia"
    assert session_of_hour(12.0)[0] == "overlap"
    assert session_of_hour(21.0)[0] == "late"


def test_session_wraps_over_multiday_grid():
    """Absolute hours beyond 24 wrap to the same session as their hour-of-day."""
    assert session_of_hour(25.0)[0] == session_of_hour(1.0)[0]
    assert session_of_hour(48.0 + 13.0)[0] == "overlap"


def test_fix_window_is_five_one_minute_buckets_inside_the_overlap():
    grid = make_time_grid(0.0, 24.0, 1.0)
    mask = fix_window_mask(grid, dt_minutes=1.0)
    assert mask.sum() == 5
    assert set(session_of_hour(grid[mask])) == {"overlap"}


def test_fix_schedule_places_everything_in_the_window():
    grid = make_time_grid(0.0, 24.0, 1.0)
    q = fix_schedule(50.0, grid, dt_minutes=1.0)
    mask = fix_window_mask(grid, dt_minutes=1.0)
    assert q.sum() == pytest.approx(50.0)
    assert q[~mask].sum() == pytest.approx(0.0)
    assert np.allclose(q[mask], 10.0)


def test_fix_schedule_raises_when_horizon_misses_the_fix():
    grid = make_time_grid(0.0, 4.0, 1.0)  # 00:00-04:00, no 16:00
    with pytest.raises(ValueError, match="fix window"):
        fix_schedule(10.0, grid, dt_minutes=1.0)


def test_weekend_gap_gets_zero_quantity_and_the_rest_still_sums():
    grid = make_time_grid(0.0, 48.0, 60.0)
    tradeable = weekend_mask(grid, weekend_start_hour=20.0, weekend_end_hour=30.0)
    q = twap_schedule(120.0, len(grid), tradeable)
    assert q[~tradeable].sum() == pytest.approx(0.0)
    assert q.sum() == pytest.approx(120.0)
    assert np.all(q[tradeable] > 0)


def test_all_buckets_untradeable_raises():
    with pytest.raises(ValueError, match="no tradeable buckets"):
        twap_schedule(10.0, 5, np.zeros(5, dtype=bool))


# ---------------------------------------------------------------------------
# One-sided books / one-sided execution
# ---------------------------------------------------------------------------

def test_buy_parent_never_sells_back_even_with_extreme_risk_aversion():
    """One-sided constraint: a buy schedule must be non-negative throughout."""
    depths = np.array([6.0, 0.2, 0.2, 6.0])
    eta = eta_from_depth(depths)
    sigma = np.array([1.0, 5.0, 5.0, 1.0])
    n = piecewise_ac_schedule(100.0, eta, sigma, risk_aversion=50.0)
    assert np.all(n >= -1e-9)
    assert n.sum() == pytest.approx(100.0)


def test_sell_parent_is_the_exact_mirror_of_the_buy_parent():
    depths = np.array([6.0, 0.5, 3.0, 6.0])
    eta = eta_from_depth(depths)
    sigma = np.array([1.0, 2.0, 1.5, 1.0])
    buy = piecewise_ac_schedule(100.0, eta, sigma, risk_aversion=1.0)
    sell = piecewise_ac_schedule(-100.0, eta, sigma, risk_aversion=1.0)
    assert np.allclose(buy, -sell, atol=1e-12)
    assert np.all(sell <= 1e-9)


def test_allow_sells_can_trade_against_the_parent_but_still_sums():
    depths = np.array([6.0, 0.2, 0.2, 6.0])
    eta = eta_from_depth(depths)
    sigma = np.array([1.0, 5.0, 5.0, 1.0])
    n = piecewise_ac_schedule(100.0, eta, sigma, risk_aversion=50.0, allow_sells=True)
    assert n.sum() == pytest.approx(100.0)


def test_one_sided_schedule_costs_no_less_than_the_unconstrained_optimum():
    """Constraining sign-flips cannot improve the unconstrained objective."""
    depths = np.array([6.0, 0.2, 0.2, 6.0])
    eta = eta_from_depth(depths)
    sigma = np.array([1.0, 5.0, 5.0, 1.0])
    lam = 50.0
    free = piecewise_ac_schedule(100.0, eta, sigma, lam, allow_sells=True)
    onesided = piecewise_ac_schedule(100.0, eta, sigma, lam, allow_sells=False)
    assert ac_expected_cost(onesided, eta, sigma, lam) >= ac_expected_cost(
        free, eta, sigma, lam
    ) - 1e-9


# ---------------------------------------------------------------------------
# Degenerate parents and grids
# ---------------------------------------------------------------------------

def test_zero_parent_quantity_returns_a_flat_zero_schedule():
    """A zero parent used to return an all-NaN schedule (0/0 renormalisation)."""
    n = piecewise_ac_schedule(0.0, np.array([1.0, 2.0, 3.0]), np.ones(3), 1e-6)
    assert np.all(n == 0.0)
    assert np.all(np.isfinite(n))


def test_single_bucket_takes_the_whole_parent():
    assert piecewise_ac_schedule(42.0, np.array([1.0]), np.array([1.0]), 1.0) == pytest.approx(
        np.array([42.0])
    )
    assert ac_closed_form_schedule(42.0, 1, 1.0, 1.0, 1.0) == pytest.approx(np.array([42.0]))


def test_zero_risk_aversion_is_exactly_twap_under_constant_liquidity():
    n = ac_closed_form_schedule(100.0, 8, eta=1.0, sigma=1.0, risk_aversion=0.0)
    assert np.allclose(n, 12.5)


def test_zero_vol_is_exactly_twap_under_constant_liquidity():
    """sigma = 0 (a dead, pegged market) removes the timing-risk term."""
    n = ac_closed_form_schedule(100.0, 8, eta=1.0, sigma=0.0, risk_aversion=5.0)
    assert np.allclose(n, 12.5)


def test_schedulers_reject_zero_parent_quantity():
    for fn in (
        lambda: twap_schedule(0.0, 5),
        lambda: liquidity_weighted_schedule(0.0, np.ones(5)),
        lambda: pov_schedule(0.0, np.ones(5), 0.5),
    ):
        with pytest.raises(ValueError, match="non-zero"):
            fn()


def test_pov_raises_when_the_horizon_cannot_absorb_the_parent():
    """Illiquid cross + low participation = incomplete execution, stated loudly."""
    with pytest.raises(ValueError, match="incomplete"):
        pov_schedule(100.0, np.full(3, 1.0), participation=0.1)


# ---------------------------------------------------------------------------
# NaN / Inf rejection
# ---------------------------------------------------------------------------

def test_session_lookup_rejects_nonfinite_hours():
    """A NaN timestamp used to yield the bogus session label 'None'."""
    with pytest.raises(ValueError, match="NaN/Inf"):
        session_of_hour(np.array([np.nan, 5.0]))
    with pytest.raises(ValueError, match="NaN/Inf"):
        session_of_hour(np.array([np.inf]))


def test_pair_profile_lookup_rejects_nonfinite_hours():
    with pytest.raises(ValueError, match="NaN/Inf"):
        EURUSD.spread_pips_at(np.array([np.nan]))
    with pytest.raises(ValueError, match="NaN/Inf"):
        EURUSD.depth_at(np.array([np.inf]))


def test_eta_from_depth_rejects_nonfinite_and_nonpositive_depth():
    with pytest.raises(ValueError, match="finite"):
        eta_from_depth(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="positive"):
        eta_from_depth(np.array([1.0, 0.0]))


def test_ac_schedules_reject_nonfinite_parameters():
    with pytest.raises(ValueError, match="finite"):
        piecewise_ac_schedule(10.0, np.array([1.0, np.nan]), np.ones(2), 1e-6)
    with pytest.raises(ValueError, match="finite"):
        piecewise_ac_schedule(10.0, np.ones(2), np.array([1.0, np.nan]), 1e-6)
    with pytest.raises(ValueError, match="finite"):
        piecewise_ac_schedule(np.nan, np.ones(2), np.ones(2), 1e-6)
    with pytest.raises(ValueError, match="finite"):
        ac_closed_form_schedule(10.0, 5, eta=1.0, sigma=np.nan, risk_aversion=1e-6)


def test_liquidity_weighted_schedule_rejects_nan_depths():
    with pytest.raises(ValueError, match="finite"):
        liquidity_weighted_schedule(10.0, np.array([1.0, np.nan, 3.0]))


def test_schedulers_reject_nonfinite_parent_quantity():
    with pytest.raises(ValueError, match="finite"):
        twap_schedule(np.nan, 5)
    with pytest.raises(ValueError, match="finite"):
        twap_schedule(np.inf, 5)
