"""Zero-volume / one-tick market edge cases, NaN-Inf rejection, and the
Almgren-Chriss high-urgency numerical-stability regression."""

import numpy as np
import pytest

from eq_algo import (ACParams, IntradayConfig, IntradayMarket, ac_trades,
                     ac_trajectory, pov_schedule, twap_schedule, vwap_schedule)
from eq_algo.benchmarks import benchmark_slippage, twap, vwap
from eq_algo.tca import is_decomposition, tca_report


# ---------------------------------------------------------------------------
# Almgren-Chriss numerical stability
# ---------------------------------------------------------------------------

def test_ac_extreme_risk_aversion_does_not_overflow_to_nan():
    """Regression: sinh(kappa*T) overflows past kappa*T ~ 710, and the naive
    ratio then returned inf/inf = NaN for the whole schedule."""
    p = ACParams(total_shares=1_000_000.0, n_slices=26, sigma=1.0, eta=1e-6)
    for lam in (1e6, 1e12, 1e18, 1e30):
        x = ac_trajectory(p, lam)
        n = ac_trades(p, lam)
        assert np.all(np.isfinite(x)), f"non-finite trajectory at lam={lam}"
        assert np.all(np.isfinite(n))
        assert x[0] == pytest.approx(p.total_shares, rel=1e-12)
        assert x[-1] == 0.0
        assert np.all(np.diff(x) <= 1e-9)                 # monotone liquidation
        assert n.sum() == pytest.approx(p.total_shares, rel=1e-9)
        assert np.all(n >= -1e-9)                         # no buying back


def test_ac_infinite_urgency_limit_dumps_first_slice():
    """lambda -> inf must converge to 'all in the first slice', not NaN."""
    p = ACParams(total_shares=50_000.0, n_slices=20, sigma=1.0, eta=1e-6)
    n = ac_trades(p, 1e20)
    assert n[0] / p.total_shares > 0.999
    assert n[1:].sum() / p.total_shares < 1e-3


def test_ac_front_loading_monotone_in_risk_aversion():
    """Property: the fraction done in the first slice is non-decreasing in
    lambda across the whole numerically representable range."""
    p = ACParams(total_shares=10_000.0, n_slices=13, sigma=1.0, eta=1e-6)
    lams = [0.0, 1e-6, 1e-4, 1e-2, 1.0, 1e3, 1e9, 1e18]
    first = [ac_trades(p, lam)[0] for lam in lams]
    assert np.all(np.diff(first) >= -1e-9)


# ---------------------------------------------------------------------------
# Zero-volume and one-tick markets
# ---------------------------------------------------------------------------

def test_schedule_into_zero_volume_bucket_raises_informatively():
    cfg = IntradayConfig(n_buckets=4, price_noise=0.0)
    mkt = IntradayMarket(cfg)
    vols = np.array([1000.0, 0.0, 1000.0, 1000.0])
    with pytest.raises(ValueError, match="zero market volume"):
        mkt.execute(np.array([100.0, 100.0, 100.0, 100.0]), side=1, seed=0,
                    market_volumes=vols)


def test_fully_halted_day_zero_schedule_is_legal_but_has_no_vwap():
    """A day with no volume at all: an empty schedule is accepted (nothing was
    traded) but VWAP is undefined and must say so rather than divide by zero."""
    cfg = IntradayConfig(n_buckets=4, price_noise=0.0)
    mkt = IntradayMarket(cfg)
    res = mkt.execute(np.zeros(4), side=1, seed=0, market_volumes=np.zeros(4))
    assert res.filled_qty == 0.0
    with pytest.raises(ValueError, match="average price undefined"):
        _ = res.avg_price
    with pytest.raises(ValueError, match="total volume must be > 0"):
        vwap(res.fills["mid"].to_numpy(), res.fills["market_volume"].to_numpy())
    # TWAP is still well defined on a halted tape (mids exist).
    assert np.isfinite(twap(res.fills["mid"].to_numpy()))


def test_pov_on_all_zero_volume_day_raises_with_capacity_zero():
    with pytest.raises(ValueError, match="cannot complete"):
        pov_schedule(1000.0, np.zeros(10), participation=0.2)


def test_one_tick_market_all_volume_in_one_bucket():
    """Degenerate one-print day: the whole day's volume trades in one bucket.
    VWAP collapses to that print, and a VWAP schedule routes everything there."""
    cfg = IntradayConfig(n_buckets=5, price_noise=0.0, temp_coef=0.0,
                         perm_coef=0.0)
    mkt = IntradayMarket(cfg)
    vols = np.array([0.0, 0.0, 10_000.0, 0.0, 0.0])
    q = vwap_schedule(1_000.0, vols)
    assert q[2] == pytest.approx(1_000.0)
    assert q.sum() == pytest.approx(1_000.0)
    res = mkt.execute(q, side=1, seed=0, market_volumes=vols)
    s = benchmark_slippage(res)
    # Only one print on the tape, and we trade in that bucket at mid+half
    # spread, so slippage vs VWAP is exactly the half-spread in bps.
    assert s["vs_vwap_bps"] == pytest.approx(cfg.spread_bps / 2.0, rel=1e-9)


def test_single_bucket_day_vwap_equals_twap_equals_arrival():
    cfg = IntradayConfig(n_buckets=1, price_noise=0.0, temp_coef=0.0,
                         perm_coef=0.0)
    mkt = IntradayMarket(cfg)
    res = mkt.execute(np.array([500.0]), side=-1, seed=3)
    s = benchmark_slippage(res)
    assert s["vs_vwap_bps"] == pytest.approx(s["vs_twap_bps"], rel=1e-12)
    assert s["vs_vwap_bps"] == pytest.approx(s["vs_arrival_bps"], rel=1e-12)
    # Selling: cost is the half spread, positive by the sign convention.
    assert s["vs_arrival_bps"] == pytest.approx(cfg.spread_bps / 2.0, rel=1e-9)


def test_zero_volatility_market_isolates_pure_impact_cost():
    """sigma_daily = 0 kills noise AND both impact terms (they scale with
    sigma), leaving exactly the half-spread as the only cost."""
    cfg = IntradayConfig(n_buckets=6, sigma_daily=0.0, spread_bps=8.0)
    mkt = IntradayMarket(cfg)
    res = mkt.execute(twap_schedule(6_000.0, 6), side=1, seed=0)
    rep = tca_report(res)
    assert rep.bps()["trading_bps"] == pytest.approx(8.0 / 2.0, rel=1e-9)
    assert rep.bps()["opportunity_bps"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# NaN / Inf rejection
# ---------------------------------------------------------------------------

def test_nan_schedule_rejected_not_silently_filled():
    """Regression: NaN passed every comparison (NaN < 0 is False) and produced
    NaN fill prices and a NaN average execution price."""
    cfg = IntradayConfig(n_buckets=5, price_noise=0.0)
    mkt = IntradayMarket(cfg)
    bad = np.array([np.nan, 100.0, 100.0, 100.0, 100.0])
    with pytest.raises(ValueError, match="NaN or infinite"):
        mkt.execute(bad, side=1, seed=0)
    with pytest.raises(ValueError, match="NaN or infinite"):
        mkt.execute(np.array([np.inf, 1.0, 1.0, 1.0, 1.0]), side=1, seed=0)


def test_nan_market_volumes_rejected():
    cfg = IntradayConfig(n_buckets=3, price_noise=0.0)
    mkt = IntradayMarket(cfg)
    with pytest.raises(ValueError, match="NaN or infinite"):
        mkt.execute(np.array([10.0, 10.0, 10.0]), side=1, seed=0,
                    market_volumes=np.array([100.0, np.nan, 100.0]))


def test_benchmarks_reject_non_finite_tapes():
    with pytest.raises(ValueError, match="finite"):
        vwap(np.array([100.0, np.nan]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        vwap(np.array([100.0, 101.0]), np.array([1.0, np.inf]))
    with pytest.raises(ValueError, match="finite"):
        twap(np.array([100.0, np.nan]))
    with pytest.raises(ValueError, match="finite"):
        vwap_schedule(100.0, np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="finite"):
        pov_schedule(100.0, np.array([1000.0, np.nan]), participation=0.5)


def test_is_decomposition_rejects_non_finite_fills():
    with pytest.raises(ValueError, match="finite"):
        is_decomposition(1, 100.0, 10.0, 10.0, 10.0, [50.0, np.nan], [10.0, 10.0])
    with pytest.raises(ValueError, match="finite"):
        is_decomposition(1, 100.0, 10.0, 10.0, 10.0, [50.0, 50.0], [10.0, np.inf])


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def test_is_components_sum_to_total_on_partial_fill():
    """Perold identity must hold exactly even with an unfilled tail."""
    rep = is_decomposition(side=1, parent_qty=1000.0, decision_price=99.5,
                           arrival_price=100.0, final_price=101.5,
                           fill_qty=[300.0, 250.0], fill_price=[100.2, 100.9])
    assert rep.delay_cost + rep.trading_cost + rep.opportunity_cost == \
        pytest.approx(rep.total_is, abs=1e-9)
    b = rep.bps()
    assert b["delay_bps"] + b["trading_bps"] + b["opportunity_bps"] == \
        pytest.approx(b["total_is_bps"], abs=1e-9)


def test_buy_and_sell_costs_are_sign_symmetric():
    """Same absolute price moves must give the same signed cost for a buy and
    a sell (the side factor is the only difference)."""
    buy = is_decomposition(1, 1000.0, 100.0, 100.0, 100.0,
                           [1000.0], [100.5])
    sell = is_decomposition(-1, 1000.0, 100.0, 100.0, 100.0,
                            [1000.0], [99.5])
    assert buy.total_is == pytest.approx(sell.total_is, rel=1e-12)
    assert buy.total_is > 0  # both paid 50bp of cost


def test_larger_orders_cost_more_monotonicity():
    """Square-root impact: cost in bps is non-decreasing in parent size."""
    cfg = IntradayConfig(n_buckets=10, price_noise=0.0)
    mkt = IntradayMarket(cfg)
    costs = []
    for qty in (10_000.0, 50_000.0, 100_000.0, 200_000.0):
        res = mkt.execute(twap_schedule(qty, 10), side=1, seed=0)
        costs.append(benchmark_slippage(res)["vs_arrival_bps"])
    assert np.all(np.diff(costs) > 0)
