"""Benchmark tests: hand-computed VWAP/TWAP on toy tapes, exact IS vs
arrival, scheduler properties (VWAP tracks profile, POV respects cap)."""

import numpy as np
import pytest

from eq_algo import (IntradayConfig, IntradayMarket, arrival_price,
                     benchmark_slippage, pov_schedule, slippage_bps, twap,
                     twap_schedule, u_shaped_profile, vwap, vwap_schedule)


def test_vwap_hand_computed():
    p = np.array([10.0, 11.0, 12.0])
    v = np.array([1.0, 2.0, 1.0])
    assert vwap(p, v) == pytest.approx(11.0, abs=1e-12)


def test_twap_and_arrival_hand_computed():
    p = np.array([10.0, 11.0, 12.0])
    assert twap(p) == pytest.approx(11.0, abs=1e-12)
    assert arrival_price(p) == pytest.approx(10.0, abs=1e-12)


def test_slippage_sign_convention():
    assert slippage_bps(101.0, 100.0, side=1) == pytest.approx(100.0, abs=1e-9)
    assert slippage_bps(101.0, 100.0, side=-1) == pytest.approx(-100.0, abs=1e-9)
    assert slippage_bps(99.0, 100.0, side=-1) == pytest.approx(100.0, abs=1e-9)


def test_is_decision_to_fill_exact_on_toy_order():
    """Deterministic market with no impact and no noise: slippage vs arrival
    is exactly the half-spread; vs a lower decision price it adds the drift."""
    cfg = IntradayConfig(mid0=100.0, spread_bps=6.0, temp_coef=0.0,
                         perm_coef=0.0, price_noise=0.0, n_buckets=8)
    mkt = IntradayMarket(cfg)
    res = mkt.execute(twap_schedule(8000.0, 8), side=1, seed=0,
                      decision_price=99.5)
    s = benchmark_slippage(res)
    assert s["vs_arrival_bps"] == pytest.approx(3.0, rel=1e-10)  # half of 6 bps
    # decision->arrival drift: (100 - 99.5)/99.5 plus the half spread on 100
    expected = (res.avg_price - 99.5) / 99.5 * 1e4
    assert s["vs_decision_bps"] == pytest.approx(expected, rel=1e-12)
    assert res.avg_price == pytest.approx(100.0 * (1 + 3e-4), rel=1e-12)


def test_twap_schedule_equal_and_complete():
    q = twap_schedule(50_000.0, 26)
    assert q.shape == (26,)
    np.testing.assert_allclose(q, 50_000.0 / 26)
    assert q.sum() == pytest.approx(50_000.0, rel=1e-14)


def test_vwap_schedule_tracks_volume_profile():
    prof = u_shaped_profile(26)
    q = vwap_schedule(80_000.0, prof)
    assert q.sum() == pytest.approx(80_000.0, rel=1e-12)
    np.testing.assert_allclose(q / 80_000.0, prof, rtol=1e-12)  # proportional
    # heavier at the open/close than midday, like the market
    assert q[0] > q[13] and q[-1] > q[13]


def test_pov_respects_participation_cap_each_bucket():
    rng = np.random.default_rng(2)
    vols = rng.uniform(10_000, 100_000, 26)
    q = pov_schedule(100_000.0, vols, participation=0.15)
    assert np.all(q <= 0.15 * vols + 1e-9)
    assert q.sum() == pytest.approx(100_000.0, rel=1e-12)
    # front section fully uses the cap until the residual bucket
    filled = np.where(q > 0)[0]
    assert np.all(np.diff(filled) == 1) and filled[0] == 0


def test_pov_infeasible_raises_informative_error():
    vols = np.full(26, 10_000.0)
    with pytest.raises(ValueError, match="Split across days"):
        pov_schedule(100_000.0, vols, participation=0.10)


def test_vwap_validation():
    with pytest.raises(ValueError):
        vwap(np.array([1.0, 2.0]), np.array([0.0, 0.0]))     # zero volume
    with pytest.raises(ValueError):
        vwap(np.array([1.0]), np.array([1.0, 2.0]))          # shape mismatch
    with pytest.raises(ValueError):
        vwap(np.array([1.0, 2.0]), np.array([-1.0, 2.0]))    # negative volume
    with pytest.raises(ValueError):
        twap(np.array([]))


def test_scheduler_validation():
    with pytest.raises(ValueError):
        twap_schedule(0.0, 10)
    with pytest.raises(ValueError):
        twap_schedule(100.0, 0)
    with pytest.raises(ValueError):
        vwap_schedule(100.0, np.array([-0.5, 1.5]))
    with pytest.raises(ValueError):
        pov_schedule(100.0, np.ones(5), participation=0.0)
    with pytest.raises(ValueError):
        pov_schedule(100.0, np.ones(5), participation=1.5)


def test_slippage_bps_validation():
    with pytest.raises(ValueError):
        slippage_bps(100.0, 0.0, 1)
    with pytest.raises(ValueError):
        slippage_bps(100.0, 100.0, 2)


def test_benchmark_slippage_consistency_on_simulated_day():
    """VWAP/TWAP slippage recomputed by hand from the fills frame matches."""
    cfg = IntradayConfig(vol_noise=0.2, temp_coef=0.5)
    mkt = IntradayMarket(cfg)
    res = mkt.execute(twap_schedule(30_000.0, cfg.n_buckets), side=1, seed=17)
    s = benchmark_slippage(res)
    mids = res.fills["mid"].to_numpy()
    vols = res.fills["market_volume"].to_numpy()
    vw = (mids * vols).sum() / vols.sum()
    assert s["vs_vwap_bps"] == pytest.approx(
        (res.avg_price - vw) / vw * 1e4, rel=1e-12)
    assert s["vs_twap_bps"] == pytest.approx(
        (res.avg_price - mids.mean()) / mids.mean() * 1e4, rel=1e-12)
