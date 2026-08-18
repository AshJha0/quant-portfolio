"""Intraday simulator tests: profile shape, permanent vs temporary impact,
seeded reproducibility, zero-participation, fill model exactness."""

import numpy as np
import pytest

from eq_algo import IntradayConfig, IntradayMarket, u_shaped_profile


def test_volume_profile_sums_to_one_and_is_u_shaped():
    for n in (13, 26, 78):
        p = u_shaped_profile(n)
        assert p.sum() == pytest.approx(1.0, abs=1e-12)
        assert (p > 0).all()
        mid = p[n // 2]
        assert p[0] > mid and p[-1] > mid          # ends heavier than midday
        assert p[0] == pytest.approx(p[-1], rel=1e-10)  # symmetric


def test_volume_profile_validation():
    with pytest.raises(ValueError):
        u_shaped_profile(0)
    with pytest.raises(ValueError):
        u_shaped_profile(10, curvature=-1.0)


def test_permanent_impact_persists_temporary_reverts():
    """Deterministic mid path (price_noise=0): a burst in bucket 0 shifts all
    later mids by exactly the permanent move, while the half-spread and
    square-root temporary impact appear only in the fill price."""
    cfg = IntradayConfig(mid0=100.0, day_volume=1e6, n_buckets=10,
                         sigma_daily=0.02, spread_bps=4.0, temp_coef=1.0,
                         perm_coef=0.5, price_noise=0.0)
    mkt = IntradayMarket(cfg)
    q = np.zeros(10)
    q[0] = 50_000.0
    res = mkt.execute(q, side=1, seed=0)
    part = 50_000.0 / 1e6
    perm = 0.5 * 0.02 * part * 100.0               # 0.05 currency
    temp = 1.0 * 0.02 * np.sqrt(part) * 100.0      # sqrt law, currency
    half_spread = 100.0 * 4.0e-4 / 2.0
    fill = res.fills["price"].iloc[0]
    assert fill == pytest.approx(100.0 + half_spread + temp, rel=1e-12)
    # every later mid carries the permanent move and nothing else
    np.testing.assert_allclose(res.mids[1:], 100.0 + perm, rtol=1e-12)
    # temporary component reverted: post-trade mid is far below the fill
    assert res.mids[1] < fill - 0.9 * temp


def test_permanent_impact_linear_temporary_sqrt_in_participation():
    cfg = IntradayConfig(price_noise=0.0, n_buckets=4, perm_coef=0.5, temp_coef=1.0)
    mkt = IntradayMarket(cfg)

    def one_shot(x):
        q = np.zeros(4)
        q[0] = x
        r = mkt.execute(q, side=1, seed=0)
        return r.fills["perm_move"].iloc[0], r.fills["temp_cost"].iloc[0]

    p1, t1 = one_shot(10_000.0)
    p4, t4 = one_shot(40_000.0)
    assert p4 == pytest.approx(4.0 * p1, rel=1e-12)   # linear permanent
    assert t4 == pytest.approx(2.0 * t1, rel=1e-12)   # sqrt temporary


def test_zero_participation_zero_impact():
    cfg = IntradayConfig(vol_noise=0.1)
    mkt = IntradayMarket(cfg)
    res = mkt.execute(np.zeros(cfg.n_buckets), side=1, seed=5)
    assert res.filled_qty == 0.0
    assert (res.fills["temp_cost"] == 0.0).all()
    assert (res.fills["perm_move"] == 0.0).all()
    assert (res.fills["half_spread_cost"] == 0.0).all()
    with pytest.raises(ValueError, match="no shares filled"):
        _ = res.avg_price
    # mid path identical to an untouched market with the same seed
    res2 = mkt.execute(np.zeros(cfg.n_buckets), side=1, seed=5)
    np.testing.assert_array_equal(res.mids, res2.mids)


def test_own_trades_move_the_market_vs_untraded_path():
    cfg = IntradayConfig(vol_noise=0.15)
    mkt = IntradayMarket(cfg)
    sched = np.full(cfg.n_buckets, 100_000.0 / cfg.n_buckets)
    traded = mkt.execute(sched, side=1, seed=9)
    quiet = mkt.execute(np.zeros(cfg.n_buckets), side=1, seed=9)
    assert traded.mids[-1] > quiet.mids[-1]           # buys push the close up
    sold = mkt.execute(sched, side=-1, seed=9)
    assert sold.mids[-1] < quiet.mids[-1]             # sells push it down


def test_seeded_reproducibility():
    cfg = IntradayConfig(vol_noise=0.2)
    mkt = IntradayMarket(cfg)
    sched = np.full(cfg.n_buckets, 1000.0)
    a = mkt.execute(sched, side=1, seed=123)
    b = mkt.execute(sched, side=1, seed=123)
    np.testing.assert_array_equal(a.mids, b.mids)
    assert a.fills.equals(b.fills)
    c = mkt.execute(sched, side=1, seed=124)
    assert not np.array_equal(a.mids, c.mids)


def test_zero_volume_bucket_raises_informative_error():
    cfg = IntradayConfig(n_buckets=4)
    mkt = IntradayMarket(cfg)
    q = np.array([100.0, 100.0, 100.0, 100.0])
    tape = np.array([50_000.0, 0.0, 50_000.0, 50_000.0])
    with pytest.raises(ValueError, match="zero market volume"):
        mkt.execute(q, side=1, seed=0, market_volumes=tape)


def test_child_exceeding_bucket_volume_raises():
    cfg = IntradayConfig(n_buckets=4)
    mkt = IntradayMarket(cfg)
    q = np.array([60_000.0, 0.0, 0.0, 0.0])
    tape = np.array([50_000.0, 50_000.0, 50_000.0, 50_000.0])
    with pytest.raises(ValueError, match="participation > 100%"):
        mkt.execute(q, side=1, seed=0, market_volumes=tape)


@pytest.mark.parametrize("bad", [
    lambda m, n: m.execute(np.full(n + 1, 1.0), side=1),        # wrong length
    lambda m, n: m.execute(-np.ones(n), side=1),                # negative qty
    lambda m, n: m.execute(np.ones(n), side=0),                 # bad side
    lambda m, n: m.execute(np.ones(n), side=1,
                           market_volumes=np.ones(n - 1)),      # bad tape len
])
def test_execute_validation(bad):
    cfg = IntradayConfig(n_buckets=6)
    mkt = IntradayMarket(cfg)
    with pytest.raises(ValueError):
        bad(mkt, 6)


@pytest.mark.parametrize("kwargs", [
    {"mid0": -1.0}, {"day_volume": 0.0}, {"n_buckets": 0},
    {"sigma_daily": -0.1}, {"temp_coef": -1.0}, {"price_noise": -0.5},
])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        IntradayConfig(**kwargs)
