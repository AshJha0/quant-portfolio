"""TCA tests: Perold IS decomposition identity (exact), hand-checked toy
order, attribution consistency, aggregation."""

import numpy as np
import pandas as pd
import pytest

from eq_algo import (IntradayConfig, IntradayMarket, aggregate_tca,
                     is_decomposition, slippage_attribution, tca_report,
                     twap_schedule)


def test_is_components_sum_to_total_exactly():
    rng = np.random.default_rng(0)
    for _ in range(25):
        X = rng.uniform(1_000, 100_000)
        filled_frac = rng.uniform(0.3, 1.0)
        nfills = rng.integers(1, 10)
        q = rng.uniform(0, 1, nfills)
        q = q / q.sum() * X * filled_frac
        p = rng.uniform(90, 110, nfills)
        side = int(rng.choice([1, -1]))
        rep = is_decomposition(side, X, decision_price=rng.uniform(90, 110),
                               arrival_price=rng.uniform(90, 110),
                               final_price=rng.uniform(90, 110),
                               fill_qty=q, fill_price=p)
        assert rep.delay_cost + rep.trading_cost + rep.opportunity_cost == \
            pytest.approx(rep.total_is, abs=1e-10 * max(1.0, abs(rep.total_is)))
        b = rep.bps()
        assert b["delay_bps"] + b["trading_bps"] + b["opportunity_bps"] == \
            pytest.approx(b["total_is_bps"], abs=1e-8)


def test_is_decomposition_hand_checked_toy_order():
    # buy 100; decide at 10.00, arrive at 10.10; fill 60@10.20 and 20@10.30;
    # close 10.50 with 20 unfilled
    rep = is_decomposition(side=1, parent_qty=100.0, decision_price=10.0,
                           arrival_price=10.1, final_price=10.5,
                           fill_qty=[60.0, 20.0], fill_price=[10.2, 10.3])
    assert rep.delay_cost == pytest.approx(10.0, abs=1e-12)        # 0.10 * 100
    assert rep.trading_cost == pytest.approx(10.0, abs=1e-12)      # 6 + 4
    assert rep.opportunity_cost == pytest.approx(8.0, abs=1e-12)   # 20 * 0.40
    assert rep.total_is == pytest.approx(28.0, abs=1e-12)
    assert rep.bps()["total_is_bps"] == pytest.approx(280.0, abs=1e-9)
    assert rep.avg_fill_price == pytest.approx(10.225, abs=1e-12)


def test_is_decomposition_sell_side_signs():
    # sell decided at 100, arrival dropped to 99 -> delay cost is positive
    rep = is_decomposition(side=-1, parent_qty=10.0, decision_price=100.0,
                           arrival_price=99.0, final_price=98.0,
                           fill_qty=[10.0], fill_price=[98.5])
    assert rep.delay_cost == pytest.approx(10.0, abs=1e-12)
    assert rep.trading_cost == pytest.approx(5.0, abs=1e-12)   # sold 0.5 under arrival
    assert rep.opportunity_cost == 0.0
    assert rep.total_is == pytest.approx(15.0, abs=1e-12)


def test_tca_report_on_simulated_order_identity():
    cfg = IntradayConfig(vol_noise=0.2, temp_coef=0.8)
    mkt = IntradayMarket(cfg)
    res = mkt.execute(twap_schedule(40_000.0, cfg.n_buckets), side=1, seed=3,
                      decision_price=99.7)
    rep = tca_report(res)
    assert rep.filled_qty == pytest.approx(40_000.0, rel=1e-12)
    assert rep.opportunity_cost == 0.0                     # fully filled
    assert rep.delay_cost + rep.trading_cost == pytest.approx(rep.total_is, abs=1e-8)
    # total IS ties out against avg price directly (fully filled case)
    direct = (rep.avg_fill_price - 99.7) * 40_000.0
    assert rep.total_is == pytest.approx(direct, rel=1e-10)


def test_slippage_attribution_components_sum_exactly():
    cfg = IntradayConfig(vol_noise=0.2, temp_coef=0.8)
    mkt = IntradayMarket(cfg)
    res = mkt.execute(twap_schedule(40_000.0, cfg.n_buckets), side=1, seed=8)
    att = slippage_attribution(res)
    per_bucket = att.drop(index="TOTAL")
    np.testing.assert_allclose(
        per_bucket["drift"] + per_bucket["spread"] + per_bucket["temporary"],
        per_bucket["total"], atol=1e-10)
    tot = att.loc["TOTAL"]
    assert tot["drift"] + tot["spread"] + tot["temporary"] == \
        pytest.approx(tot["total"], abs=1e-10)
    # qty-weighted total equals average fill slippage per share
    assert tot["total"] == pytest.approx(res.avg_price - res.arrival_price,
                                         rel=1e-10)


def test_aggregate_tca_stats():
    reps = []
    cfg = IntradayConfig(vol_noise=0.1)
    mkt = IntradayMarket(cfg)
    sched = twap_schedule(20_000.0, cfg.n_buckets)
    for s in range(10):
        reps.append(tca_report(mkt.execute(sched, side=1, seed=s,
                                           decision_price=99.9)))
    agg = aggregate_tca(reps)
    assert list(agg.index) == ["delay", "trading", "opportunity", "total_is"]
    manual_mean = np.mean([r.bps()["total_is_bps"] for r in reps])
    assert agg.loc["total_is", "mean"] == pytest.approx(manual_mean, rel=1e-12)
    # equal notionals -> weighted mean equals plain mean
    assert agg.loc["total_is", "notional_weighted"] == \
        pytest.approx(manual_mean, rel=1e-12)


def test_tca_validation():
    with pytest.raises(ValueError, match="overfill"):
        is_decomposition(1, 10.0, 100.0, 100.0, 100.0, [11.0], [100.0])
    with pytest.raises(ValueError):
        is_decomposition(2, 10.0, 100.0, 100.0, 100.0, [1.0], [100.0])
    with pytest.raises(ValueError):
        is_decomposition(1, 10.0, -1.0, 100.0, 100.0, [1.0], [100.0])
    with pytest.raises(ValueError):
        is_decomposition(1, 10.0, 100.0, 100.0, 100.0, [-1.0], [100.0])
    with pytest.raises(ValueError):
        aggregate_tca([])
    cfg = IntradayConfig()
    res = IntradayMarket(cfg).execute(np.zeros(cfg.n_buckets), side=1, seed=0)
    with pytest.raises(ValueError, match="no filled buckets"):
        slippage_attribution(res)
