"""Fast mode equals event mode to 1e-10 on every fixture (W24 acceptance)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qp_bt import EventEngine, LinearCost, NextBarFill, SqrtImpactCost, WeightPath, run_fast
from qp_core.stats import Financing


def _panel(seed, t_n=120, n=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=t_n)
    p = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (t_n, n)), axis=0))
    prices = pd.DataFrame(p, index=idx, columns=[f"S{i}" for i in range(n)])
    tgt = np.full((t_n, n), np.nan)
    for t in range(0, t_n, int(rng.integers(1, 6))):
        tgt[t] = rng.normal(0, 0.5, n)          # long-short, sometimes levered
    rate = pd.Series(rng.uniform(0.0, 0.06, t_n), index=idx)
    sigma = np.full((t_n, n), 0.02)
    adv = np.full((t_n, n), 5e7)
    return prices, tgt, rate, sigma, adv


FILLS = [NextBarFill(0.5, 1), NextBarFill(1.0, 1), NextBarFill(0.0, 1, 10.0),
         NextBarFill(1.0, 0, 5.0), NextBarFill(0.3, 3)]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("fill", FILLS)
def test_fast_equals_event(seed, fill):
    prices, tgt, rate, sigma, adv = _panel(seed)
    fin = Financing(rate=rate, borrow_spread_bp=50.0, lend_spread_bp=10.0)
    impact = SqrtImpactCost(0.7, 1e8, sigma, adv)

    def cost(bar, dv):
        return LinearCost(5.0)(bar, dv) + impact(bar, dv)

    kw = dict(fill=fill, cost=cost, financing=fin, borrow_bps_annual=75.0)
    fast = run_fast(prices.pct_change(), tgt, **kw)
    event = EventEngine(prices, **kw).run(WeightPath(tgt))
    pd.testing.assert_frame_equal(fast.ledger, event.ledger, rtol=0, atol=1e-10,
                                  check_exact=False)
    np.testing.assert_allclose(fast.weights, event.weights, rtol=0, atol=1e-10)
    # nothing trivially equal: the fixture trades, pays and earns
    assert fast.ledger["cost"].sum() > 0 and fast.ledger["borrow"].sum() > 0
    assert (fast.ledger["financing"] != 0).any()


def test_fast_mode_rejects_partial_fills():
    prices, tgt, *_ = _panel(0, 10, 2)
    with pytest.raises(ValueError, match="EventEngine"):
        run_fast(prices.pct_change(), tgt, NextBarFill(max_participation=0.1))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("fill", FILLS)
def test_fast_equals_event_with_asset_carry(seed, fill):
    """Per-asset carry (FX roll) agrees between the modes, with everything else on."""
    prices, tgt, rate, _sigma, _adv = _panel(seed + 10)
    rng = np.random.default_rng(seed)
    d = rng.normal(0.0, 2e-4, prices.shape)
    fin = Financing(rate=rate, borrow_spread_bp=50.0, lend_spread_bp=10.0)
    kw = dict(fill=fill, cost=LinearCost(3.0), financing=fin, borrow_bps_annual=75.0,
              asset_carry=d)
    fast = run_fast(prices.pct_change(), tgt, **kw)
    event = EventEngine(prices, **kw).run(WeightPath(tgt))
    pd.testing.assert_frame_equal(fast.ledger, event.ledger, rtol=0, atol=1e-10,
                                  check_exact=False)
    assert (fast.ledger["carry"] != 0).any()


def test_asset_carry_hand_example():
    """One asset, held at w = 1 from close 0 (auction fill at close 0, no cost).

    Period 1: r = +1%, d = 0.0002 (2 bp carry for the night). The book held
    overnight is w = 1 of NAV 1, so carry = 0.0002 and net = 0.0102.
    Period 2: no order; the book drifted to h = 1.01/1.0102. r = -1%,
    d = -0.0001: carry = -0.0001 h, gross = h * (-0.01).
    """
    idx = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"A": [100.0, 101.0, 99.99]}, index=idx)
    tgt = np.array([[1.0], [np.nan], [np.nan]])
    d = np.array([[0.0], [2e-4], [-1e-4]])
    for res in (run_fast(prices.pct_change(), tgt, NextBarFill(1.0, 0), asset_carry=d),
                EventEngine(prices, NextBarFill(1.0, 0), asset_carry=d).run(WeightPath(tgt))):
        led = res.ledger
        assert led["carry"].iloc[1] == pytest.approx(2e-4, abs=1e-15)
        assert led["net_ret"].iloc[1] == pytest.approx(0.0102, abs=1e-15)
        h = 1.01 / 1.0102
        assert led["carry"].iloc[2] == pytest.approx(-1e-4 * h, abs=1e-15)
        assert led["net_ret"].iloc[2] == pytest.approx(h * (99.99 / 101.0 - 1) - 1e-4 * h,
                                                       abs=1e-15)
    with pytest.raises(ValueError, match="shape"):
        run_fast(prices.pct_change(), tgt, asset_carry=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="finite"):
        run_fast(prices.pct_change(), tgt, asset_carry=np.full((3, 1), np.nan))
