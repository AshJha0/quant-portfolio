"""Event mode: unit orders, partial fills, latency, calendar, OMS mapping."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from qp_bt import (
    Clock,
    EventEngine,
    MarketOrder,
    NextBarFill,
    TargetPosition,
    TargetWeights,
    solve_post_trade_nav,
)


class Once:
    def __init__(self, bar, intents):
        self.bar, self.intents = bar, intents
        self.seen = []

    def on_bar(self, ctx):
        self.seen.append(ctx.bar)
        return list(self.intents) if ctx.bar == self.bar else []


def _prices(n=6):
    idx = pd.bdate_range("2025-03-03", periods=n)
    return pd.DataFrame({"A": [10.0, 11.0, 12.0, 11.0, 13.0, 14.0][:n]}, index=idx)


def test_next_bar_fill_never_uses_the_decision_close():
    res = EventEngine(_prices(), NextBarFill(0.5, 1)).run(Once(1, [MarketOrder(0, 100.0)]))
    (f,) = res.fills
    assert (f.decision_bar, f.bar, f.price) == (1, 2, 11.5)   # 11 + 0.5 (12 - 11)


def test_lag_moves_the_fill_and_drops_orders_past_the_end():
    res = EventEngine(_prices(), NextBarFill(1.0, 3)).run(Once(1, [MarketOrder(0, 1.0)]))
    assert [(f.bar, f.price) for f in res.fills] == [(4, 13.0)]
    late = EventEngine(_prices(), NextBarFill(1.0, 3)).run(Once(4, [MarketOrder(0, 1.0)]))
    assert late.fills == []


def test_lag_zero_auction_fills_at_the_decision_close_in_the_same_row():
    res = EventEngine(_prices(), NextBarFill(1.0, 0, 10.0)).run(Once(2, [MarketOrder(0, 10.0)]))
    (f,) = res.fills
    assert (f.bar, f.price) == (2, 12.0)
    assert f.costs["extra"] == pytest.approx(10.0 * 12.0 * 1e-3)
    assert res.ledger["cost"].iloc[2] > 0
    assert res.ledger["cost"].iloc[3] == 0


def test_partial_fills_respect_participation_and_finish_later():
    prices = _prices()
    vol = pd.DataFrame({"A": [1000.0] * 6}, index=prices.index)
    eng = EventEngine(prices, NextBarFill(1.0, 1, max_participation=0.1), volumes=vol,
                      initial_nav=1e6)
    res = eng.run(Once(0, [MarketOrder(0, 250.0)]))
    assert [f.quantity for f in res.fills] == [100.0, 100.0, 50.0]
    assert [f.bar for f in res.fills] == [1, 2, 3]
    assert [f.leaves_qty for f in res.fills] == [150.0, 50.0, 0.0]
    assert [f.cum_qty for f in res.fills] == [100.0, 200.0, 250.0]
    assert res.positions["A"].tolist() == [0.0, 100.0, 200.0, 250.0, 250.0, 250.0]


def test_target_position_resolves_at_fill_prices():
    eng = EventEngine(_prices(), NextBarFill(0.5, 1))
    res = eng.run(Once(0, [TargetPosition(lambda pf: np.array([1000.0 / pf[0]]))]))
    assert res.positions["A"].iloc[1] == pytest.approx(1000.0 / 10.5)


def test_nav_identity_holds_every_bar():
    eng = EventEngine(_prices(), NextBarFill(0.5, 1), borrow_bps_annual=100.0)
    res = eng.run(Once(0, [TargetWeights(np.array([-0.5]))]))
    led = res.ledger
    np.testing.assert_allclose(led["nav"], np.cumprod(1 + led["net_ret"]), rtol=1e-14)
    assert led["borrow"].iloc[2:].gt(0).all()


def test_strategy_sees_every_bar_once_in_order():
    s = Once(-1, [])
    EventEngine(_prices()).run(s)
    assert s.seen == list(range(6))


def test_clock_rejects_holiday_bars():
    idx = pd.DatetimeIndex(["2025-12-24", "2025-12-25", "2025-12-26"])
    with pytest.raises(ValueError, match="non-business"):
        Clock(idx, "USD")
    Clock(pd.DatetimeIndex(["2025-12-24", "2025-12-26"]), "USD")


def test_clock_rejects_unsorted_index():
    with pytest.raises(ValueError, match="increasing"):
        Clock(pd.Index([2, 1]))


@pytest.mark.parametrize("kw", [dict(fraction=1.5), dict(lag=-1), dict(lag=0, fraction=0.5),
                                dict(extra_cost_bp=-1.0), dict(max_participation=0.0)])
def test_invalid_fill_models(kw):
    with pytest.raises(ValueError):
        NextBarFill(**kw)


def test_solver_wipes_out_when_cost_exceeds_the_nav():
    # selling a book of 1 at 200% cost leaves nothing: N = 0, absorbing
    assert solve_post_trade_nav(1.0, np.ones(1), np.zeros(1), lambda dv: 2.0 * abs(dv).sum()) == 0.0


def test_solver_bisects_when_iteration_would_diverge():
    # buying at 200% cost from flat: N = 1 - 2 N, so N = 1/3; plain iteration diverges
    n = solve_post_trade_nav(1.0, np.zeros(1), np.ones(1), lambda dv: 2.0 * abs(dv).sum())
    assert n == pytest.approx(1.0 / 3.0, abs=1e-15)


def test_fast_mode_absorbs_a_wipe_out():
    from qp_bt import LinearCost, run_fast

    rets = pd.DataFrame({"A": [0.0, 0.01, 0.01, 0.01]})
    tgt = np.array([[1.0], [0.0], [1.0], [np.nan]])
    led = run_fast(rets, tgt, NextBarFill(0.5, 1), LinearCost(20_000.0)).ledger
    assert led["net_ret"].iloc[2] == -1.0
    assert (led["nav"].iloc[2:] == 0.0).all()


def test_prices_must_be_positive():
    with pytest.raises(ValueError):
        EventEngine(pd.DataFrame({"A": [1.0, -1.0]}))


def _oms():
    path = pathlib.Path(__file__).resolve().parents[2] / "oms" / "interface.py"
    spec = importlib.util.spec_from_file_location("oms_interface", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["oms_interface"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_fill_records_map_onto_valid_oms_execution_reports():
    oms = _oms()
    prices = _prices()
    vol = pd.DataFrame({"A": [1000.0] * 6}, index=prices.index)
    res = EventEngine(prices, NextBarFill(1.0, 1, max_participation=0.1), volumes=vol,
                      initial_nav=1e6).run(
        Once(0, [MarketOrder(0, -250.0)]))
    order = oms.Order("c1", "A", oms.Side.SELL, 250.0)
    cum = 0.0
    reports = []
    for k, f in enumerate(res.fills):
        cum += abs(f.quantity)
        reports.append(oms.ExecutionReport.fill(order, f"e{k}", last_qty=abs(f.quantity),
                                                last_px=f.price, cum_qty=cum))
    assert [r.ord_status for r in reports] == [oms.OrdStatus.PARTIALLY_FILLED] * 2 + [
        oms.OrdStatus.FILLED]
    assert [r.leaves_qty for r in reports] == [150.0, 50.0, 0.0]


def test_oms_interface_invariants():
    oms = _oms()
    o = oms.Order("c1", "A", oms.Side.BUY, 100.0)
    assert oms.signed_qty(o) == 100.0
    with pytest.raises(ValueError):
        oms.Order("c2", "A", oms.Side.BUY, 0.0)
    with pytest.raises(ValueError):
        oms.Order("c3", "A", oms.Side.BUY, 1.0, oms.OrdType.LIMIT)
    with pytest.raises(ValueError, match="overfill"):
        oms.ExecutionReport.fill(o, "e", last_qty=150.0, last_px=1.0, cum_qty=150.0)
    with pytest.raises(ValueError, match="CumQty"):
        oms.ExecutionReport("c1", "e", oms.ExecType.NEW, oms.OrdStatus.NEW, 100.0, 0.0, 90.0)
    with pytest.raises(ValueError, match="leaves"):
        oms.ExecutionReport("c1", "e", oms.ExecType.CANCELED, oms.OrdStatus.CANCELED,
                            100.0, 10.0, 90.0)


def test_half_spread_cost_per_asset_and_per_bar():
    from qp_bt import HalfSpreadCost

    c = HalfSpreadCost(np.array([1e-4, 4e-4]))
    assert c(0, np.array([0.5, -0.25])) == pytest.approx(1.5e-4, abs=1e-18)
    c2 = HalfSpreadCost(np.array([[1e-4, 4e-4], [2e-4, 0.0]]))
    assert c2(1, np.array([0.5, -0.25])) == pytest.approx(1e-4, abs=1e-18)
    with pytest.raises(ValueError):
        HalfSpreadCost(np.array([-1e-4]))
