"""W26 central risk book: netting conserves risk, CRB never costs more, P&L identity."""

from __future__ import annotations

import numpy as np
import pytest
from qp_exec.crb import (
    CentralRiskBook,
    CostModel,
    Fill,
    FlowConfig,
    internalised_volume,
    saved_cost,
    simulate_crb,
    to_oms_orders,
)

INSTR = ["ES", "NQ", "ZN"]


def _book(seed: int = 0, n: int = 300) -> CentralRiskBook:
    rng = np.random.default_rng(seed)
    crb = CentralRiskBook(INSTR)
    for _ in range(n):
        crb.book(
            Fill(
                str(rng.choice(["cash", "deriv", "macro"])),
                str(rng.choice(INSTR)),
                float(rng.normal(0, 10)),
                float(rng.uniform(90, 110)),
            )
        )
    return crb


def test_hand_example() -> None:
    crb = CentralRiskBook(["ES"])
    crb.book(Fill("a", "ES", 10.0, 100.0))
    crb.book(Fill("b", "ES", -7.0, 100.0))
    assert crb.net() == {"ES": 3.0}
    assert crb.gross() == {"ES": 17.0}
    assert internalised_volume([10.0, -7.0]) == 7.0
    cm = CostModel({"ES": 1.0}, {"ES": 1e6}, {"ES": 0.01}, eta=0.5)
    rep = crb.netting_report(cm)
    # Hand: c(v) = v * 100 * (1e-4 + 0.5 * 0.01 * sqrt(v / 1e6)).
    c = lambda v: v * 100 * (1e-4 + 0.005 * (v / 1e6) ** 0.5)  # noqa: E731
    assert rep.standalone_cost == pytest.approx(c(17.0), rel=1e-14)
    assert rep.crb_cost == pytest.approx(c(3.0), rel=1e-14)


@pytest.mark.parametrize("seed", range(5))
def test_netting_conserves_factor_risk(seed: int) -> None:
    crb = _book(seed)
    b = np.random.default_rng(99).normal(size=(4, len(INSTR)))
    total = sum(crb.factor_exposure(b, desk=d) for d in crb.positions)
    assert np.max(np.abs(crb.factor_exposure(b) - total)) < 1e-9
    f = np.diag([0.1, 0.2, 0.3, 0.4])
    assert crb.factor_risk(b, f) >= 0.0


@pytest.mark.parametrize("seed", range(5))
def test_crb_never_costs_more(seed: int) -> None:
    crb = _book(seed)
    cm = CostModel(dict.fromkeys(INSTR, 0.5), dict.fromkeys(INSTR, 1e5), dict.fromkeys(INSTR, 0.015))
    rep = crb.netting_report(cm)
    assert rep.crb_cost <= rep.standalone_cost + 1e-12
    assert rep.saved == pytest.approx(rep.standalone_cost - rep.crb_cost)
    for i in INSTR:
        assert abs(rep.net[i]) <= rep.gross[i] + 1e-12


def test_unknown_instrument_and_bad_fill() -> None:
    crb = CentralRiskBook(["ES"])
    with pytest.raises(KeyError):
        crb.book(Fill("a", "XX", 1.0, 1.0))
    with pytest.raises(ValueError):
        Fill("a", "ES", 1.0, -1.0)


@pytest.mark.parametrize("delay", [1, 3, 10])
def test_pnl_decomposition_1e9(delay: int) -> None:
    r = simulate_crb(FlowConfig(steps=400), delay=delay, seed=delay)
    assert abs(r.total - (r.spread_capture + r.inventory_pnl - r.hedge_cost)) < 1e-9
    assert r.hedge_cost >= 0.0 and r.internalised <= r.client_volume


def test_delay_internalises_and_cuts_hedge_volume() -> None:
    cfg = FlowConfig(steps=1000)
    fast = simulate_crb(cfg, delay=1, seed=4)
    slow = simulate_crb(cfg, delay=10, seed=4)
    assert fast.internalised == 0.0
    assert slow.internalised > 0.0
    assert slow.hedge_volume < fast.hedge_volume
    assert slow.hedge_cost < fast.hedge_cost


def test_saved_cost_ci_brackets_mean() -> None:
    sc = saved_cost(FlowConfig(steps=300), delay=5, n_seeds=40, n_boot=500)
    assert sc.lower <= sc.mean <= sc.upper
    assert sc.per_seed.size == 40


def test_oms_seam_matches_submit_signature() -> None:
    r = simulate_crb(FlowConfig(steps=100), delay=5, seed=2)
    kws = to_oms_orders(r.orders)
    assert len(kws) == len(r.orders) > 0
    assert set(kws[0]) == {"clordid", "side", "symbol", "order_qty", "price"}
    assert all(k["side"] in ("1", "2") and isinstance(k["order_qty"], int) for k in kws)
    assert len({k["clordid"] for k in kws}) == len(kws)


def test_oms_seam_against_real_orderbook() -> None:
    oms = pytest.importorskip("qp_oms.core.book")
    from qp_core.eventlog import EventLog

    book = oms.OrderBook(EventLog.in_memory())
    r = simulate_crb(FlowConfig(steps=60), delay=5, seed=2)
    for kw in to_oms_orders(r.orders):
        book.submit(**kw, ts_ns=0)
    assert len(list(book.orders())) == len(r.orders)
