"""W27b: dark venue rules, counterparties, leakage, learning and the split LP."""

from __future__ import annotations

import os

import numpy as np
import pytest
from qp_exec.lob import Side
from qp_exec.sor import Market, MarketParams, SmartRouter, VenueScores, expected_cost, solve_allocation
from qp_exec.sor import studies as S
from qp_exec.sor.dark import ConditionalPool, DarkOrder, MidpointPool, PeriodicAuctionPool
from qp_exec.sor.gaming import auc


def _quiet() -> Market:
    """A market with no dark background flow, so venue rules can be tested by hand."""
    return Market(0, MarketParams(dark_rate=0.0, lit_rate=0.0))


# ------------------------------------------------------------------ venue rules
def test_midpoint_cross_at_mid_in_time_priority() -> None:
    m = _quiet()
    dv = m.dark["DARK_MID"]
    assert isinstance(dv, MidpointPool)
    dv.submit(DarkOrder(m.new_id(), "router", Side.BUY, 5, 0, 0))
    dv.submit(DarkOrder(m.new_id(), "bg", Side.BUY, 5, 0, 1))
    dv.submit(DarkOrder(m.new_id(), "bg", Side.SELL, 3, 0, 2))
    assert [f.qty for f in m.fills] == [3]
    assert m.fills[0].price == m.nbbo_mid()
    assert m.fills[0].fee_ticks == pytest.approx(0.1)
    assert [o.rem for o in dv.resting] == [2, 5]


def test_min_qty_blocks_small_contras_but_not_large() -> None:
    m = _quiet()
    dv = m.dark["DARK_MID"]
    dv.submit(DarkOrder(m.new_id(), "router", Side.BUY, 30, 10, 0))
    dv.submit(DarkOrder(m.new_id(), "bg", Side.SELL, 4, 0, 1, ioc=True))
    assert m.fills == []
    dv.submit(DarkOrder(m.new_id(), "bg", Side.SELL, 12, 0, 2, ioc=True))
    assert [f.qty for f in m.fills] == [12]
    # remaining 18 >= 10, so the floor still binds; at remaining < min_qty it relaxes
    dv.submit(DarkOrder(m.new_id(), "bg", Side.SELL, 9, 0, 3, ioc=True))
    assert [f.qty for f in m.fills] == [12]


def test_venue_floor_applies_to_everyone() -> None:
    m = _quiet()
    dv = m.dark["DARK_BLOCK"]  # floor 20
    dv.submit(DarkOrder(m.new_id(), "router", Side.BUY, 50, 0, 0))
    dv.submit(DarkOrder(m.new_id(), "bg", Side.SELL, 19, 0, 1, ioc=True))
    assert m.fills == []
    dv.submit(DarkOrder(m.new_id(), "bg", Side.SELL, 25, 0, 2, ioc=True))
    assert [f.qty for f in m.fills] == [25]


def test_periodic_auction_ignores_arrival_time_within_interval() -> None:
    outs = []
    for early in (True, False):
        m = _quiet()
        dv = m.dark["DARK_FBA"]
        assert isinstance(dv, PeriodicAuctionPool)
        first, second = ("router", "bg") if early else ("bg", "router")
        dv.submit(DarkOrder(m.new_id(), "bg", Side.SELL, 10, 0, 1))
        dv.submit(DarkOrder(m.new_id(), first if first == "router" else "bg2", Side.BUY, 10, 0, 2))
        dv.submit(DarkOrder(m.new_id(), second if second == "router" else "bg2", Side.BUY, 10, 0, 3))
        assert m.fills == []  # nothing trades before the auction
        m.run_until(dv.sched.interval_us + 1)
        outs.append(sum(f.qty for f in m.fills))
    # time priority inside the batch still decides who of two equal bids fills
    assert outs == [10, 0]


def test_periodic_auction_uncrosses_only_on_interval() -> None:
    m = _quiet()
    dv = m.dark["DARK_FBA"]
    dv.submit(DarkOrder(m.new_id(), "router", Side.BUY, 7, 0, 0))
    dv.submit(DarkOrder(m.new_id(), "bg", Side.SELL, 7, 0, 0))
    m.run_until(dv.sched.interval_us - m.p.step_us + 1)
    assert m.fills == []
    m.run_until(dv.sched.interval_us + 1)
    assert [f.qty for f in m.fills] == [7]


def test_conditional_firm_up_failure_and_suspension() -> None:
    m = _quiet()
    dv = m.dark["DARK_COND"]
    assert isinstance(dv, ConditionalPool)
    dv.firm_up = lambda o, x: 0  # the router always declines
    for i in range(6):
        dv.submit(DarkOrder(m.new_id(), "router", Side.BUY, 20, 0, i))
        dv.submit(DarkOrder(m.new_id(), "bg", Side.SELL, 20, 0, i, ioc=True))
        dv.cancel(dv.resting[0].oid) if dv.resting else None
    assert m.fills == []
    assert dv.firm_stats["router"].firms == 0
    assert dv.suspended("router")
    n = len(dv.resting)
    dv.submit(DarkOrder(m.new_id(), "router", Side.BUY, 20, 0, 9))
    assert len(dv.resting) == n  # suspended participants are turned away
    # each invitation told the other side ours existed
    assert sum(1 for _, who, sd in dv.leaks if who == "bg" and sd == Side.BUY) >= 5


def test_conditional_firm_up_success_trades() -> None:
    m = _quiet()
    dv = m.dark["DARK_COND"]
    dv.firm_up = lambda o, x: x
    dv.submit(DarkOrder(m.new_id(), "router", Side.BUY, 20, 0, 0))
    m.rng = np.random.default_rng(1)
    for _ in range(5):
        dv.submit(DarkOrder(m.new_id(), "bg", Side.SELL, 20, 0, 1, ioc=True))
        if m.fills:
            break
    assert [f.qty for f in m.fills] == [20]


# ------------------------------------------------------------------ studies (directional)
def test_informed_counterparties_are_more_adverse() -> None:
    r = S.adverse_selection(range(6))
    assert r["informed"].mean < r["bg"].mean


def test_min_qty_changes_the_fill_distribution() -> None:
    rows = {r.min_qty: r for r in S.min_qty_effect(range(6), mqs=(0, 5, 20))}
    assert rows[0].pinged_lots.mean > 5 > rows[5].pinged_lots.mean
    assert rows[20].pinged_lots.mean == 0
    assert rows[0].fill_size.mean < rows[5].fill_size.mean < rows[20].fill_size.mean
    assert rows[0].fills_per_episode.mean > rows[20].fills_per_episode.mean


def test_iois_leak_and_non_actionable_leak_more() -> None:
    rows = {r.mode: r for r in S.ioi_leakage(range(6))}
    assert rows["none"].detections.mean == 0
    assert 0 < rows["actionable"].detections.mean < rows["non-actionable"].detections.mean


def test_conditional_penalty_suspends_the_predator() -> None:
    r = S.conditional_firm_up(range(4))
    assert r.rates["predator"].mean < 0.3 < r.rates["bg"].mean
    assert r.rates["router"].mean == 1.0
    assert r.predator_suspended == 1.0


def test_markout_learning_moves_flow_off_the_toxic_venue() -> None:
    off, on = S.score_learning(range(4))
    assert not off.learning and on.learning
    assert on.toxic_share_late.mean < off.toxic_share_late.mean
    assert on.toxic_score.mean < 0  # learned: fills there go against us


# ------------------------------------------------------------------ split LP vs convex solver
def test_live_lit_dark_split_matches_cvxpy() -> None:
    cp = pytest.importorskip("cvxpy")
    m = Market(4)
    rt = SmartRouter(m, Side.BUY, 120, 3_000_000, dark=S.ALL_DARK, scores=VenueScores())
    rt.run()
    assert rt.decisions
    lit = dark = False
    for d in rt.decisions[:: max(1, len(rt.decisions) // 6)]:
        D, c, k = d["D"], np.asarray(d["c"]), len(d["c"])
        lit |= any(v.startswith("LIT") for v in d["venues"])
        dark |= any(v.startswith("DARK") for v in d["venues"])
        M = cp.Variable(nonneg=True)
        L = cp.Variable(k, nonneg=True)
        fills = cp.minimum(cp.reshape(L, (1, k), order="C"), D)
        obj = ((d["h"] + d["f"] - d["lam_u"]) * M + cp.sum(fills @ (c - d["lam_u"])) / D.shape[0]
               + d["lam_u"] * d["S"])
        cp.Problem(cp.Minimize(obj), [M + cp.sum(L) <= d["S"]]).solve(
            solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
        a = solve_allocation(S=d["S"], h=d["h"], f=d["f"], c=list(c), lam_u=d["lam_u"], D=D)
        assert a.cost == pytest.approx(d["alloc"].cost, abs=1e-12)
        assert a.cost == pytest.approx(float(obj.value), abs=1e-6)
        assert expected_cost(float(M.value), list(L.value), S=d["S"], h=d["h"], f=d["f"],
                             c=list(c), lam_u=d["lam_u"], D=D) == pytest.approx(a.cost, abs=1e-6)
    assert lit and dark


# ------------------------------------------------------------------ AUC helper
def test_auc_known_values() -> None:
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0]), np.array([1.0])) == 0.5
    assert auc(np.array([1.0, 3.0]), np.array([2.0])) == 0.5


# ------------------------------------------------------------------ anti-gaming
def test_randomised_router_is_not_pingable() -> None:
    """With randomisation no 1-lot ping ever takes a lot of ours in the dark."""
    from qp_exec.sor.dark import Predator

    for s in range(3):
        m = Market(s, MarketParams(informed_share=0.0))
        m.predators.append(Predator(m, ping_prob=0.5))
        r = SmartRouter(m, Side.BUY, 100, 2_000_000, dark=("DARK_MID", "DARK_FBA"),
                        randomise=True, seed=s).run()
        assert not [f for f in r.fills if f.liquidity == "D" and f.counterparty == "predator"]


@pytest.mark.skipif(not os.environ.get("QP_SLOW"), reason="about 15 minutes; set QP_SLOW=1")
def test_detector_auc_acceptance() -> None:
    """Plan acceptance A1: detector AUC below 0.55 under randomisation.

    Randomising size, timing and min-qty alone is not enough (AUC 0.59,
    CI 0.53-0.66 at n=300): the parent depletes contra liquidity in the dark,
    which pings then find less of on its side. Using each dark venue in only
    a quarter of intervals removes that channel; the price is that more of
    the order goes to lit venues, where net demand shows (imbalance AUC 0.70).
    """
    from qp_exec.sor.gaming import detector_auc

    det = detector_auc(False, n=300, seed=2)
    rnd = detector_auc(True, n=300, seed=2, dark_participation=0.25)
    assert det.ci[0] > 0.55  # the regular router is detectable
    assert rnd.auc < 0.55
    assert rnd.ci[0] < 0.5 < rnd.ci[1]
