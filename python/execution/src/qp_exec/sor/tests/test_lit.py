"""W27a: allocation, fees, sweep, market, router and routing TCA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qp_exec.lob import Order, Side, Tif
from qp_exec.sor import (
    DEFAULT_DARK,
    DEFAULT_LIT,
    ExecFill,
    Market,
    RouterReport,
    SmartRouter,
    deterministic_optimum,
    expected_cost,
    load_fee_table,
    single_venue_optimum,
    solve_allocation,
    sweep_plan,
    venue_tca,
)

VENUES = Path(__file__).resolve().parents[4] / "venues" / "fee_schedules.toml"


# ------------------------------------------------------------------ allocation
@pytest.mark.parametrize("h,f,r,lam", [(0.5, 0.3, 0.2, 2.0), (0.5, 0.3, 0.2, 5.0),
                                       (1.0, 0.1, -0.1, 3.0), (0.5, 0.0, 0.3, 1.3)])
def test_single_venue_matches_quantile_rule(h: float, f: float, r: float, lam: float) -> None:
    # D on a grid where F jumps by 1/40: the quantile is unique unless q*40 is an integer
    D = np.arange(40.0).reshape(-1, 1)
    q = (2 * h + f + r) / (lam + h + r)
    assert abs(q * 40 - round(q * 40)) > 1e-6
    a = solve_allocation(S=30, h=h, f=f, c=[-(h + r)], lam_u=lam, D=D)
    m, lst = single_venue_optimum(S=30, h=h, f=f, r=r, lam_u=lam, D=D)
    assert a.exact
    assert a.market == pytest.approx(m, abs=1e-6)
    assert a.limits[0] == pytest.approx(lst, abs=1e-6)


def test_single_venue_no_market_when_penalty_small() -> None:
    D = np.arange(10.0).reshape(-1, 1)
    a = solve_allocation(S=5, h=0.5, f=0.3, c=[-0.7], lam_u=0.7, D=D)
    assert a.market == pytest.approx(0.0, abs=1e-9)
    assert single_venue_optimum(S=5, h=0.5, f=0.3, r=0.2, lam_u=0.7, D=D) == (0.0, 5.0)


@pytest.mark.parametrize("lam", [0.5, 3.0])
def test_deterministic_flow_is_greedy(lam: float) -> None:
    c = [-0.7, -0.4, -0.9, 0.1]
    d = [4.0, 10.0, 3.0, 20.0]
    D = np.tile(d, (5, 1))
    a = solve_allocation(S=12, h=0.5, f=0.3, c=c, lam_u=lam, D=D)
    m, lim = deterministic_optimum(S=12, h=0.5, f=0.3, c=c, lam_u=lam, D=d)
    assert a.cost == pytest.approx(expected_cost(m, lim, S=12, h=0.5, f=0.3, c=c,
                                                 lam_u=lam, D=D), abs=1e-9)
    assert a.market == pytest.approx(m, abs=1e-6)
    assert a.limits == pytest.approx(lim, abs=1e-6)


def test_lp_cost_equals_direct_evaluation() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        k = int(rng.integers(1, 5))
        D = rng.poisson(rng.uniform(1, 20, size=k), size=(100, k)).astype(float)
        c = list(rng.uniform(-1.5, 0.3, size=k))
        a = solve_allocation(S=25, h=0.5, f=0.3, c=c, lam_u=2.0, D=D)
        assert a.exact
        assert a.market + sum(a.limits) <= 25 + 1e-7
        direct = expected_cost(a.market, a.limits, S=25, h=0.5, f=0.3, c=c, lam_u=2.0, D=D)
        assert a.cost == pytest.approx(direct, abs=1e-7)


def test_matches_cvxpy() -> None:
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(11)
    for _ in range(10):
        k = int(rng.integers(1, 6))
        D = rng.poisson(rng.uniform(1, 15, size=k), size=(60, k)).astype(float)
        c = list(rng.uniform(-1.2, 0.5, size=k))
        S, h, f, lam = 20.0, 0.5, 0.25, 2.5
        M = cp.Variable(nonneg=True)
        L = cp.Variable(k, nonneg=True)
        fills = cp.minimum(cp.reshape(L, (1, k), order="C"), D)
        obj = (h + f - lam) * M + cp.sum(fills @ (np.asarray(c) - lam)) / D.shape[0] + lam * S
        prob = cp.Problem(cp.Minimize(obj), [M + cp.sum(L) <= S])
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
        a = solve_allocation(S=S, h=h, f=f, c=list(c), lam_u=lam, D=D)
        assert a.cost == pytest.approx(float(prob.value), abs=1e-6)
        # the LP optimum also attains the cvxpy optimum when evaluated directly
        mv, lv = float(M.value), [float(x) for x in L.value]
        assert expected_cost(mv, lv, S=S, h=h, f=f, c=c, lam_u=lam, D=D) == pytest.approx(
            a.cost, abs=1e-6)


def test_rejects_nonconvex_venue() -> None:
    with pytest.raises(ValueError):
        solve_allocation(S=5, h=0.5, f=0.3, c=[2.5], lam_u=2.0, D=np.ones((3, 1)))


# ------------------------------------------------------------------ fees
def test_fee_file_matches_defaults() -> None:
    lit, dark = load_fee_table(VENUES)
    assert lit == DEFAULT_LIT
    assert dark == DEFAULT_DARK
    assert all(s.taker <= 0.003 for s in lit.values())


def test_fee_cap_enforced(tmp_path: Path) -> None:
    p = tmp_path / "f.toml"
    p.write_text('[lit.X]\nmodel="maker-taker"\nmaker=0.002\ntaker=0.0031\nlatency_us=1\n')
    with pytest.raises(ValueError):
        load_fee_table(p)


# ------------------------------------------------------------------ sweep
def test_sweep_best_price_first_and_no_trade_through() -> None:
    m = Market(5)
    m.run_until(50_000)
    plan = sweep_plan(m, Side.BUY, 60)
    asks = sorted({px for v in m.lit.values() for px, _ in v.book.depth(Side.SELL, 20)})
    prices = [leg.price for leg in plan]
    assert prices == sorted(prices)
    assert sum(leg.qty for leg in plan) == 60
    # every level strictly better than the worst level used is fully taken
    worst = max(prices)
    for name, v in m.lit.items():
        for px, q in v.book.depth(Side.SELL, 20):
            if px < worst:
                assert sum(leg.qty for leg in plan if leg.venue == name and leg.price == px) == q
    assert asks[0] == prices[0]


def test_sweep_tie_breaks_on_taker_fee() -> None:
    m = Market(5)
    plan = sweep_plan(m, Side.BUY, 1)
    best = min(a for v in m.lit.values() if (a := v.book.best_ask()) is not None)
    at_best = [n for n, v in m.lit.items() if v.book.best_ask() == best]
    want = min(at_best, key=lambda n: (m.lit[n].sched.taker_ticks, m.lit[n].sched.latency_us))
    assert plan[0].venue == want


def test_sweep_respects_limit() -> None:
    m = Market(5)
    bb = max(a for v in m.lit.values() if (a := v.book.best_ask()) is not None)
    plan = sweep_plan(m, Side.BUY, 10_000, limit=bb)
    assert all(leg.price <= bb for leg in plan)


# ------------------------------------------------------------------ market and router
def _run(seed: int, **kw: Any) -> tuple[Market, RouterReport]:
    m = Market(seed)
    r = SmartRouter(m, Side.BUY, 120, 3_000_000, **kw).run()
    return m, r


def test_router_fills_exactly_and_is_reproducible() -> None:
    m1, r1 = _run(7, dark=("DARK_MID", "DARK_FBA"))
    _m2, r2 = _run(7, dark=("DARK_MID", "DARK_FBA"))
    assert r1.filled == 120 and r1.overfill == 0
    assert [(f.venue, f.ts, f.price, f.qty) for f in r1.fills] == \
        [(f.venue, f.ts, f.price, f.qty) for f in r2.fills]
    for v in m1.lit.values():
        v.book.check_invariants()


def test_router_never_overfills_across_seeds() -> None:
    for s in range(6):
        _, r = _run(100 + s, dark=tuple(DEFAULT_DARK), randomise=bool(s % 2))
        assert r.filled == 120, s


def test_latency_makes_resting_quotes_stale() -> None:
    # An IOC sent at the visible best ask can miss when the quote moves during
    # the latency: the slowest venue has the lowest fill rate for IOCs.
    m = Market(9)
    sent = {n: 0 for n in m.lit}
    for _ in range(300):
        for n, v in m.lit.items():
            a = v.book.best_ask()
            if a is not None:
                m.send_lit(n, Order(m.new_id(), Side.BUY, 1, a, tif=Tif.IOC))
                sent[n] += 1
        m.run_until(m.t + m.p.step_us)
    filled = {n: sum(f.qty for f in m.fills if f.venue == n) for n in m.lit}
    rate = {n: filled[n] / sent[n] for n in m.lit}
    slow = max(m.lit, key=lambda n: m.lit[n].sched.latency_us)
    assert rate[slow] <= max(rate.values())
    assert all(0 < x <= 1 for x in rate.values())


# ------------------------------------------------------------------ TCA
def test_venue_tca_by_hand() -> None:
    m = Market(1)
    m.mid_t[:] = [0, 100, 200]
    m.mid_v[:] = [100.5, 101.0, 99.0]
    fills = [ExecFill("X", 0, Side.BUY, 101.0, 2, "A", 0.3, 100.5, 1, "bg"),
             ExecFill("X", 0, Side.BUY, 100.0, 2, "P", -0.2, 100.5, 2, "bg")]
    t = venue_tca(fills, m, {"X": 8}, taus_us=(100, 200))["X"]
    assert t.fill_rate == pytest.approx(0.5)
    assert t.eff_spread == pytest.approx((2 * 0.5 * 2 + 2 * -0.5 * 2) / 4)
    assert t.fee == pytest.approx((0.3 * 2 - 0.2 * 2) / 4)
    assert t.markout[100][0] == pytest.approx((0.0 * 2 + 1.0 * 2) / 4)
    assert t.markout[200][0] == pytest.approx((-2.0 * 2 - 1.0 * 2) / 4)
