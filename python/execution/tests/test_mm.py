"""W26 market making: closed forms, inventory control, stale pull, runs on the W25 book."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qp_exec.mm import (
    GLFT,
    ASParams,
    ControlParams,
    GLFTParams,
    QuoteController,
    avellaneda_stoikov,
    glft_asymptotic,
)

# ------------------------------------------------------------ Avellaneda-Stoikov


def test_as_paper_numbers() -> None:
    # Avellaneda and Stoikov (2008) section 4 parameters, worked by hand.
    p = ASParams(gamma=0.1, sigma=2.0, k=1.5)
    q = avellaneda_stoikov(100.0, 0, 1.0, p)
    assert q.reservation == 100.0
    assert q.spread == pytest.approx(0.4 + 20 * math.log(1 + 0.1 / 1.5), abs=1e-12)
    assert q.spread == pytest.approx(1.690770, abs=1e-6)
    q3 = avellaneda_stoikov(100.0, -3, 0.5, p)
    assert q3.reservation == pytest.approx(100.0 + 3 * 0.1 * 4 * 0.5, abs=1e-12)


def test_as_limits() -> None:
    # gamma -> 0: risk-neutral, quotes at mid +- 1/k, no skew.
    p = ASParams(gamma=1e-9, sigma=2.0, k=1.5)
    q = avellaneda_stoikov(100.0, 4, 1.0, p)
    assert q.reservation == pytest.approx(100.0, abs=1e-7)
    assert q.spread == pytest.approx(2 / 1.5, abs=1e-7)
    # At the horizon the inventory term vanishes.
    p2 = ASParams(gamma=0.1, sigma=2.0, k=1.5)
    q0 = avellaneda_stoikov(100.0, 7, 0.0, p2)
    assert q0.reservation == 100.0
    assert q0.spread == pytest.approx(20 * math.log1p(0.1 / 1.5), abs=1e-12)


def test_as_inventory_skew_and_bound() -> None:
    p = ASParams(gamma=0.1, sigma=2.0, k=1.5, q_max=3)
    rs = [avellaneda_stoikov(100.0, q, 1.0, p).reservation for q in range(-2, 3)]
    assert all(a > b for a, b in zip(rs, rs[1:], strict=False))
    assert avellaneda_stoikov(100.0, 3, 1.0, p).bid is None
    assert avellaneda_stoikov(100.0, -3, 1.0, p).ask is None
    with pytest.raises(ValueError):
        ASParams(gamma=0.0, sigma=1.0, k=1.0)


# ------------------------------------------------------------------------ GLFT

P = GLFTParams(gamma=0.05, sigma=0.3, k=1.5, A=1.0, q_max=6)


def test_glft_symmetry_and_monotone() -> None:
    g = GLFT(P)
    for tau in (0.5, 5.0, None):
        for q in range(-5, 6):
            db, da = g.deltas(q, tau)
            db2, da2 = g.deltas(-q, tau)
            assert db == pytest.approx(da2, abs=1e-10) and da == pytest.approx(db2, abs=1e-10)
        dbs = [g.deltas(q, tau)[0] for q in range(-6, 6)]
        das = [g.deltas(q, tau)[1] for q in range(-5, 7)]
        assert all(b > a for a, b in zip(dbs, dbs[1:], strict=False))  # longer -> lower bid
        assert all(b < a for a, b in zip(das, das[1:], strict=False))  # longer -> lower ask
    assert g.deltas(6, None)[0] is None and g.deltas(-6, None)[1] is None


def test_glft_at_horizon_is_myopic() -> None:
    # tau = 0: v = 1, so both deltas equal (1/gamma) ln(1+gamma/k), any q.
    g = GLFT(P)
    base = math.log1p(P.gamma / P.k) / P.gamma
    for q in range(-5, 6):
        db, da = g.deltas(q, 0.0)
        assert db == pytest.approx(base, abs=1e-12) and da == pytest.approx(base, abs=1e-12)


def test_glft_matches_matrix_exponential_and_ground_state() -> None:
    from scipy.linalg import expm

    g = GLFT(P)
    Q = P.q_max
    qs = np.arange(-Q, Q + 1)
    M = np.diag(P.alpha * qs**2.0) + np.diag(np.full(2 * Q, -P.eta), 1) + np.diag(np.full(2 * Q, -P.eta), -1)
    tau = 3.0
    v = expm(-M * tau) @ np.ones(2 * Q + 1)  # direct, unshifted
    base = math.log1p(P.gamma / P.k) / P.gamma
    for q in range(-Q, Q):
        i = q + Q
        assert g.deltas(q, tau)[0] == pytest.approx(math.log(v[i] / v[i + 1]) / P.k + base, abs=1e-10)
    # Long horizon converges to the ground eigenvector quotes.
    for q in range(-Q, Q):
        assert g.deltas(q, 500.0)[0] == pytest.approx(g.deltas(q, None)[0], abs=1e-9)


def test_glft_asymptotic_close_to_exact_stationary() -> None:
    # The asymptotic formula approximates the stationary quotes near q = 0.
    g = GLFT(GLFTParams(gamma=0.01, sigma=0.3, k=1.5, A=1.0, q_max=20))
    pa = GLFTParams(gamma=0.01, sigma=0.3, k=1.5, A=1.0, q_max=20)
    for q in (-2, 0, 2):
        ex = g.quote(100.0, q)
        ap = glft_asymptotic(100.0, q, pa)
        assert ex.bid is not None and ap.bid is not None
        assert ex.bid == pytest.approx(ap.bid, abs=0.02 * (100 - ap.bid))
        assert ex.reservation == pytest.approx(ap.reservation, abs=0.02)


def test_glft_small_gamma_matches_as() -> None:
    # gamma -> 0 at q = 0, short horizon, wide inventory band: both models quote
    # mid +- 1/k. (The bound itself has a small option value, so q_max is wide.)
    g = GLFT(GLFTParams(gamma=1e-6, sigma=0.3, k=1.5, A=1.0, q_max=20))
    db, da = g.deltas(0, 0.1)
    a = avellaneda_stoikov(0.0, 0, 0.1, ASParams(gamma=1e-6, sigma=0.3, k=1.5))
    assert a.bid is not None and a.ask is not None
    assert db == pytest.approx(-a.bid, abs=1e-5) and da == pytest.approx(a.ask, abs=1e-5)


# -------------------------------------------------------------------- controls


def test_controller_stale_pulls_and_crossed_book() -> None:
    c = QuoteController(ControlParams(tick=0.01, max_position=5))
    assert c.decide(99.98, 100.02, 0, 9998, 10002, False, 0.0).reason == "ok"
    d = c.decide(99.98, 100.02, 0, 9998, 10002, False, 1.0, hedge_stale=True)
    assert (d.bid, d.ask, d.reason) == (None, None, "stale")
    assert c.decide(99.98, 100.02, 0, 10002, 10002, False, 2.0).bid is None  # locked book
    assert c.decide(99.98, 100.02, 0, None, 10002, False, 3.0).ask is None  # one-sided


def test_controller_widen_pull_and_no_cross() -> None:
    c = QuoteController(ControlParams(tick=0.01, max_position=5, widen_at=3, widen_ticks=2))
    d = c.decide(99.98, 100.02, 3, 9990, 10010, False, 0.0)
    assert (d.bid, d.ask) == (9996, 10002)
    d = c.decide(99.98, 100.02, 5, 9990, 10010, False, 1.0)
    assert d.bid is None and d.ask == 10002
    d = c.decide(99.98, 100.02, -5, 9990, 10010, False, 2.0)
    assert d.ask is None and d.bid == 9998
    d = c.decide(100.5, 100.6, 0, 9999, 10001, False, 3.0)  # model above the market
    assert d.bid == 10000 and d.ask == 10060  # bid clamped below the best ask


def test_controller_throttle_never_blocks_pull() -> None:
    c = QuoteController(ControlParams(tick=0.01, max_position=5, max_updates=2, window=10.0))
    assert c.decide(99.98, 100.02, 0, 9990, 10010, False, 0.0).reason == "ok"
    assert c.decide(99.97, 100.03, 0, 9990, 10010, False, 1.0).reason == "ok"
    d = c.decide(99.96, 100.04, 0, 9990, 10010, False, 2.0)
    assert d.reason == "throttled" and (d.bid, d.ask) == (9997, 10003)
    assert c.decide(99.96, 100.04, 0, 9990, 10010, True, 3.0).reason == "stale"
    assert c.decide(99.96, 100.04, 0, 9990, 10010, False, 11.5).reason == "ok"


# ------------------------------------------------------------- against W25 book


def test_pnl_decomposition_sums_and_inventory_bounded() -> None:
    pytest.importorskip("qp_exec.lob")  # W25 book (lane r12-l1-book)
    from qp_exec.mm.sim import MMConfig, run_mm

    for model in (ASParams(gamma=0.5, sigma=0.002, k=100.0), GLFTParams(0.5, 0.002, 100.0, 1.0, 5)):
        r = run_mm(MMConfig(steps=3000, seed=7), model)
        assert r.fills > 100
        assert abs(r.spread_pnl + r.inventory_pnl - r.total_pnl) < 1e-9
        assert r.max_abs_q <= 5  # the controller's max_position, unit clips
        assert r.spread_pnl > 0  # a passive maker earns the spread on average


def test_stale_flag_pulls_quotes_on_the_book() -> None:
    pytest.importorskip("qp_exec.lob")  # W25 book (lane r12-l1-book)
    from qp_exec.mm.sim import MMConfig, run_mm

    stale = frozenset(range(500, 700))
    r = run_mm(MMConfig(steps=1000, seed=3, stale_steps=stale), ASParams(0.5, 0.002, 100.0))
    assert r.quoted_while_stale == 0 and r.stale_pulls >= 1
    flat = run_mm(MMConfig(steps=1000, seed=3), ASParams(0.5, 0.002, 100.0))
    assert flat.fills > r.fills  # quoting resumes but the stale window lost flow


def test_skew_mean_reverts_inventory() -> None:
    pytest.importorskip("qp_exec.lob")  # W25 book (lane r12-l1-book)
    from qp_exec.mm.sim import MMConfig, run_mm

    risky = run_mm(MMConfig(steps=4000, seed=11, widen_at=None, max_position=50),
                   ASParams(gamma=5.0, sigma=0.002, k=100.0))
    neutral = run_mm(MMConfig(steps=4000, seed=11, widen_at=None, max_position=50),
                     ASParams(gamma=1e-6, sigma=0.002, k=100.0))
    assert np.mean(np.abs(risky.q_path)) < np.mean(np.abs(neutral.q_path))
