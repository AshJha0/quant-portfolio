"""W26 market making: toxicity-aware spreads and fill-rate vs markout by tier.

Every simulated number here comes from the W25 reference book (price-time
matching), with synthetic tiered taker flow. The flow is labelled
illustrative: an "informed" taker sees the fundamental ``lookahead`` steps
ahead, which no real counterparty does exactly.
"""

from __future__ import annotations

import math
import statistics

import pytest
from qp_exec.mm import VPIN, ASParams, ControlParams, MarkoutToxicity, QuoteController, tier_report
from qp_exec.mm.sim import MMConfig, TierSpec, ToxicityParams, run_mm

# ------------------------------------------------------------------ units


def test_markout_ewma_by_hand() -> None:
    t = MarkoutToxicity(horizon=2, lam=0.5, beta=1.0, max_widen=3)
    t.on_fill(0, 1, 100.0)  # bought at 100
    t.on_fill(1, -1, 101.0)  # sold at 101
    t.on_mid(1, 99.0)  # nothing matured yet (0 + 2 > 1)
    assert t.matured == 0 and t.widen_ticks() == 0
    t.on_mid(2, 98.0)  # first: +1 * (98 - 100) = -2
    assert t.mean == -2.0 and t.widen_ticks() == 2
    t.on_mid(3, 104.0)  # second: -1 * (104 - 101) = -3 -> 0.5*-2 + 0.5*-3
    assert t.mean == -2.5 and t.widen_ticks() == 3
    t2 = MarkoutToxicity(horizon=1, lam=1.0, beta=1.0, max_widen=2)
    t2.on_fill(0, 1, 100.0)
    t2.on_mid(1, 90.0)
    assert t2.widen_ticks() == 2  # capped
    t3 = MarkoutToxicity(horizon=1, lam=1.0)
    t3.on_fill(0, 1, 100.0, qty=3)
    t3.on_mid(1, 101.0)
    assert t3.matured == 3 and t3.mean == 1.0 and t3.widen_ticks() == 0
    with pytest.raises(ValueError):
        MarkoutToxicity(horizon=0)
    with pytest.raises(ValueError):
        t3.on_fill(0, 0, 1.0)


def test_markout_decay_prevents_lockout() -> None:
    t = MarkoutToxicity(horizon=1, lam=1.0, beta=1.0, max_widen=5, decay=0.5)
    t.on_fill(0, 1, 100.0)
    t.on_mid(1, 96.0)
    assert t.widen_ticks() == 4
    for s in range(2, 5):  # no fills: -4 -> -2 -> -1 -> -0.5
        t.on_mid(s, 96.0)
    assert t.mean == -0.5 and t.widen_ticks() == 1
    t.on_mid(5, 96.0)
    t.on_mid(6, 96.0)
    assert t.widen_ticks() == 1  # ceil(0.125) = 1; tiny but not yet zero
    no_decay = MarkoutToxicity(horizon=1, lam=1.0)
    no_decay.on_fill(0, 1, 100.0)
    no_decay.on_mid(1, 96.0)
    for s in range(2, 50):
        no_decay.on_mid(s, 96.0)
    assert no_decay.widen_ticks() == 4  # without decay the estimate never moves


def test_vpin_buckets_by_hand() -> None:
    v = VPIN(bucket_volume=10, n_buckets=2, threshold=0.2, scale=10, max_widen=3)
    assert v.value() is None and v.widen_ticks() == 0
    v.on_trade(1, 8)
    v.on_trade(-1, 2)  # bucket 1: |8-2|/10 = 0.6
    assert v.complete_buckets == 1 and v.value() == 0.6
    assert v.widen_ticks() == 3  # floor(10 * 0.4) = 4, capped at 3
    v.on_trade(1, 13)  # fills bucket 2 with 10 buys (1.0), 3 carry into bucket 3
    assert v.value() == 0.8
    v.on_trade(-1, 7)  # bucket 3: |3-7|/10 = 0.4; window is the last two
    assert v.value() == pytest.approx(0.7)
    with pytest.raises(ValueError):
        VPIN(bucket_volume=0)


def test_tier_report_by_hand() -> None:
    fills = [("a", 1.0), ("a", 3.0), ("b", -2.0)]
    rep = {s.tier: s for s in tier_report(fills, {"a": 4, "b": 10, "c": 0})}
    a = rep["a"]
    assert (a.fills, a.fill_rate, a.mean_markout, a.markout_pnl) == (2, 0.5, 2.0, 4.0)
    half = 1.959963984540054 * math.sqrt(2.0) / math.sqrt(2)
    assert a.ci95 == pytest.approx((2.0 - half, 2.0 + half))
    assert rep["b"].fill_rate == 0.1 and math.isnan(rep["b"].ci95[0])
    assert math.isnan(rep["c"].fill_rate) and rep["c"].fills == 0


def test_controller_extra_ticks_widens_both_sides() -> None:
    c = QuoteController(ControlParams(tick=0.01, max_position=5))
    d = c.decide(99.99, 100.01, 0, 9990, 10010, False, 0.0, extra_ticks=2)
    assert (d.bid, d.ask) == (9997, 10003)
    with pytest.raises(ValueError):
        c.decide(99.99, 100.01, 0, 9990, 10010, False, 1.0, extra_ticks=-1)


# ---------------------------------------------------- on the W25 book

MODEL = ASParams(0.5, 0.002, 100.0)
SEEDS = range(4)


def _run(informed: float, tox: ToxicityParams | None, seed: int):  # type: ignore[no-untyped-def]
    tiers = (TierSpec("retail", 0.5, 0.0), TierSpec("hf", 0.6, informed, 10))
    cfg = MMConfig(steps=1200, seed=seed, p_move=0.5, bg_offsets=(4, 5, 6, 7), tiers=tiers,
                   toxicity=tox, markout_horizon=10)
    return run_mm(cfg, MODEL)


def test_tier_report_separates_informed_counterparty() -> None:
    for seed in SEEDS:
        r = _run(0.9, None, seed)
        s = {t.tier: t for t in r.tier_stats}
        # Informed flow: lower realised edge, and the 95% intervals do not overlap.
        assert s["hf"].ci95[1] < s["retail"].ci95[0]
        assert s["hf"].fills > 100 and s["retail"].fills > 100
        assert 0 < s["hf"].fill_rate < 1 and 0 < s["retail"].fill_rate < 1
        assert len(r.fill_markouts) <= r.fills
        benign = {t.tier: t for t in _run(0.0, None, seed).tier_stats}
        # Same flow model with no information: the intervals overlap.
        assert benign["hf"].ci95[0] < benign["retail"].ci95[1]
        assert benign["retail"].ci95[0] < benign["hf"].ci95[1]


def test_vpin_widens_only_under_toxic_flow_and_pays() -> None:
    vp = ToxicityParams("vpin", bucket_volume=20, n_buckets=10, threshold=0.25, scale=10)
    gain_toxic, gain_benign, w_toxic, w_benign = [], [], [], []
    for seed in SEEDS:
        plain, aware = _run(0.9, None, seed), _run(0.9, vp, seed)
        gain_toxic.append(aware.total_pnl - plain.total_pnl)
        w_toxic.append(statistics.mean(aware.widen_path))
        plain_b, aware_b = _run(0.0, None, seed), _run(0.0, vp, seed)
        gain_benign.append(aware_b.total_pnl - plain_b.total_pnl)
        w_benign.append(statistics.mean(aware_b.widen_path))
    assert all(g > 0 for g in gain_toxic)
    assert statistics.mean(w_toxic) > 0.05
    assert statistics.mean(w_benign) < 0.01
    assert max(abs(g) for g in gain_benign) < 0.5


def test_markout_widening_reacts_more_to_toxic_flow() -> None:
    mk = ToxicityParams("markout", horizon=10, lam=0.05, beta=1.0)
    for seed in SEEDS:
        plain, aware = _run(0.9, None, seed), _run(0.9, mk, seed)
        assert aware.total_pnl > plain.total_pnl
        assert max(aware.widen_path) <= mk.max_widen
    w_t = statistics.mean(statistics.mean(_run(0.9, mk, s).widen_path) for s in SEEDS)
    w_b = statistics.mean(statistics.mean(_run(0.0, mk, s).widen_path) for s in SEEDS)
    assert w_t > w_b


def test_untiered_runs_are_unchanged_by_the_new_fields() -> None:
    a = run_mm(MMConfig(steps=300, seed=5), MODEL)
    assert a.tier_stats == [] and a.fill_markouts == [] and set(a.widen_path) == {0}
    with pytest.raises(ValueError):
        ToxicityParams("bogus")
