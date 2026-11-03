"""Planted-pattern tests: every planted case is found; false-positive rates
on benign-only flow are measured over seeds and bounded."""

from __future__ import annotations

from functools import cache

import pytest
from qp_posttrade.model import OrderEvent
from qp_posttrade.sim import SESSION
from qp_posttrade.sim import generate as _generate
from qp_posttrade.surveillance import Alert, detect_marking_close, detect_spoofing, detect_wash

SEEDS = range(20)


@cache
def generate(seed: int, plant: bool = True) -> tuple[list[OrderEvent], list]:  # type: ignore[type-arg]
    return _generate(seed, plant=plant)


def _all(events: list[OrderEvent]) -> list[Alert]:
    return detect_spoofing(events) + detect_wash(events) + detect_marking_close(events, SESSION)


@pytest.mark.parametrize("seed", SEEDS)
def test_every_planted_case_found(seed: int) -> None:
    ev, planted = generate(seed)
    got = {(a.kind, a.trader, a.symbol) for a in _all(ev)}
    for p in planted:
        assert (p.kind, p.trader, p.symbol) in got, p


@pytest.mark.parametrize("seed", SEEDS)
def test_planted_runs_raise_no_alert_on_benign_traders(seed: int) -> None:
    ev, _ = generate(seed)
    assert all(a.trader.startswith("BAD") for a in _all(ev))


def test_false_positive_rates_on_benign_flow() -> None:
    counts: dict[str, int] = {}
    for seed in SEEDS:
        ev, _ = generate(seed, plant=False)
        for a in _all(ev):
            counts[a.kind] = counts.get(a.kind, 0) + 1
    # 20 benign sessions x 2 symbols x 8 traders (3 two-sided market makers
    # cancelling ~90 % of quotes, 5 directional traders). Stated rate: zero
    # false alerts in 20 sessions, i.e. below 0.15 per session at 95 %
    # (rule of three, 3/20).
    assert counts == {}


def test_alerts_carry_evidence_and_stamp() -> None:
    ev, _ = generate(1)
    alerts = _all(ev)
    assert alerts
    for a in alerts:
        assert a.stamp == "synthetic venue"
        assert a.evidence


def _e(
    ts: float, tr: str, oid: str, kind: str, side: str, qty: float, px: float = 100.0, acct: str = "ACC"
) -> OrderEvent:
    return OrderEvent(ts, tr, acct, "S", oid, kind, side, qty, px)  # type: ignore[arg-type]


def test_spoof_needs_the_cancel() -> None:
    ev = [
        _e(0, "T", "a", "NEW", "SELL", 5000, 100.1),
        _e(1, "T", "b", "NEW", "BUY", 100),
        _e(1.5, "T", "b", "FILL", "BUY", 100),
    ]
    assert detect_spoofing(ev) == []
    ev.append(_e(2, "T", "a", "CANCEL", "SELL", 5000))
    (a,) = detect_spoofing(ev)
    assert a.kind == "SPOOFING"
    assert a.evidence == ("b", "a")


def test_layering_needs_three_levels() -> None:
    ev = [_e(0, "T", f"s{k}", "NEW", "SELL", 2000, 100.1 + k / 100) for k in range(3)]
    ev += [_e(1, "T", "b", "NEW", "BUY", 100), _e(1.5, "T", "b", "FILL", "BUY", 100)]
    ev += [_e(2, "T", f"s{k}", "CANCEL", "SELL", 2000) for k in range(3)]
    (a,) = detect_spoofing(ev)
    assert a.kind == "LAYERING"


def test_two_sided_quoting_is_not_spoofing() -> None:
    ev = [
        _e(0, "T", "a", "NEW", "SELL", 5000, 100.1),
        _e(0, "T", "c", "NEW", "BUY", 5000, 99.9),
        _e(1.5, "T", "c", "FILL", "BUY", 100),
        _e(2, "T", "a", "CANCEL", "SELL", 5000),
    ]
    assert detect_spoofing(ev) == []


def test_wash_needs_same_beneficial_owner() -> None:
    buy = _e(0, "X", "a", "FILL", "BUY", 100)
    assert detect_wash([buy, _e(0.1, "Y", "b", "FILL", "SELL", 100, acct="OTHER")]) == []
    assert len(detect_wash([buy, _e(0.1, "Y", "b", "FILL", "SELL", 100)])) == 1
    linked = OrderEvent(0.1, "Y", "OTHER", "S", "b", "FILL", "SELL", 100, 100.0, beneficial_owner="ACC")
    assert len(detect_wash([buy, linked])) == 1


def test_marking_close_ignores_two_sided_late_trading() -> None:
    ev = [_e(SESSION - 100, "T", "a", "FILL", "BUY", 1000), _e(SESSION - 50, "T", "b", "FILL", "SELL", 1000)]
    assert detect_marking_close(ev, SESSION) == []
    assert len(detect_marking_close(ev[:1], SESSION)) == 1
