"""State machine tests: scenarios after FIX 4.4 Vol. 4 Appendix D, plus property tests.

Scenario names follow the Appendix D groups (D1 vanilla, D2 cancel,
D3 replace, D4 unsolicited / reinstatement, D5 status / restatement,
D6 trade cancel). Each scenario lists (ExecType, expected OrdStatus,
CumQty, LeavesQty) per step, the same columns the Appendix tables use.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qp_core.eventlog import EventLog
from qp_oms.core import (
    ExecReport,
    ExecType,
    IllegalTransition,
    Order,
    OrderBook,
    OrdStatus,
    apply_cancel_reject,
    apply_report,
    request_cancel,
    request_replace,
)

E = ExecType
S = OrdStatus


def _new(qty: int = 10_000) -> Order:
    return Order.new("C1", side="1", symbol="XYZ", order_qty=qty, price=10.0)


def _ack(o: Order) -> Order:
    return apply_report(o, ExecReport(E.NEW, "ack", "C1"))


def _fill(o: Order, eid: str, qty: int, px: float = 10.0, clordid: str | None = None) -> Order:
    return apply_report(o, ExecReport(E.TRADE, eid, clordid or o.clordid, last_qty=qty, last_px=px))


def _state(o: Order) -> tuple[OrdStatus, int, int]:
    return o.ord_status, o.cum_qty, o.leaves_qty


# --- D1 vanilla -----------------------------------------------------------
def test_d1_filled_order() -> None:
    o = _new()
    assert _state(o) == (S.PENDING_NEW, 0, 10_000)
    o = apply_report(o, ExecReport(E.PENDING_NEW, "p", "C1"))
    o = _ack(o)
    assert _state(o) == (S.NEW, 0, 10_000)
    o = _fill(o, "f1", 2000, 10.0)
    assert _state(o) == (S.PARTIALLY_FILLED, 2000, 8000)
    o = _fill(o, "f2", 1000, 11.0)
    assert _state(o) == (S.PARTIALLY_FILLED, 3000, 7000)
    o = _fill(o, "f3", 7000, 12.0)
    assert _state(o) == (S.FILLED, 10_000, 0)
    assert o.avg_px == pytest.approx((2000 * 10 + 1000 * 11 + 7000 * 12) / 10_000)


def test_d1_rejected_order() -> None:
    o = apply_report(_new(), ExecReport(E.REJECTED, "r", "C1"))
    assert _state(o) == (S.REJECTED, 0, 0)
    with pytest.raises(IllegalTransition):
        _fill(o, "f", 1)


def test_d1_overfill_is_illegal() -> None:
    o = _fill(_ack(_new(100)), "f1", 60)
    with pytest.raises(IllegalTransition, match="overfill"):
        _fill(o, "f2", 41)


def test_d1_fill_before_ack_is_illegal() -> None:
    with pytest.raises(IllegalTransition):
        _fill(_new(), "f", 1)


# --- D2 cancel ------------------------------------------------------------
def test_d2_cancel_partially_filled_order() -> None:
    o = _fill(_ack(_new()), "f1", 1000)
    o = request_cancel(o, "C2")
    assert o.ord_status is S.PARTIALLY_FILLED  # not pending until the venue says so
    o = apply_report(o, ExecReport(E.PENDING_CANCEL, "pc", "C2"))
    assert _state(o) == (S.PENDING_CANCEL, 1000, 9000)
    o = apply_report(o, ExecReport(E.CANCELED, "cx", "C2"))
    assert _state(o) == (S.CANCELED, 1000, 0)
    assert o.clordid == "C2" and o.chain == ("C1",)


def test_d2_fill_during_pending_cancel_then_cancel() -> None:
    o = request_cancel(_ack(_new()), "C2")
    o = apply_report(o, ExecReport(E.PENDING_CANCEL, "pc", "C2"))
    o = _fill(o, "f1", 1000, clordid="C1")  # venue fills against the original ClOrdID
    assert _state(o) == (S.PENDING_CANCEL, 1000, 9000)
    o = apply_report(o, ExecReport(E.CANCELED, "cx", "C2"))
    assert _state(o) == (S.CANCELED, 1000, 0)


def test_d2_order_fills_during_pending_cancel_then_cancel_rejected() -> None:
    o = request_cancel(_ack(_new(100)), "C2")
    o = apply_report(o, ExecReport(E.PENDING_CANCEL, "pc", "C2"))
    o = _fill(o, "f1", 100, clordid="C1")
    assert _state(o) == (S.FILLED, 100, 0)
    with pytest.raises(IllegalTransition):
        apply_report(o, ExecReport(E.CANCELED, "cx", "C2"))
    o = apply_cancel_reject(o, "C2")
    assert _state(o) == (S.FILLED, 100, 0) and o.outstanding is None


def test_d2_cancel_reject_returns_to_prior_status() -> None:
    o = _fill(_ack(_new()), "f1", 500)
    o = request_cancel(o, "C2")
    o = apply_report(o, ExecReport(E.PENDING_CANCEL, "pc", "C2"))
    o = apply_cancel_reject(o, "C2")
    assert _state(o) == (S.PARTIALLY_FILLED, 500, 9500) and o.clordid == "C1"


def test_d2_cancel_before_ack() -> None:
    o = request_cancel(_new(), "C2")
    o = apply_report(o, ExecReport(E.CANCELED, "cx", "C2"))
    assert _state(o) == (S.CANCELED, 0, 0)


def test_d2_cancel_of_filled_order_is_refused_locally() -> None:
    o = _fill(_ack(_new(10)), "f", 10)
    with pytest.raises(IllegalTransition):
        request_cancel(o, "C2")


def test_d2_pending_cancel_without_request_is_illegal() -> None:
    with pytest.raises(IllegalTransition):
        apply_report(_ack(_new()), ExecReport(E.PENDING_CANCEL, "pc", "C1"))


# --- D3 replace -----------------------------------------------------------
def test_d3_replace_increase_qty() -> None:
    o = _fill(_ack(_new()), "f1", 1000)
    o = request_replace(o, "C2", order_qty=12_000, price=10.0)
    o = apply_report(o, ExecReport(E.PENDING_REPLACE, "pr", "C2"))
    assert _state(o) == (S.PENDING_REPLACE, 1000, 9000)
    o = apply_report(o, ExecReport(E.REPLACED, "rp", "C2", order_qty=12_000, price=10.0))
    assert _state(o) == (S.PARTIALLY_FILLED, 1000, 11_000)
    assert o.clordid == "C2" and o.chain == ("C1",)


def test_d3_replace_down_to_cum_qty_fills_order() -> None:
    o = _fill(_ack(_new()), "f1", 3000)
    o = request_replace(o, "C2", order_qty=3000, price=10.0)
    o = apply_report(o, ExecReport(E.REPLACED, "rp", "C2"))
    assert _state(o) == (S.FILLED, 3000, 0)


def test_d3_replace_below_cum_qty_is_illegal() -> None:
    o = _fill(_ack(_new()), "f1", 3000)
    o = request_replace(o, "C2", order_qty=2000, price=10.0)
    with pytest.raises(IllegalTransition, match="below CumQty"):
        apply_report(o, ExecReport(E.REPLACED, "rp", "C2"))


def test_d3_fill_during_pending_replace() -> None:
    o = request_replace(_ack(_new()), "C2", order_qty=8000, price=10.0)
    o = apply_report(o, ExecReport(E.PENDING_REPLACE, "pr", "C2"))
    o = _fill(o, "f1", 1000, clordid="C1")
    assert _state(o) == (S.PENDING_REPLACE, 1000, 9000)
    o = apply_report(o, ExecReport(E.REPLACED, "rp", "C2"))
    assert _state(o) == (S.PARTIALLY_FILLED, 1000, 7000)


def test_d3_two_replaces_chain_clordids() -> None:
    o = _ack(_new())
    o = apply_report(request_replace(o, "C2", order_qty=9000, price=10.0), ExecReport(E.REPLACED, "r1", "C2"))
    o = apply_report(request_replace(o, "C3", order_qty=8000, price=9.0), ExecReport(E.REPLACED, "r2", "C3"))
    assert o.clordid == "C3" and o.chain == ("C1", "C2") and o.order_qty == 8000 and o.price == 9.0


def test_d3_second_request_while_outstanding_is_illegal() -> None:
    o = request_replace(_ack(_new()), "C2", order_qty=9000, price=10.0)
    with pytest.raises(IllegalTransition, match="outstanding"):
        request_cancel(o, "C3")


def test_d3_reused_clordid_is_illegal() -> None:
    o = apply_report(
        request_replace(_ack(_new()), "C2", order_qty=9000, price=10.0), ExecReport(E.REPLACED, "r", "C2")
    )
    with pytest.raises(IllegalTransition, match="fresh"):
        request_cancel(o, "C1")


# --- D4 unsolicited -------------------------------------------------------
def test_d4_unsolicited_cancel() -> None:
    o = apply_report(_fill(_ack(_new()), "f", 10), ExecReport(E.CANCELED, "u", "C1"))
    assert _state(o) == (S.CANCELED, 10, 0)


def test_d4_expired_and_done_for_day() -> None:
    o = _fill(_ack(_new()), "f", 10)
    assert _state(apply_report(o, ExecReport(E.EXPIRED, "x", "C1"))) == (S.EXPIRED, 10, 0)
    assert _state(apply_report(o, ExecReport(E.DONE_FOR_DAY, "d", "C1"))) == (S.DONE_FOR_DAY, 10, 0)


def test_d4_terminal_is_absorbing() -> None:
    o = apply_report(_ack(_new()), ExecReport(E.CANCELED, "u", "C1"))
    for et in (E.NEW, E.CANCELED, E.EXPIRED, E.DONE_FOR_DAY, E.PENDING_CANCEL):
        with pytest.raises(IllegalTransition):
            apply_report(o, ExecReport(et, "z", "C1"))


# --- D5 restatement -------------------------------------------------------
def test_d5_restated_qty_down() -> None:
    o = _fill(_ack(_new()), "f", 1000)
    o = apply_report(o, ExecReport(E.RESTATED, "rs", "C1", order_qty=5000))
    assert _state(o) == (S.PARTIALLY_FILLED, 1000, 4000)


# --- D6 trade cancel ------------------------------------------------------
def test_d6_trade_cancel_reopens_filled_order() -> None:
    o = _fill(_fill(_ack(_new(100)), "f1", 40, 10.0), "f2", 60, 11.0)
    assert o.ord_status is S.FILLED
    o = apply_report(o, ExecReport(E.TRADE_CANCEL, "h1", "C1", exec_ref_id="f2"))
    assert _state(o) == (S.PARTIALLY_FILLED, 40, 60) and o.avg_px == 10.0


def test_d6_trade_cancel_unknown_fill_is_illegal() -> None:
    with pytest.raises(IllegalTransition):
        apply_report(_ack(_new()), ExecReport(E.TRADE_CANCEL, "h", "C1", exec_ref_id="nope"))


# --- dedupe and foreign ClOrdID ------------------------------------------
def test_duplicate_execid_is_noop() -> None:
    o = _fill(_ack(_new()), "f1", 100)
    assert _fill(o, "f1", 100) is o


def test_foreign_clordid_is_illegal() -> None:
    with pytest.raises(IllegalTransition):
        apply_report(_ack(_new()), ExecReport(E.TRADE, "f", "OTHER", last_qty=1, last_px=1.0))


def test_order_new_validates() -> None:
    for kw in ({"order_qty": 0}, {"side": "9"}):
        args = {"side": "1", "symbol": "X", "order_qty": 1, "price": None} | kw
        with pytest.raises(ValueError):
            Order.new("C", **args)  # type: ignore[arg-type]


# --- OrderBook: log-then-apply and replay --------------------------------
def test_book_illegal_event_logs_nothing() -> None:
    book = OrderBook(EventLog.in_memory())
    book.submit("A", side="1", symbol="X", order_qty=10, price=1.0)
    n = len(book.log)
    with pytest.raises(IllegalTransition):
        book.on_report(ExecReport(E.TRADE, "f", "A", last_qty=1, last_px=1.0))
    assert len(book.log) == n and book.get("A").cum_qty == 0


def test_book_duplicate_clordid_rejected() -> None:
    book = OrderBook(EventLog.in_memory())
    book.submit("A", side="1", symbol="X", order_qty=10, price=1.0)
    with pytest.raises(IllegalTransition):
        book.submit("A", side="1", symbol="X", order_qty=10, price=1.0)


def test_book_replay_from_disk(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "oms.jsonl"
    with EventLog(p) as log:
        book = OrderBook(log)
        book.submit("A", side="2", symbol="X", order_qty=10, price=1.0, ts_ns=1)
        book.on_report(ExecReport(E.NEW, "e0", "A"), ts_ns=2)
        book.on_report(ExecReport(E.TRADE, "e1", "A", last_qty=4, last_px=1.5), ts_ns=3)
        book.request_replace("A", "B", order_qty=8, price=1.0, ts_ns=4)
        book.on_report(ExecReport(E.REPLACED, "e2", "B"), ts_ns=5)
        book.request_cancel("B", "C", ts_ns=6)
        book.on_cancel_reject("C", ts_ns=7)
        live = book.snapshot()
    with EventLog(p) as again:
        assert OrderBook.replay(again).snapshot() == live
    assert live["A"].clordid == "B" and live["A"].leaves_qty == 4


# --- property tests -------------------------------------------------------
_ACTIONS = st.lists(
    st.tuples(
        st.sampled_from(
            [
                "ack",
                "fill",
                "cancel_req",
                "replace_req",
                "pend",
                "cancel",
                "replaced",
                "creject",
                "expire",
                "bust",
                "dup",
            ]
        ),
        st.integers(min_value=1, max_value=60),
    ),
    max_size=40,
)


@settings(max_examples=300)
@given(_ACTIONS)
def test_property_random_sequences_hold_invariants(actions: list[tuple[str, int]]) -> None:
    o = Order.new("K0", side="1", symbol="X", order_qty=100, price=1.0)
    n = 0
    for act, k in actions:
        n += 1
        eid = f"e{n}"
        before = o
        try:
            if act == "ack":
                o = apply_report(o, ExecReport(E.NEW, eid, o.clordid))
            elif act == "fill":
                o = apply_report(o, ExecReport(E.TRADE, eid, o.clordid, last_qty=k, last_px=1.0 + k / 100))
            elif act == "cancel_req":
                o = request_cancel(o, f"K{n}")
            elif act == "replace_req":
                o = request_replace(o, f"K{n}", order_qty=k * 2, price=1.0)
            elif act == "pend" and o.outstanding is not None:
                et = E.PENDING_CANCEL if o.outstanding.kind == "cancel" else E.PENDING_REPLACE
                o = apply_report(o, ExecReport(et, eid, o.outstanding.clordid))
            elif act == "cancel":
                cid = o.outstanding.clordid if o.outstanding and o.outstanding.kind == "cancel" else o.clordid
                o = apply_report(o, ExecReport(E.CANCELED, eid, cid))
            elif act == "replaced" and o.outstanding is not None:
                o = apply_report(o, ExecReport(E.REPLACED, eid, o.outstanding.clordid))
            elif act == "creject" and o.outstanding is not None:
                o = apply_cancel_reject(o, o.outstanding.clordid)
            elif act == "expire":
                o = apply_report(o, ExecReport(E.EXPIRED, eid, o.clordid))
            elif act == "bust" and o.fills:
                ref = sorted(o.fills)[k % len(o.fills)]
                o = apply_report(o, ExecReport(E.TRADE_CANCEL, eid, o.clordid, exec_ref_id=ref))
            elif act == "dup" and o.exec_ids:
                ref = sorted(o.exec_ids)[k % len(o.exec_ids)]
                assert apply_report(o, ExecReport(E.TRADE, ref, o.clordid, last_qty=1, last_px=1.0)) is o
        except IllegalTransition:
            assert o is before  # failed transitions change nothing
        o.check()
        if o.is_working:
            assert o.cum_qty + o.leaves_qty == o.order_qty
        else:
            assert o.leaves_qty == 0
        assert o.ord_status in set(OrdStatus)
        if before.is_closed and before.phase is not o.phase:
            raise AssertionError("terminal phase changed")


@settings(max_examples=100)
@given(st.lists(st.integers(min_value=1, max_value=30), max_size=20))
def test_property_book_replay_equals_live(qtys: list[int]) -> None:
    book = OrderBook(EventLog.in_memory())
    book.submit("R", side="1", symbol="X", order_qty=200, price=1.0)
    book.on_report(ExecReport(E.NEW, "n", "R"))
    for i, q in enumerate(qtys):
        try:
            book.on_report(ExecReport(E.TRADE, f"t{i}", "R", last_qty=q, last_px=1.0 + i))
        except IllegalTransition:
            pass
    assert OrderBook.replay(book.log).snapshot() == book.snapshot()
