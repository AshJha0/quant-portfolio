"""Spec cases for every order type of the reference book (lob/SPEC.md)."""

from __future__ import annotations

import pytest
from qp_exec.lob import Order, OrdType, Peg, Phase, ReferenceBook, Reject, Side, Tif
from qp_exec.lob.events import random_events, replay_digests

B, S = Side.BUY, Side.SELL


def book_with(*orders: Order) -> ReferenceBook:
    b = ReferenceBook()
    for i, o in enumerate(orders):
        assert b.add(o, ts=i + 1).accepted
    return b


def test_price_then_time_priority() -> None:
    b = book_with(Order(1, S, 5, 101), Order(2, S, 5, 100), Order(3, S, 5, 100))
    r = b.add(Order(9, B, 12, 101), ts=10)
    assert [(f.maker_id, f.price, f.qty) for f in r.fills] == [(2, 100, 5), (3, 100, 5), (1, 101, 2)]
    assert r.resting_qty == 0
    assert b.order_state(1) == (101, 3, 0, True)


def test_limit_day_rests_remainder() -> None:
    b = book_with(Order(1, S, 5, 101))
    r = b.add(Order(2, B, 8, 101), ts=5)
    assert r.resting_qty == 3 and b.best_bid() == 101 and b.best_ask() is None


def test_ioc_cancels_remainder() -> None:
    b = book_with(Order(1, S, 5, 101))
    r = b.add(Order(2, B, 8, 102, tif=Tif.IOC), ts=5)
    assert sum(f.qty for f in r.fills) == 5 and r.resting_qty == 0
    assert b.order_state(2) is None


def test_fok_all_or_nothing_counts_hidden_and_reserve() -> None:
    b = book_with(Order(1, S, 2, 101), Order(2, S, 2, 101, hidden=True),
                  Order(3, S, 6, 102, display_qty=1))
    assert b.add(Order(9, B, 11, 102, tif=Tif.FOK), 9).reject == Reject.FOK_UNFILLABLE
    assert b.total_resting() == 10  # untouched
    r = b.add(Order(10, B, 10, 102, tif=Tif.FOK), 10)
    assert r.accepted and sum(f.qty for f in r.fills) == 10 and b.total_resting() == 0


def test_market_needs_ioc_or_fok_and_sweeps() -> None:
    b = book_with(Order(1, S, 3, 101), Order(2, S, 3, 150))
    assert b.add(Order(8, B, 1, ord_type=OrdType.MARKET), 8).reject == Reject.INVALID
    r = b.add(Order(9, B, 10, ord_type=OrdType.MARKET, tif=Tif.IOC), 9)
    assert [f.price for f in r.fills] == [101, 150] and r.resting_qty == 0


def test_post_only_rejects_on_hidden_cross_too() -> None:
    b = book_with(Order(1, S, 3, 101, hidden=True))
    assert b.best_ask() is None  # nothing displayed
    assert b.add(Order(2, B, 1, 101, post_only=True), 2).reject == Reject.POST_ONLY_WOULD_CROSS
    assert b.add(Order(3, B, 1, 100, post_only=True), 3).accepted


def test_displayed_before_hidden_at_same_price() -> None:
    b = book_with(Order(1, S, 3, 101, hidden=True), Order(2, S, 3, 101))
    r = b.add(Order(3, B, 4, 101, tif=Tif.IOC), 3)
    assert [(f.maker_id, f.qty) for f in r.fills] == [(2, 3), (1, 1)]


def test_iceberg_refill_goes_to_back_within_sweep() -> None:
    b = book_with(Order(1, S, 10, 100, display_qty=2), Order(2, S, 3, 100))
    assert b.depth(S, 1) == [(100, 5)]
    r = b.add(Order(3, B, 7, 100, tif=Tif.IOC), 9)
    # slice of 1 (2), then 2 (3), then refilled slice of 1 (2)
    assert [(f.maker_id, f.qty) for f in r.fills] == [(1, 2), (2, 3), (1, 2)]
    assert b.order_state(1) == (100, 2, 4, True)


def test_midpoint_peg_rounds_passively_and_tracks() -> None:
    b = book_with(Order(1, B, 5, 100), Order(2, S, 5, 105))
    b.add(Order(3, B, 4, peg=Peg.MID), 3)
    b.add(Order(4, S, 4, peg=Peg.MID), 4)
    assert b.order_state(3)[0] == 102 and b.order_state(4)[0] == 103
    assert b.best_bid() == 100  # midpoint pegs are not displayed
    b.cancel(2, 5)  # no ask: both pegs go inactive
    assert b.order_state(3)[3] is False and b.order_state(4)[3] is False
    r = b.add(Order(5, S, 1, 104), 6)
    # both reprice to the integer midpoint 102 and trade; 3 is older, so maker
    assert [(f.taker_id, f.maker_id, f.price, f.qty) for f in r.fills] == [(4, 3, 102, 4)]
    assert b.order_state(3) is None and b.order_state(4) is None
    b.check_invariants()


def test_midpoint_pegs_cross_at_integer_mid() -> None:
    b = book_with(Order(1, B, 5, 100), Order(2, S, 5, 104),
                  Order(3, B, 4, peg=Peg.MID))
    r = b.add(Order(4, S, 3, peg=Peg.MID), 9)
    assert [(f.taker_id, f.maker_id, f.price, f.qty) for f in r.fills] == [(4, 3, 102, 3)]
    assert b.order_state(3) == (102, 1, 0, True) and b.order_state(4) is None


def test_primary_peg_joins_best_and_follows() -> None:
    b = book_with(Order(1, B, 5, 100), Order(2, S, 5, 105))
    b.add(Order(3, B, 2, peg=Peg.PRIMARY), 3)
    assert b.depth(B, 1) == [(100, 7)] and b.queue_ahead(3) == 5
    b.add(Order(4, B, 1, 101), 4)
    assert b.order_state(3)[0] == 101 and b.queue_ahead(3) == 1


def test_modify_down_keeps_priority_up_loses_it() -> None:
    b = book_with(Order(1, S, 5, 100), Order(2, S, 5, 100))
    b.modify(1, 2)
    assert b.queue_ahead(2) == 2
    b.modify(1, 9)
    assert b.queue_ahead(1) == 5
    b.modify(2, 0)
    assert b.order_state(2) is None


def test_invalid_combinations() -> None:
    b = ReferenceBook()
    bad = [Order(1, B, 0, 100), Order(2, B, 1, 0), Order(3, B, 1, 100, hidden=True, display_qty=1),
           Order(4, B, 1, 100, post_only=True, tif=Tif.IOC), Order(5, B, 1, peg=Peg.MID, tif=Tif.IOC),
           Order(6, B, 1, 100, hidden=True, tif=Tif.IOC)]
    for o in bad:
        assert b.add(o, 1).reject == Reject.INVALID, o
    assert b.add(Order(7, B, 1, 100), 1).accepted
    assert b.add(Order(7, B, 1, 100), 1).reject == Reject.DUPLICATE_ID


def test_auction_uncross_max_volume_then_imbalance_then_reference() -> None:
    b = ReferenceBook()
    b.start_auction()
    assert b.add(Order(1, B, 1, 100, tif=Tif.IOC), 1).reject == Reject.NOT_IN_AUCTION
    for o in [Order(1, B, 5, 103), Order(2, B, 5, 101), Order(3, S, 4, 100), Order(4, S, 4, 102)]:
        assert b.add(o, 1).accepted
    # volume at 101: D=10,S=4 ->4 ; at 102: D=5,S=8 ->5 ; at 103: D=5,S=8 ->5 ; tie -> imbalance equal -> ref
    assert b.auction_price(ref_price=103) == (103, 5)
    assert b.auction_price(ref_price=100) == (102, 5)
    fills = b.uncross(ts=9, ref_price=100)
    assert [(f.taker_id, f.maker_id, f.price, f.qty) for f in fills] == [(1, 3, 102, 4), (1, 4, 102, 1)]
    assert b.phase == Phase.CONTINUOUS
    b.check_invariants()
    assert b.best_bid() == 101 and b.depth(S, 1) == [(102, 3)]


def test_random_stream_keeps_invariants_and_conserves_quantity() -> None:
    """Every lot is accounted for: resting = submitted - filled, per order."""
    from qp_exec.lob.events import ADD, CANCEL, MODIFY, apply_event
    b = ReferenceBook()
    rem: dict[int, int] = {}
    traded = 0
    for ev in random_events(20000, seed=7):
        op, oid, qty = ev[0], ev[2], ev[4]
        if op == ADD:
            before = set(b._orders)
        fills = apply_event(b, ev)
        b.check_invariants()
        if op == ADD and oid not in before and (oid in b._orders or fills):
            rem[oid] = qty
        elif op == ADD and oid not in before:
            # rejected, or an IOC/FOK that found nothing: nothing to track
            pass
        if op == MODIFY and oid in rem and b.order_state(oid) is not None:
            rem[oid] = qty
        for f in fills:
            rem[f.taker_id] -= f.qty
            rem[f.maker_id] -= f.qty
            traded += f.qty
        for k in list(rem):
            st = b.order_state(k)
            assert rem[k] >= 0
            if st is None:
                del rem[k]
            else:
                assert st[1] + st[2] == rem[k], (ev, k, st, rem[k])
        if op == CANCEL:
            assert b.order_state(oid) is None
    assert traded > 1000
    assert sum(rem.values()) == b.total_resting()


@pytest.mark.parametrize("seed", [1, 2])
def test_digests_deterministic(seed: int) -> None:
    ev = list(random_events(3000, seed))
    assert replay_digests(ev) == replay_digests(ev, check=True)
