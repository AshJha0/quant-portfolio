from hypothesis import given, settings
from hypothesis import strategies as st
from qp_oms.risk import Limits, OrderIntent, RiskBook, Side

L = Limits(
    max_order_qty=100,
    max_order_notional=20_000,
    price_band=0.05,
    position_limit=300,
    credit_limit=50_000,
    throttle_msgs=5,
    throttle_window_ns=1_000,
)


def book() -> RiskBook:
    return RiskBook(L, ref_px={"X": 100.0, "Y": 50.0})


def o(i: int, side: Side = Side.BUY, qty: float = 10, px: float = 100.0, sym: str = "X") -> OrderIntent:
    return OrderIntent(f"c{i}", "A", "S", sym, side, qty, px)


def test_each_check_names_its_reason() -> None:
    b = book()
    b.restricted.add("Y")
    assert b.check_new(o(1, sym="Y", px=50), 0) == "RESTRICTED"
    assert b.check_new(o(2, qty=101), 10_000) == "FAT_FINGER_QTY"
    assert b.check_new(o(3, qty=100, px=201), 20_000) == "MAX_NOTIONAL"
    assert b.check_new(o(4, px=105.01), 30_000) == "PRICE_BAND"
    assert b.check_new(o(5, px=95.0), 40_000) is None  # band edge inclusive
    assert b.check_new(o(5, px=104), 50_000) == "DUP_CLORDID"
    assert b.check_new(o(6, px=95.0), 60_000) == "DUP_CONTENT"
    assert b.check_new(o(7, Side.SELL, px=95), 70_000) == "SELF_TRADE"
    assert b.check_new(o(8, Side.SELL, px=96), 80_000) is None
    assert b.check_new(o(9, qty=1, px=95), 1_000_000_000_000) is None  # dup window passed
    assert b.check_new(o(10, qty=1, sym="Z"), 1_000_000_001_000) == "NO_REF_PRICE"


def test_throttle_rejects_not_queues() -> None:
    b = book()
    res = [b.check_new(o(i, qty=1, px=100 - i * 0.1), i) for i in range(7)]
    assert res[:5] == [None] * 5 and res[5:] == ["THROTTLE", "THROTTLE"]
    assert b.check_new(o(99, qty=1, px=99), 1_001) is None  # window slid past t=0,1


def test_position_limit_counts_working_orders() -> None:
    b = book()
    for i in range(3):
        assert b.check_new(o(i, qty=100, px=100 - i * 0.5), i * 2_000) is None
    assert b.check_new(o(9, qty=1, px=98), 10_000) == "POSITION_LIMIT"
    b.on_fill("c0", 100)  # fill moves working to position: still 300 worst case
    assert b.check_new(o(10, qty=1, px=97.9), 20_000) == "POSITION_LIMIT"
    b.on_terminal("c1")
    assert b.check_new(o(11, qty=1, px=97.8), 30_000) is None


def test_credit_limit() -> None:
    b = RiskBook(Limits(100, 1e9, 0.5, 1e9, 15_000, 99, 1), ref_px={"X": 100.0})
    assert b.check_new(o(1, qty=100), 0) is None
    assert b.check_new(o(2, qty=60, px=99), 0) == "CREDIT_LIMIT"


def test_amend_up_checked_and_pending_counts_max() -> None:
    b = book()
    assert b.check_new(o(1, qty=100), 0) is None
    assert b.check_new(o(2, qty=100, px=99.5), 2_000) is None
    assert b.check_amend("c1", 50, 4_000) is None  # amend down pending
    assert b.worst_case("A", "X")[0] == 200  # still counts 100 until ack
    assert b.check_new(o(3, qty=100, px=99), 6_000) is None  # 300: at the limit
    assert b.check_new(o(4, qty=1, px=98.9), 8_000) == "POSITION_LIMIT"
    b.on_amend_result("c1", accepted=True)  # now 250
    assert b.check_new(o(5, qty=50, px=98.8), 10_000) is None
    assert b.check_amend("c5", 51, 12_000) == "POSITION_LIMIT"


@settings(max_examples=300, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 5), st.integers(1, 100), st.integers(0, 9)), max_size=60))
def test_property_venue_exposure_never_exceeds_limit(ops: list[tuple[int, int, int]]) -> None:
    """A venue applies amends/cancels/fills in any order; true worst case <= limit."""
    lim = Limits(100, 1e12, 1.0, 250, 1e12, 10**9, 1)
    b = RiskBook(lim, ref_px={"X": 100.0})
    venue: dict[str, float] = {}  # open qty at venue
    pend_amend: dict[str, float] = {}
    pend_cancel: set[str] = set()
    pos = 0.0
    n = 0
    for t, (kind, q, pick) in enumerate(ops):
        ids = sorted(venue)
        tgt = ids[pick % len(ids)] if ids else None
        if kind == 0:
            n += 1
            if (
                b.check_new(OrderIntent(f"c{n}", "A", "S", "X", Side.BUY, q, 100 - n * 1e-3), t * 10**10)
                is None
            ):
                venue[f"c{n}"] = q
        elif tgt is None:
            continue
        elif kind == 1 and tgt not in pend_amend and tgt not in pend_cancel:
            if b.check_amend(tgt, q, t * 10**10) is None:
                pend_amend[tgt] = q
        elif kind == 2:
            pend_cancel.add(tgt)
        elif kind == 3 and tgt in pend_amend:  # venue applies amend
            venue[tgt] = pend_amend.pop(tgt)
            b.on_amend_result(tgt, True)
        elif kind == 4:  # partial fill
            f = min(venue[tgt], q / 10)
            venue[tgt] -= f
            pos += f
            if tgt in pend_amend:
                pend_amend[tgt] = max(0.0, pend_amend[tgt] - f)
            b.on_fill(tgt, f)
            if venue[tgt] <= 1e-12 and tgt not in pend_amend:
                del venue[tgt]
                pend_cancel.discard(tgt)
        elif kind == 5 and tgt in pend_cancel:
            del venue[tgt]
            pend_cancel.discard(tgt)
            pend_amend.pop(tgt, None)
            b.on_terminal(tgt)
        # true worst case: venue qty, or pending amend qty that the venue may still apply
        true_worst = pos + sum(max(v, pend_amend.get(k, 0.0)) for k, v in venue.items())
        assert true_worst <= lim.position_limit + 1e-9


def test_ems_slices_exempt_from_content_dup_but_not_clordid() -> None:
    b = book()
    a = OrderIntent("p.1", "A", "S", "X", Side.BUY, 10, 100.0, "p")
    assert b.check_new(a, 0) is None
    assert b.check_new(OrderIntent("p.2", "A", "S", "X", Side.BUY, 10, 100.0, "p"), 1) is None
    assert b.check_new(a, 2) == "DUP_CLORDID"
