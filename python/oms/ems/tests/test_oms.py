import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qp_oms.ems import ExecReport, Oms, ParentOrder, twap_slices
from qp_oms.kill import KillSwitch
from qp_oms.risk import Limits, RiskBook, Side

LIM = Limits(1_000, 1e9, 0.1, 1e6, 1e12, 1_000, 1)


def mk(qty: float = 100, alloc: dict[str, float] | None = None) -> Oms:
    oms = Oms("QP", RiskBook(LIM, ref_px={"X": 100.0}), KillSwitch())
    oms.new_parent(ParentOrder("P1", "RATES", "MM1", "X", Side.BUY, qty, alloc or {"a1": 0.5, "a2": 0.5}))
    return oms


def test_twap_slices_sum_and_whole_lots() -> None:
    assert twap_slices(10, 3) == [4.0, 3.0, 3.0]
    assert twap_slices(2, 5) == [1.0, 1.0]


def test_children_fills_roll_up_and_dedupe() -> None:
    oms = mk()
    kids = oms.work("P1", 4, 100.0, 0)
    assert [k.qty for k in kids] == [25.0] * 4
    assert oms.on_exec(ExecReport("e1", kids[0].cl_ord_id, 25, 100.0, terminal=True))
    assert not oms.on_exec(ExecReport("e1", kids[0].cl_ord_id, 25, 100.0, terminal=True))  # PossDup
    assert oms.on_exec(ExecReport("e2", kids[1].cl_ord_id, 10, 101.0))
    p = oms.parents["P1"]
    assert p.cum_qty == 35 and p.avg_px == pytest.approx((25 * 100 + 10 * 101) / 35)
    assert oms.positions[("QP:RATES", "X")] == 35
    assert oms.work("P1", 2, 100.0, 1) == []  # nothing unworked: 25 filled + 75 live


def test_overfill_raises() -> None:
    oms = mk(qty=10)
    (k,) = oms.work("P1", 1, 100.0, 0)
    with pytest.raises(ValueError, match="overfill"):
        oms.on_exec(ExecReport("e1", k.cl_ord_id, 11, 100.0))


def test_allocation_largest_remainder_hand_example() -> None:
    # 3 accounts at 1/3 on 100 lots: 34, 33, 33 (a1 wins the tie by name).
    oms = mk(alloc={"a1": 1 / 3, "a2": 1 / 3, "a3": 1 / 3})
    (k,) = oms.work("P1", 1, 100.0, 0)
    oms.on_exec(ExecReport("e1", k.cl_ord_id, 100, 99.5, terminal=True))
    assert oms.allocate("P1") == {"a1": (34.0, 99.5), "a2": (33.0, 99.5), "a3": (33.0, 99.5)}


def test_kill_stops_sends_and_all_working_cancelled_and_acked() -> None:
    """W28 acceptance: a kill stops sends, and all working orders are cancelled and acked."""
    oms = mk()
    kids = oms.work("P1", 4, 100.0, 0)
    oms.on_exec(ExecReport("e1", kids[0].cl_ord_id, 25, 100.0, terminal=True))
    to_cancel = oms.kill_and_cancel(("QP", "RATES", "*", "*"), "alice")
    assert to_cancel == [k.cl_ord_id for k in kids[1:]]
    n_sent = len(oms.sent)
    oms.new_parent(ParentOrder("P2", "RATES", "MM2", "X", Side.SELL, 10, {"a1": 1.0}))
    assert oms.work("P2", 1, 100.0, 5) == [] and len(oms.sent) == n_sent
    assert oms.rejects[-1][1] == "KILLED:DESK:QP/RATES/*/*"
    for i, c in enumerate(to_cancel):
        oms.on_exec(ExecReport(f"x{i}", c, 0, 0.0, terminal=True))
    assert oms.kill.state(1).value == "KILLED"
    assert oms.risk.working_notional("QP:RATES") == 0


@settings(max_examples=200, deadline=None)
@given(st.integers(1, 500), st.integers(1, 7), st.lists(st.floats(0.01, 1), min_size=1, max_size=5))
def test_property_allocation_sums_exactly(qty: int, n: int, w: list[float]) -> None:
    s = sum(w)
    alloc = {f"a{i}": x / s for i, x in enumerate(w)}
    alloc[f"a{len(w) - 1}"] += 1 - sum(alloc.values())
    oms = mk(qty=qty, alloc=alloc)
    for i, k in enumerate(oms.work("P1", n, 100.0, 0)):
        oms.on_exec(ExecReport(f"e{i}", k.cl_ord_id, k.qty, 100.0 + i, terminal=True))
    a = oms.allocate("P1")
    assert sum(q for q, _ in a.values()) == qty
    assert all(abs(q - alloc[k] * qty) < 1 for k, (q, _) in a.items())
