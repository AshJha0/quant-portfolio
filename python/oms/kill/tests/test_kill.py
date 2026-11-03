import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qp_oms.kill import KillState, KillSwitch, Level, covers, make_scope, replay, validate_scope

P7 = ("QP", "RATES", "MM1", "UST10Y")
DESK = ("QP", "RATES", "*", "*")


def test_worked_example_from_contract() -> None:
    ks = KillSwitch()
    ks.on_order_working(7, P7)
    kid, cancels = ks.engage(DESK, "alice")
    assert cancels == [7] and ks.state(kid) is KillState.ENGAGED
    assert ks.check(("QP", "RATES", "MM2", "UST2Y")) == "KILLED:DESK:QP/RATES/*/*"
    assert not ks.request_reset(kid, "alice")  # cancels outstanding
    ks.on_order_terminal(7)
    assert ks.state(kid) is KillState.KILLED
    assert ks.request_reset(kid, "alice")
    assert not ks.approve_reset(kid, "alice")  # K4
    assert ks.state(kid) is KillState.RESET_PENDING
    assert ks.approve_reset(kid, "bob")
    assert ks.check(P7) is None


def test_scope_validation() -> None:
    assert validate_scope(("QP", "*", "*", "*")) is Level.FIRM
    with pytest.raises(ValueError):
        validate_scope(("QP", "*", "MM1", "*"))
    with pytest.raises(ValueError):
        validate_scope(("*", "*", "*", "*"))


def test_nesting_k5_and_idempotent_k7() -> None:
    ks = KillSwitch()
    firm, _ = ks.engage(("QP", "*", "*", "*"), "a")
    ks.engage(make_scope(Level.SYMBOL, P7), "a")
    assert ks.engage(("QP", "*", "*", "*"), "a") == (firm, [])
    assert ks.check(P7) == "KILLED:FIRM:QP/*/*/*"
    ks.request_reset(firm, "a")
    ks.approve_reset(firm, "b")
    assert ks.check(P7) == "KILLED:SYMBOL:QP/RATES/MM1/UST10Y"
    assert ks.check(("QP", "RATES", "MM1", "UST2Y")) is None


def test_cancel_on_disconnect_k6() -> None:
    ks = KillSwitch()
    ks.on_order_working(1, P7)
    ks.on_order_working(2, ("QP", "RATES", "MM1", "UST2Y"))
    ks.on_order_working(3, ("QP", "FX", "MM9", "EURUSD"))
    res = ks.cancel_on_disconnect([P7, ("QP", "RATES", "MM1", "UST2Y")])
    assert len(res) == 1 and res[0][1] == [1, 2]
    assert ks.check(("QP", "FX", "MM9", "EURUSD")) is None


def test_fill_racing_cancel_counts_as_terminal_k3() -> None:
    ks = KillSwitch()
    ks.on_order_working(1, P7)
    ks.on_order_working(2, P7)
    kid, _ = ks.engage(DESK, "a")
    ks.on_order_terminal(2)  # filled before cancel arrived
    assert ks.state(kid) is KillState.ENGAGED
    ks.on_order_terminal(1)
    assert ks.state(kid) is KillState.KILLED


paths = st.tuples(
    st.just("QP"), st.sampled_from(["R", "F"]), st.sampled_from(["S1", "S2"]), st.sampled_from(["A", "B"])
)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 3), paths, st.sampled_from(list(Level))), max_size=40))
def test_property_no_covered_order_survives_a_kill(ops: list) -> None:
    ks = KillSwitch()
    working: dict[int, tuple] = {}
    nid = 0
    for kind, p, lvl in ops:
        if kind <= 1:
            if ks.check(p) is None:
                nid += 1
                working[nid] = p
                ks.on_order_working(nid, p)
        elif kind == 2:
            scope = make_scope(lvl, p)
            _, cancels = ks.engage(scope, "x")
            for o in cancels:
                assert covers(scope, working[o])
                del working[o]
                ks.on_order_terminal(o)
            # invariant: no working order is covered by an active kill
            for o, wp in working.items():
                assert ks.check(wp) is None, (o, wp)
        elif working:
            o = min(working)
            del working[o]
            ks.on_order_terminal(o)


def test_golden_vectors() -> None:
    v = json.loads((Path(__file__).parents[1] / "vectors.json").read_text(encoding="utf-8"))
    for case in v["cases"]:
        assert replay(case["commands"]) == case["expected"], case["name"]
