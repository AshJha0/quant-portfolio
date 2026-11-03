from qp_oms.recon import break_report, reconcile


def test_clean_books_have_no_breaks() -> None:
    f = {"e1": ("A", "X", 10.0, 100.0)}
    assert reconcile(f, dict(f), {("A", "X"): 10}, {("A", "X"): 10}) == []


def test_one_missing_fill_is_one_break_not_two() -> None:
    o = {"e1": ("A", "X", 10.0, 100.0)}
    d = {"e1": ("A", "X", 10.0, 100.0), "e2": ("A", "X", -4.0, 101.0)}
    br = reconcile(o, d, {("A", "X"): 10}, {("A", "X"): 6})
    assert [(b.kind, b.key, b.explained) for b in br] == [("MISSING_IN_OMS", "e2", False)]
    br = reconcile(o, d, {("A", "X"): 10}, {("A", "X"): 6}, in_flight=frozenset({"e2"}))
    assert br[0].explained


def test_qty_px_and_unexplained_position_breaks() -> None:
    o = {"e1": ("A", "X", 10.0, 100.0), "e2": ("A", "Y", 5.0, 50.0)}
    d = {"e1": ("A", "X", 9.0, 100.0), "e2": ("A", "Y", 5.0, 50.5)}
    br = reconcile(o, d, {("A", "X"): 10, ("A", "Y"): 5, ("A", "Z"): 1}, {("A", "X"): 9, ("A", "Y"): 5})
    assert [(b.kind, b.key) for b in br] == [
        ("QTY_MISMATCH", "e1"),
        ("PX_MISMATCH", "e2"),
        ("POSITION_MISMATCH", "A|Z"),
    ]
    rep = break_report(br).splitlines()
    assert len(rep) == 3 and '"kind": "POSITION_MISMATCH"' in rep[0]
