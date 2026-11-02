"""Limits: utilisation, warning and breach, aggregation, two-sided limits."""

import pytest
from eq_var.limits import Limit, breaches, check_limits


def tree():
    return Limit("desk", 100.0, children=[
        Limit("book_a", 60.0),
        Limit("book_b", 50.0, children=[Limit("book_b_delta", 1000.0, two_sided=True)]),
    ])


def test_statuses_and_order():
    st = check_limits(tree(), {"book_a": 45.0, "book_b": 52.0, "book_b_delta": -850.0})
    by = {s.name: s for s in st}
    assert [s.name for s in st] == ["desk", "book_a", "book_b", "book_b_delta"]
    assert by["desk"].value == 97.0 and by["desk"].status == "warning"
    assert by["book_a"].utilisation == 0.75 and by["book_a"].status == "ok"
    assert by["book_b"].status == "breach"
    assert by["book_b_delta"].utilisation == 0.85 and by["book_b_delta"].status == "warning"
    assert [s.name for s in breaches(st)] == ["book_b"]


def test_explicit_parent_value_overrides_sum():
    st = check_limits(tree(), {"desk": 70.0, "book_a": 45.0, "book_b": 52.0, "book_b_delta": 0.0})
    assert st[0].value == 70.0 and st[0].status == "ok"


def test_exact_limit_is_not_breach():
    assert check_limits(Limit("x", 10.0), {"x": 10.0})[0].status == "warning"


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        check_limits(tree(), {"book_a": 1.0})
    with pytest.raises(ValueError):
        Limit("x", 0.0)
    with pytest.raises(ValueError):
        Limit("x", 1.0, warning=1.5)
