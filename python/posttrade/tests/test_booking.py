from __future__ import annotations

from datetime import date, datetime

import pytest
from qp_posttrade.booking import Blotter, Confirm, match_confirms
from qp_posttrade.model import Fill


def _f(eid: str, qty: float, px: float, side: str = "BUY", sym: str = "XYZ", day: int = 2) -> Fill:
    return Fill(eid, "A1", sym, side, qty, px, datetime(2026, 11, day, 15), "V1")  # type: ignore[arg-type]


def test_book_groups_vwap_and_t_plus_1() -> None:
    b = Blotter("US_EQUITY")
    (t,) = b.book([_f("e1", 100, 10.0), _f("e2", 300, 10.4), _f("e1", 100, 10.0)], "CPTY")
    assert t.qty == 400
    assert t.avg_px == pytest.approx(10.3)
    assert t.value_date == date(2026, 11, 3)
    assert t.exec_ids == ("e1", "e2")  # duplicate drop-copy resend booked once


def test_fx_booking_uses_spot() -> None:
    b = Blotter("FX_SPOT")
    (t,) = b.book([_f("e1", 1e6, 1.08, sym="EURUSD", day=9)], "BANK")
    assert t.value_date == date(2026, 11, 12)


def test_lifecycle_append_only() -> None:
    b = Blotter("US_EQUITY")
    (t,) = b.book([_f("e1", 100, 10.0)], "CPTY")
    b.amend(t.trade_id, qty=90)
    b.novate(t.trade_id, "CCP")
    c = b.cancel(t.trade_id)
    h = b.history[t.trade_id]
    assert [x.version for x in h] == [1, 2, 3, 4]
    assert h[0].qty == 100 and h[1].qty == 90
    assert h[2].counterparty == "CCP" and h[2].qty == 90
    assert c.status == "CANCELLED" and b.live() == []
    with pytest.raises(ValueError):
        b.amend(t.trade_id, qty=1)


def test_novation_must_change_counterparty_and_amend_needs_field() -> None:
    b = Blotter("US_EQUITY")
    (t,) = b.book([_f("e1", 100, 10.0)], "CPTY")
    with pytest.raises(ValueError):
        b.novate(t.trade_id, "CPTY")
    with pytest.raises(ValueError):
        b.amend(t.trade_id)


def test_confirm_matching_finds_every_break_kind() -> None:
    b = Blotter("US_EQUITY")
    ts = b.book(
        [
            _f("e1", 100, 10.0),
            _f("e2", 200, 20.0, sym="ABC"),
            _f("e3", 50, 5.0, sym="QQQ"),
            _f("e4", 10, 1.0, sym="ZZZ"),
        ],
        "CPTY",
    )
    ids = {t.symbol: t.trade_id for t in ts}
    vd = date(2026, 11, 3)
    td = date(2026, 11, 2)
    confirms = [
        Confirm("c1", "A1", "XYZ", "BUY", 100, 10.0, td, vd),  # clean
        Confirm("c2", "A1", "ABC", "BUY", 200, 20.01, td, vd),  # price break
        Confirm("c3", "A1", "QQQ", "BUY", 50, 5.0, td, date(2026, 11, 4)),  # value-date break
        Confirm("c4", "A1", "NOP", "BUY", 1, 1.0, td, vd),  # we have no such trade
    ]
    r = match_confirms(b.live(), confirms)
    assert r.matched == [(ids["XYZ"], "c1")]
    assert (ids["ABC"], "c2", ("px",)) in r.mismatched
    assert (ids["QQQ"], "c3", ("value_date",)) in r.mismatched
    assert r.unmatched_ours == [ids["ZZZ"]]
    assert r.unmatched_theirs == ["c4"]


def test_bad_fill_rejected() -> None:
    with pytest.raises(ValueError):
        _f("x", 100, 1.0, side="HOLD")
    with pytest.raises(ValueError):
        _f("x", 0, 1.0)
