"""W10 stores: bitemporal trade log, immutable snapshots, exact restatement."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from qp_core.errors import DataError, InputError
from qp_core.instruments import Cash, EquityOption, FixedCashflows, FXForward, Trade
from qp_core.market import DiscountCurve, MarketSnapshot, VolSurface
from qp_core.store import PVStore, SnapshotStore, TradeStore, restate

D1, D2, D3 = dt.date(2025, 6, 2), dt.date(2025, 6, 3), dt.date(2025, 6, 4)


def snap(day: dt.date, spot: float) -> MarketSnapshot:
    return MarketSnapshot(
        day,
        quotes={"FX/EURUSD": spot, "EQ/SPX": 5000.0 + 10 * spot},
        curves={"USD": DiscountCurve.from_zero_rates([1.0, 5.0], [0.041, 0.038]),
                "EUR": DiscountCurve.from_zero_rates([1.0, 5.0], [0.02, 0.023])},
        surfaces={"EQ/SPX": VolSurface.flat(0.2)},
    )


def fill(root: Path) -> tuple[TradeStore, SnapshotStore]:
    ts, ss = TradeStore(root), SnapshotStore(root)
    ts.book(Trade("F1", FXForward("EURUSD", 1e6, 1.12, dt.date(2026, 6, 4))), D1, "2025-06-02T17:00:00")
    ts.book(Trade("B1", FixedCashflows("USD", [(dt.date(2027, 6, 2), 1e6)])), D1, "2025-06-02T17:05:00")
    ts.book(Trade("O1", EquityOption("SPX", "USD", 5100.0, dt.date(2026, 6, 2)), quantity=-3.0), D2,
            "2025-06-03T12:00:00")
    ts.cancel("B1", D3, "2025-06-04T08:00:00")
    for d, s in ((D1, 1.10), (D2, 1.105), (D3, 1.098)):
        ss.put(snap(d, s))
    return ts, ss


def test_book_as_of_each_day(tmp_path: Path) -> None:
    ts, _ = fill(tmp_path)
    assert [t.trade_id for t in ts.portfolio(D1)] == ["B1", "F1"]
    assert [t.trade_id for t in ts.portfolio(D2)] == ["B1", "F1", "O1"]
    assert [t.trade_id for t in ts.portfolio(D3)] == ["F1", "O1"]


def test_restating_a_past_date_reproduces_stored_pv_exactly(tmp_path: Path) -> None:
    ts, ss = fill(tmp_path)
    pvs = PVStore(tmp_path)
    for d in (D1, D2, D3):
        pvs.put(restate(ts, ss, d, "EUR"))
    # A back-dated amendment recorded later must not change what was known then.
    ts.amend(Trade("F1", FXForward("EURUSD", 2e6, 1.12, dt.date(2026, 6, 4))), D1, "2025-06-05T09:00:00")
    # Reopen everything from disk: a fresh process.
    ts2, ss2, pv2 = TradeStore(tmp_path), SnapshotStore(tmp_path), PVStore(tmp_path)
    for d in (D1, D2, D3):
        stored = pv2.get(d, "EUR")
        again = restate(ts2, ss2, d, "EUR")
        assert again.by_trade == stored.by_trade
        assert again.total == stored.total
    corrected = restate(ts2, ss2, D1, "EUR", known_at="2025-06-06T00:00:00")
    assert corrected.by_trade["F1"] == 2 * pv2.get(D1, "EUR").by_trade["F1"]
    assert len(pv2.to_frame()) == 2 + 3 + 2


def test_store_rules(tmp_path: Path) -> None:
    ts, ss = fill(tmp_path)
    with pytest.raises(InputError):
        ts.book(Trade("F1", Cash("USD", 1.0)), D2)
    with pytest.raises(InputError):
        ts.amend(Trade("ZZ", Cash("USD", 1.0)), D2)
    with pytest.raises(DataError):
        ss.put(snap(D1, 1.2))
    ss.put(snap(D1, 1.2), label="stress")
    assert ss.get(D1, "stress").quote("FX/EURUSD") == 1.2
    assert ss.get(D1) == snap(D1, 1.10)
    assert ss.dates() == [D1, D2, D3]
    with pytest.raises(DataError):
        ss.get(dt.date(2020, 1, 1))
    # Rebooking after a cancel is allowed.
    ts.book(Trade("B1", Cash("USD", 5.0)), D3, "2025-06-04T10:00:00")
    assert "B1" in [t.trade_id for t in ts.portfolio(D3)]
