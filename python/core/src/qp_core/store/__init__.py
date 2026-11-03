"""Trade, snapshot and PV stores, so a past day can be restated exactly.

Plain idea
----------
Nothing is overwritten. A trade store is an append-only log of events
(book, amend, cancel), each stamped with the business date it applies from
and the moment it was recorded. Snapshots are kept one per date. A PV run
stores the numbers it produced together with the knowledge time it used.
Restating a past date replays the trade events known at that time against
that date's snapshot, so the stored PV comes back to the last bit.

Formal statement
----------------
Event ``e = (id, kind, valid_from v_e, recorded k_e, trade)``. The book on
business date ``d`` as known at time ``a`` is

    B(d, a) = { last_e trade : v_e <= d, k_e <= a, kind != cancel },

grouped by trade id with "last" ordered by ``(k_e, sequence)``. This is the
bitemporal model of ``qp_data.store`` (Snodgrass 1999) applied to trades,
and event sourcing in Fowler's sense: the log is the source of truth and a
book is a fold over it. ``restate(d, a) = sum_t pv_t(S_d)`` with ``S_d`` the
stored snapshot of ``d``.

Storage is JSON (one object per line for events, one file per snapshot).
JSON floats are ``repr`` strings that round-trip exactly, which the exact
restatement relies on. Parquet or DuckDB need ``pyarrow``/``duckdb``, which
``qp_core`` does not require; :meth:`PVStore.to_frame` gives a pandas frame
for anyone who wants to write one.

Worked example
--------------
>>> import datetime as dt, tempfile
>>> from qp_core.instruments import Cash, Trade
>>> from qp_core.market import DiscountCurve, MarketSnapshot
>>> root = tempfile.mkdtemp()
>>> ts = TradeStore(root)
>>> ts.book(Trade("T1", Cash("USD", 100.0, 1.0)), valid_from=dt.date(2025, 6, 2), recorded="2025-06-02T18:00:00")
>>> ts.amend(Trade("T1", Cash("USD", 150.0, 1.0)), valid_from=dt.date(2025, 6, 3), recorded="2025-06-03T09:00:00")
>>> [t.instrument.amount for t in ts.portfolio(dt.date(2025, 6, 2))]
[100.0]
>>> [t.instrument.amount for t in ts.portfolio(dt.date(2025, 6, 3))]
[150.0]
>>> ss = SnapshotStore(root)
>>> ss.put(MarketSnapshot(dt.date(2025, 6, 2), curves={"USD": DiscountCurve.flat(0.04)}))
>>> r = restate(ts, ss, dt.date(2025, 6, 2), "USD")
>>> round(r.total, 6)
96.078944
"""

from __future__ import annotations

from .stores import PVRecord, PVStore, SnapshotStore, TradeEvent, TradeStore, restate

__all__ = ["PVRecord", "PVStore", "SnapshotStore", "TradeEvent", "TradeStore", "restate"]
