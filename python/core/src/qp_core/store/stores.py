"""Append-only trade log, snapshot store and PV store; see the package docstring."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..errors import DataError, InputError
from ..instruments.serde import from_json, to_json
from ..instruments.trade import Portfolio, Trade
from ..market.snapshot import MarketSnapshot

__all__ = ["PVRecord", "PVStore", "SnapshotStore", "TradeEvent", "TradeStore", "restate"]

_KINDS = ("book", "amend", "cancel")


def _ts(x: str | dt.datetime | dt.date | None) -> pd.Timestamp:
    """UTC timestamp; a bare date means the end of that day (everything known by then)."""
    if x is None:
        t = pd.Timestamp.now(tz="UTC")
    elif isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        t = pd.Timestamp(dt.datetime.combine(x, dt.time.max))
    else:
        t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class TradeEvent:
    """One line of the trade log."""

    seq: int
    kind: str
    trade_id: str
    valid_from: dt.date
    recorded: pd.Timestamp
    trade: Trade | None

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "kind": self.kind, "trade_id": self.trade_id,
                "valid_from": self.valid_from.isoformat(), "recorded": self.recorded.isoformat(),
                "trade": to_json(self.trade) if self.trade is not None else None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TradeEvent:
        return cls(d["seq"], d["kind"], d["trade_id"], dt.date.fromisoformat(d["valid_from"]),
                   _ts(d["recorded"]), from_json(d["trade"]) if d["trade"] is not None else None)


class TradeStore:
    """Bitemporal, append-only trade log at ``root/trades.jsonl``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "trades.jsonl"

    def events(self) -> list[TradeEvent]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [TradeEvent.from_dict(json.loads(line)) for line in fh if line.strip()]

    def _append(self, kind: str, trade_id: str, trade: Trade | None, valid_from: dt.date,
                recorded: str | dt.datetime | None) -> None:
        if kind not in _KINDS:
            raise InputError(f"unknown event kind {kind!r}")
        ev = self.events()
        live = self._live(ev, valid_from, None)
        if kind == "book" and trade_id in live:
            raise InputError(f"trade {trade_id} already booked; amend it")
        if kind in ("amend", "cancel") and trade_id not in {e.trade_id for e in ev}:
            raise InputError(f"trade {trade_id} was never booked")
        e = TradeEvent(len(ev), kind, trade_id, valid_from, _ts(recorded), trade)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(_dumps(e.to_dict()) + "\n")

    def book(self, trade: Trade, valid_from: dt.date, recorded: str | dt.datetime | None = None) -> None:
        self._append("book", trade.trade_id, trade, valid_from, recorded)

    def amend(self, trade: Trade, valid_from: dt.date, recorded: str | dt.datetime | None = None) -> None:
        self._append("amend", trade.trade_id, trade, valid_from, recorded)

    def cancel(self, trade_id: str, valid_from: dt.date, recorded: str | dt.datetime | None = None) -> None:
        self._append("cancel", trade_id, None, valid_from, recorded)

    @staticmethod
    def _live(events: list[TradeEvent], day: dt.date, known_at: pd.Timestamp | None) -> dict[str, Trade]:
        rows = [e for e in events if e.valid_from <= day and (known_at is None or e.recorded <= known_at)]
        rows.sort(key=lambda e: (e.valid_from, e.recorded, e.seq))
        book: dict[str, Trade] = {}
        for e in rows:
            if e.kind == "cancel":
                book.pop(e.trade_id, None)
            elif e.trade is not None:
                book[e.trade_id] = e.trade
        return book

    def portfolio(self, day: dt.date, known_at: str | dt.datetime | dt.date | None = None,
                  name: str = "") -> Portfolio:
        """The book on business date ``day`` as known at ``known_at`` (default: now)."""
        live = self._live(self.events(), day, _ts(known_at) if known_at is not None else None)
        return Portfolio((live[k] for k in sorted(live)), name or f"book {day}")


class SnapshotStore:
    """One immutable snapshot per (date, label) at ``root/snapshots/<date>[__label].json``."""

    def __init__(self, root: str | Path) -> None:
        self.dir = Path(root) / "snapshots"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, day: dt.date, label: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
        return self.dir / (f"{day.isoformat()}__{safe}.json" if label else f"{day.isoformat()}.json")

    def put(self, snap: MarketSnapshot, label: str = "") -> None:
        """Store ``snap``; refuses to overwrite a different snapshot (store a new label)."""
        p = self._path(snap.as_of, label)
        text = snap.to_json()
        if p.exists() and p.read_text(encoding="utf-8") != text:
            raise DataError(f"snapshot {p.name} exists with different content; snapshots are immutable")
        p.write_text(text, encoding="utf-8")

    def get(self, day: dt.date, label: str = "") -> MarketSnapshot:
        p = self._path(day, label)
        if not p.exists():
            raise DataError(f"no snapshot for {day} {label!r}")
        return MarketSnapshot.from_json(p.read_text(encoding="utf-8"))

    def dates(self) -> list[dt.date]:
        return sorted({dt.date.fromisoformat(p.name[:10]) for p in self.dir.glob("*.json")})


@dataclass(frozen=True)
class PVRecord:
    """A stored PV run: per-trade PVs in ``ccy``, and the knowledge time used."""

    day: dt.date
    known_at: str
    ccy: str
    by_trade: dict[str, float]
    snapshot_label: str = ""

    @property
    def total(self) -> float:
        return sum(self.by_trade[k] for k in sorted(self.by_trade))

    def to_dict(self) -> dict[str, Any]:
        return {"day": self.day.isoformat(), "known_at": self.known_at, "ccy": self.ccy,
                "by_trade": dict(sorted(self.by_trade.items())), "snapshot_label": self.snapshot_label}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PVRecord:
        return cls(dt.date.fromisoformat(d["day"]), d["known_at"], d["ccy"], dict(d["by_trade"]),
                   d.get("snapshot_label", ""))


class PVStore:
    """Append-only PV log at ``root/pv.jsonl``."""

    def __init__(self, root: str | Path) -> None:
        self.path = Path(root) / "pv.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def put(self, rec: PVRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(_dumps(rec.to_dict()) + "\n")

    def __iter__(self) -> Iterator[PVRecord]:
        if not self.path.exists():
            return iter(())
        with self.path.open(encoding="utf-8") as fh:
            return iter([PVRecord.from_dict(json.loads(line)) for line in fh if line.strip()])

    def get(self, day: dt.date, ccy: str) -> PVRecord:
        """The latest stored run for ``day`` in ``ccy``."""
        hits = [r for r in self if r.day == day and r.ccy == ccy]
        if not hits:
            raise DataError(f"no stored PV for {day} in {ccy}")
        return hits[-1]

    def to_frame(self) -> pd.DataFrame:
        rows = [{"day": r.day, "known_at": r.known_at, "ccy": r.ccy, "trade_id": k, "pv": v}
                for r in self for k, v in sorted(r.by_trade.items())]
        return pd.DataFrame(rows, columns=["day", "known_at", "ccy", "trade_id", "pv"])


def restate(trades: TradeStore, snaps: SnapshotStore, day: dt.date, ccy: str,
            known_at: str | dt.datetime | dt.date | None = None, snapshot_label: str = "") -> PVRecord:
    """PV of the book on ``day`` as known at ``known_at`` against the stored snapshot of ``day``.

    ``known_at`` defaults to the end of ``day``: the book as it stood that
    evening, ignoring later back-dated amendments. Pass a later time to see
    a corrected history.
    """
    k = _ts(known_at if known_at is not None else day)
    book = trades.portfolio(day, k)
    snap = snaps.get(day, snapshot_label)
    return PVRecord(day, k.isoformat(), ccy, book.pv_by_trade(snap, ccy), snapshot_label)
