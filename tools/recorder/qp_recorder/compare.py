"""Compare a handler's per-record dump with the Python reference book.

A handler dump line is ``index stale seq digest`` (see cpp/replay_dump.cpp
and rust/src/main.rs). The rule checked is the one the C++ fault-injection
tests use: **whenever the handler says it is not stale, its book equals the
reference book at the same key** (the update id for Binance, the record
index for Coinbase level2). A stale handler is not judged; it has said its
book must not be used.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .qprc import KIND_SNAPSHOT, read
from .refbook import binance_reference, coinbase_level2_reference


@dataclass
class Result:
    records: int
    fresh: int  # records at which the handler reported not stale
    checked: int  # fresh records whose key the reference knows
    mismatches: list[int]
    final_fresh: bool


def reference(venue: str, buf: bytes, symbol: str) -> dict[int, int]:
    recs, _ = read(buf)
    if venue == "binance":
        return binance_reference(recs, symbol)
    if venue == "coinbase-level2":
        return coinbase_level2_reference(recs, symbol)
    raise ValueError(venue)


def compare(venue: str, dump: str, ref: dict[int, int]) -> Result:
    rows = [ln.split() for ln in dump.splitlines() if ln.strip()]
    fresh = checked = 0
    bad: list[int] = []
    for idx, stale, seq, dig in rows:
        if stale != "0":
            continue
        fresh += 1
        key = int(seq) if venue == "binance" else int(idx)
        want = ref.get(key)
        if want is None:
            continue
        checked += 1
        if int(dig, 16) != want:
            bad.append(int(idx))
    return Result(len(rows), fresh, checked, bad, bool(rows) and rows[-1][1] == "0")


def load(path: Path) -> bytes:
    """Recording bytes; ``.gz`` files are stored compressed (JSON text
    compresses about 7-10x) and decompressed here."""
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == ".gz" else raw


def manifest_entry(path: Path, venue: str, symbol: str, channel: str, lines: int, note: str) -> dict:
    buf = load(path)
    recs, trunc = read(buf)
    t0 = recs[0].recv_ns if recs else 0
    t1 = recs[-1].recv_ns if recs else 0
    return {
        "file": path.name,
        "venue": venue,
        "symbol": symbol,
        "channel": channel,
        "lines": lines,
        "start_utc_ns": t0,
        "seconds": round((t1 - t0) / 1e9, 3),
        "records": len(recs),
        "snapshots": sum(r.kind == KIND_SNAPSHOT for r in recs),
        "bytes": len(buf),
        "sha256": hashlib.sha256(buf).hexdigest(),
        "stored_bytes": path.stat().st_size,
        "truncated": trunc,
        "stored_in_repo": True,
        "note": note,
    }


def write_manifest(path: Path, entries: list[dict]) -> None:
    path.write_text(
        json.dumps({"format": "QPRC v1", "recordings": entries}, indent=2) + "\n", encoding="utf-8"
    )
