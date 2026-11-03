"""Write-ahead journal, clock and restart recovery (W28d).

Plain idea: write down what you are about to do, make sure it is on disk,
then do it. After a crash, read the journal back, ask the venue what it
actually saw, and fill the gaps from the venue's answer. Because nothing is
sent before it is journaled, the venue can never know an order we do not.

Formal statement.

- Record i is the canonical JSON of ``{"seq": i, "prev": h_{i-1}, "rec": r_i}``
  followed by ``"h": h_i`` where ``h_i = sha256(h_{i-1} || canon(r_i))`` and
  ``h_0`` is 64 zeros. One record per line.
- **Fsync policy:** ``flush`` + ``os.fsync`` after every record, before the
  caller is allowed to send. This is the strict policy; it costs one fsync per
  message and is the only policy offered here. (A group-commit policy would
  batch fsyncs and hold sends until the batch is durable; not implemented.)
- **Torn tail:** a crash can leave a partial last line. Recovery keeps every
  record up to the last one that parses and whose hash verifies, and
  truncates the rest. A bad hash *before* the tail is tampering, not a crash,
  and raises.

Seam: the r12-l2-oms lane ships ``qp_core.eventlog`` with the same
write-ahead contract; at integration this class becomes a thin adapter over
it. Sources: SEC Rule 17a-4(f) (non-rewriteable records; the hash chain
makes edits detectable, it does not by itself make storage WORM).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ZERO = "0" * 64
NL = b"\n"


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class JournalCorrupt(Exception):
    """A record before the tail fails its hash: tampering or a disk fault."""


@dataclass(frozen=True)
class ClockReading:
    """A timestamp with its source and a stated error bound (W28d clock)."""

    ns: int
    source: str
    err_ns: int


class WallClock:
    """System clock, forced monotonic, with a configured error bound.

    The bound is an input, not a measurement: on a host synced by NTP over the
    internet a few milliseconds is typical (RFC 5905 gives no hard bound);
    MiFID II RTS 25 asks 1 ms for most algorithmic trading and 100 us for HFT,
    which needs PTP. The default 5 ms says honestly that this laptop meets
    neither HFT bound.
    """

    def __init__(self, err_ns: int = 5_000_000, source: str = "system/ntp") -> None:
        self.err_ns = err_ns
        self.source = source
        self._last = 0

    def now(self) -> ClockReading:
        t = max(time.time_ns(), self._last + 1)
        self._last = t
        return ClockReading(t, self.source, self.err_ns)


class SimClock:
    """Deterministic clock for tests and replays."""

    def __init__(self, start_ns: int = 0, step_ns: int = 1_000) -> None:
        self._t = start_ns
        self.step = step_ns

    def now(self) -> ClockReading:
        self._t += self.step
        return ClockReading(self._t, "sim", 0)


class Journal:
    def __init__(self, path: Path, fsync: bool = True) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self.records, self.head = read_journal(self.path, repair=True)
        self._f = open(self.path, "ab")

    def append(self, rec: dict[str, Any], clock: ClockReading | None = None) -> int:
        """Write and make durable; return seq. Only after this returns may the caller send."""
        if clock is not None:
            rec = {**rec, "ts_ns": clock.ns, "ts_src": clock.source, "ts_err_ns": clock.err_ns}
        seq = len(self.records) + 1
        h = hashlib.sha256((self.head + _canon(rec)).encode()).hexdigest()
        line = _canon({"seq": seq, "prev": self.head, "rec": rec, "h": h}) + "\n"
        self._f.write(line.encode())
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())
        self.records.append(rec)
        self.head = h
        return seq

    def close(self) -> None:
        self._f.close()


def read_journal(path: Path, repair: bool = False) -> tuple[list[dict[str, Any]], str]:
    """Return (records, head hash). Truncates a torn tail when ``repair``."""
    if not path.exists():
        return [], ZERO
    data = path.read_bytes()
    end = data.rfind(NL) + 1  # bytes after the last newline are a torn write
    recs: list[dict[str, Any]] = []
    head = ZERO
    pos = 0
    for raw in data[:end].split(NL)[:-1]:
        try:
            obj = json.loads(raw)
            ok = (
                obj["seq"] == len(recs) + 1
                and obj["prev"] == head
                and obj["h"] == hashlib.sha256((head + _canon(obj["rec"])).encode()).hexdigest()
            )
        except (ValueError, KeyError, TypeError):
            ok = False
        if not ok:
            raise JournalCorrupt(f"{path}: record at byte {pos} fails verification")
        recs.append(obj["rec"])
        head = obj["h"]
        pos += len(raw) + 1
    if repair and end < len(data):
        with open(path, "r+b") as f:
            f.truncate(end)
    return recs, head
