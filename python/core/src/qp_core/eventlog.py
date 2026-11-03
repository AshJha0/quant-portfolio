"""Append-only, hash-chained event log with write-ahead ordering (W28a).

Plain idea
----------
An order system must be able to say, after a crash or in front of an
auditor, exactly what it did and in what order. The log gives three
guarantees:

1. **Write-ahead.** A caller appends the record *before* it acts on it
   (sends the order, applies the fill). ``append`` returns only after the
   record is written and, under :attr:`FsyncPolicy.EVERY_RECORD`, after
   ``os.fsync`` has returned. If the process dies after ``append`` and
   before the send, recovery sees an intent that was never acknowledged
   and can resolve it; the reverse (sent but not logged) cannot happen.
2. **Tamper evidence.** Each record carries ``prev_hash`` and ``hash``,
   where ``hash = SHA-256(canonical JSON of the record without "hash")``.
   Changing, removing or reordering any record breaks every later hash,
   and :meth:`EventLog.verify` reports the first bad sequence number.
3. **Torn-write recovery.** A crash during a write can leave a partial
   last line. On open, an unterminated final line (no ``"\\n"``) is
   truncated away and the byte count kept in ``truncated_bytes``; any
   complete line that fails verification is corruption or tampering, not
   a torn write, and raises :class:`EventLogCorrupt`.

Formal statement
----------------
Record ``i`` (``i = 1, 2, ...``) is the JSON object
``{seq: i, ts_ns, clock, err_ns, kind, payload, prev_hash, hash}`` with
``prev_hash_1 = "0" * 64`` and ``prev_hash_i = hash_{i-1}``. Canonical JSON
is ``json.dumps(obj, sort_keys=True, separators=(",", ":"))`` over UTF-8.
The chain is valid iff for every ``i`` the stored ``hash`` equals the
recomputed one and ``prev_hash_i = hash_{i-1}``.

Fsync policy (stated, per the W28 plan): ``EVERY_RECORD`` (default) calls
``flush`` then ``os.fsync`` on each append, so an acknowledged append
survives power loss to the extent the OS and disk honour fsync.
``NONE`` only flushes to the OS; it survives a process kill but not power
loss, and is meant for tests and replay tools. Group commit is not offered:
it would weaken guarantee 1 for the order path.

Each record also stores the clock source name and its stated error bound
in nanoseconds (``clock``, ``err_ns``) so later analysis knows how far a
timestamp can be trusted (W28d builds the clock discipline on this field).

Worked example
--------------
>>> log = EventLog.in_memory()
>>> r1 = log.append("order.intent", {"clordid": "A1", "qty": 100}, ts_ns=1)
>>> r2 = log.append("order.sent", {"clordid": "A1"}, ts_ns=2)
>>> r2.prev_hash == r1.hash, log.verify()
(True, 2)

References: SEC Rule 17a-4(f) (non-rewriteable, non-erasable records);
FIA "Best Practices for Automated Trading Risk Controls" (2024), audit
trail; Haber and Stornetta, "How to time-stamp a digital document",
J. Cryptology 3 (1991) for hash chaining.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from qp_core.errors import DataError

__all__ = [
    "GENESIS_HASH",
    "EventLog",
    "EventLogCorrupt",
    "FsyncPolicy",
    "LogRecord",
]

GENESIS_HASH = "0" * 64


class EventLogCorrupt(DataError):
    """The log fails verification somewhere other than a torn last line."""


class FsyncPolicy(enum.Enum):
    """When ``append`` forces the record to stable storage."""

    EVERY_RECORD = "every_record"
    NONE = "none"


def _canonical(obj: Mapping[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
        "utf-8"
    )


@dataclass(frozen=True)
class LogRecord:
    """One immutable log record."""

    seq: int
    ts_ns: int
    clock: str
    err_ns: int
    kind: str
    payload: Mapping[str, Any]
    prev_hash: str
    hash: str

    def body(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts_ns": self.ts_ns,
            "clock": self.clock,
            "err_ns": self.err_ns,
            "kind": self.kind,
            "payload": dict(self.payload),
            "prev_hash": self.prev_hash,
        }

    def expected_hash(self) -> str:
        return hashlib.sha256(_canonical(self.body())).hexdigest()

    def to_line(self) -> str:
        d = self.body()
        d["hash"] = self.hash
        return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"

    @classmethod
    def from_line(cls, line: str) -> LogRecord:
        d = json.loads(line)
        if not isinstance(d, dict):
            raise ValueError("record is not an object")
        return cls(
            seq=int(d["seq"]),
            ts_ns=int(d["ts_ns"]),
            clock=str(d["clock"]),
            err_ns=int(d["err_ns"]),
            kind=str(d["kind"]),
            payload=dict(d["payload"]),
            prev_hash=str(d["prev_hash"]),
            hash=str(d["hash"]),
        )


class EventLog:
    """Hash-chained append-only log, on disk (JSON lines) or in memory."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None,
        *,
        fsync: FsyncPolicy = FsyncPolicy.EVERY_RECORD,
        clock: str = "time.time_ns",
        err_ns: int = 1_000_000,
    ) -> None:
        self._path = Path(path) if path is not None else None
        self._fsync = fsync
        self._clock = clock
        self._err_ns = int(err_ns)
        self._records: list[LogRecord] = []
        self._fh: IO[str] | None = None
        self.truncated_bytes = 0
        if self._path is not None:
            self._load()
            self._fh = open(self._path, "a", encoding="utf-8", newline="\n")

    @classmethod
    def in_memory(cls, *, clock: str = "test", err_ns: int = 0) -> EventLog:
        return cls(None, fsync=FsyncPolicy.NONE, clock=clock, err_ns=err_ns)

    # -- loading --------------------------------------------------------
    def _load(self) -> None:
        assert self._path is not None
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            return
        raw = self._path.read_bytes()
        lines = raw.split(b"\n")
        # A file written only by append() ends in "\n", so the final split
        # element is b"" unless the last write was torn.
        offset = 0
        prev = GENESIS_HASH
        for idx, line in enumerate(lines):
            is_last = idx == len(lines) - 1
            if is_last and line == b"":
                break
            try:
                if is_last:
                    raise ValueError("unterminated last line")
                rec = LogRecord.from_line(line.decode("utf-8"))
                self._check(rec, prev, len(self._records) + 1)
            except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
                # Only an unterminated final line can come from a torn write;
                # a complete but invalid line is corruption or tampering.
                if not is_last:
                    raise EventLogCorrupt(f"{self._path}: bad record at line {idx + 1}: {exc}") from exc
                self.truncated_bytes = len(raw) - offset
                with open(self._path, "r+b") as fh:
                    fh.truncate(offset)
                    fh.flush()
                    os.fsync(fh.fileno())
                return
            self._records.append(rec)
            prev = rec.hash
            offset += len(line) + 1

    @staticmethod
    def _check(rec: LogRecord, prev: str, seq: int) -> None:
        if rec.seq != seq:
            raise ValueError(f"seq {rec.seq} where {seq} expected")
        if rec.prev_hash != prev:
            raise ValueError(f"prev_hash mismatch at seq {rec.seq}")
        if rec.expected_hash() != rec.hash:
            raise ValueError(f"hash mismatch at seq {rec.seq}")

    # -- writing --------------------------------------------------------
    def append(self, kind: str, payload: Mapping[str, Any], *, ts_ns: int | None = None) -> LogRecord:
        """Write one record durably (per the fsync policy) and return it.

        Call this *before* acting on the event (write-ahead ordering).
        """
        if not kind:
            raise ValueError("kind must be non-empty")
        seq = len(self._records) + 1
        prev = self._records[-1].hash if self._records else GENESIS_HASH
        stamp = time.time_ns() if ts_ns is None else int(ts_ns)
        # Round-trip the payload through JSON so the stored and hashed
        # forms are identical (tuples become lists, keys become strings).
        body_payload = json.loads(_canonical(dict(payload)))
        tmp = LogRecord(seq, stamp, self._clock, self._err_ns, kind, body_payload, prev, "")
        rec = LogRecord(seq, stamp, self._clock, self._err_ns, kind, body_payload, prev, tmp.expected_hash())
        if self._fh is not None:
            self._fh.write(rec.to_line())
            self._fh.flush()
            if self._fsync is FsyncPolicy.EVERY_RECORD:
                os.fsync(self._fh.fileno())
        self._records.append(rec)
        return rec

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reading --------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[LogRecord]:
        return iter(tuple(self._records))

    @property
    def head(self) -> str:
        """Hash of the last record (the chain head), or the genesis hash."""
        return self._records[-1].hash if self._records else GENESIS_HASH

    def verify(self) -> int:
        """Re-check the whole chain; return the record count or raise."""
        prev = GENESIS_HASH
        for i, rec in enumerate(self._records, start=1):
            try:
                self._check(rec, prev, i)
            except ValueError as exc:
                raise EventLogCorrupt(str(exc)) from exc
            prev = rec.hash
        return len(self._records)

    @staticmethod
    def verify_file(path: str | os.PathLike[str]) -> int:
        """Verify a log file without repairing it (a torn tail is an error)."""
        prev = GENESIS_HASH
        n = 0
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, start=1):
                if not line.endswith("\n"):
                    raise EventLogCorrupt(f"line {n}: unterminated")
                try:
                    rec = LogRecord.from_line(line)
                    EventLog._check(rec, prev, n)
                except (ValueError, KeyError, TypeError) as exc:
                    raise EventLogCorrupt(f"line {n}: {exc}") from exc
                prev = rec.hash
        return n
