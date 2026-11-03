"""Tests for the hash-chained event log (W28a)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from qp_core.eventlog import GENESIS_HASH, EventLog, EventLogCorrupt, FsyncPolicy, LogRecord


def test_hash_is_sha256_of_canonical_body() -> None:
    log = EventLog.in_memory()
    rec = log.append("k", {"b": 2, "a": 1}, ts_ns=5)
    body = '{"clock":"test","err_ns":0,"kind":"k","payload":{"a":1,"b":2},"prev_hash":"' + GENESIS_HASH
    body += '","seq":1,"ts_ns":5}'
    assert rec.hash == hashlib.sha256(body.encode()).hexdigest()
    assert rec.prev_hash == GENESIS_HASH


def test_chain_links_and_head() -> None:
    log = EventLog.in_memory()
    recs = [log.append("e", {"i": i}, ts_ns=i) for i in range(5)]
    for a, b in zip(recs, recs[1:], strict=False):
        assert b.prev_hash == a.hash
    assert log.head == recs[-1].hash
    assert log.verify() == 5


def test_round_trip_and_reopen_continues_chain(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    with EventLog(p, fsync=FsyncPolicy.EVERY_RECORD) as log:
        log.append("a", {"x": [1, 2]}, ts_ns=1)
        log.append("b", {"y": "z"}, ts_ns=2)
        head = log.head
    with EventLog(p) as log2:
        assert len(log2) == 2 and log2.head == head
        r3 = log2.append("c", {}, ts_ns=3)
        assert r3.seq == 3 and r3.prev_hash == head
    assert EventLog.verify_file(p) == 3


def test_tampered_payload_is_detected(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    with EventLog(p, fsync=FsyncPolicy.NONE) as log:
        for i in range(3):
            log.append("fill", {"qty": 100 + i}, ts_ns=i)
    lines = p.read_text().splitlines(keepends=True)
    lines[1] = lines[1].replace('"qty":101', '"qty":999')
    p.write_text("".join(lines))
    with pytest.raises(EventLogCorrupt, match="line 2"):
        EventLog(p)
    with pytest.raises(EventLogCorrupt):
        EventLog.verify_file(p)


def test_deleted_record_is_detected(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    with EventLog(p, fsync=FsyncPolicy.NONE) as log:
        for i in range(3):
            log.append("e", {"i": i}, ts_ns=i)
    lines = p.read_text().splitlines(keepends=True)
    p.write_text(lines[0] + lines[2])
    with pytest.raises(EventLogCorrupt):
        EventLog(p)


def test_tampered_last_complete_line_is_corruption_not_torn(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    with EventLog(p, fsync=FsyncPolicy.NONE) as log:
        log.append("e", {"i": 0}, ts_ns=0)
        log.append("e", {"i": 1}, ts_ns=1)
    p.write_text(p.read_text().replace('"i":1', '"i":7'))
    with pytest.raises(EventLogCorrupt):
        EventLog(p)


@pytest.mark.parametrize("cut", [1, 10, 40])
def test_torn_tail_is_truncated(tmp_path: Path, cut: int) -> None:
    p = tmp_path / "log.jsonl"
    with EventLog(p, fsync=FsyncPolicy.NONE) as log:
        log.append("e", {"i": 0}, ts_ns=0)
        good = p.stat().st_size
        log.append("e", {"i": 1}, ts_ns=1)
    raw = p.read_bytes()
    p.write_bytes(raw[: good + cut])  # partial second record, no newline
    with EventLog(p) as log2:
        assert len(log2) == 1
        assert log2.truncated_bytes == cut
        log2.append("e", {"i": 2}, ts_ns=2)
    assert EventLog.verify_file(p) == 2


def test_in_memory_verify_detects_mutation() -> None:
    log = EventLog.in_memory()
    log.append("e", {"i": 0}, ts_ns=0)
    log.append("e", {"i": 1}, ts_ns=1)
    bad = LogRecord(2, 1, "test", 0, "e", {"i": 9}, log.head, log.head)
    log._records[1] = bad  # simulate in-memory corruption
    with pytest.raises(EventLogCorrupt):
        log.verify()


def test_payload_is_normalised_through_json() -> None:
    log = EventLog.in_memory()
    rec = log.append("e", {"t": (1, 2), "n": None}, ts_ns=0)
    assert rec.payload == {"t": [1, 2], "n": None}
    assert json.loads(rec.to_line())["hash"] == rec.hash


def test_empty_kind_rejected() -> None:
    with pytest.raises(ValueError):
        EventLog.in_memory().append("", {})


@given(
    st.lists(
        st.dictionaries(st.text(max_size=5), st.integers() | st.text(max_size=5), max_size=4), max_size=20
    )
)
def test_property_reload_equals_written(payloads: list[dict[str, object]]) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "l.jsonl"
        with EventLog(p, fsync=FsyncPolicy.NONE) as log:
            for i, pl in enumerate(payloads):
                log.append("e", pl, ts_ns=i)
            written = list(log)
        with EventLog(p) as again:
            assert list(again) == written
            assert again.verify() == len(payloads)


def test_nan_payload_rejected() -> None:
    with pytest.raises(ValueError):
        EventLog.in_memory().append("e", {"px": float("nan")})
