"""Offline tests for the QPRC recorder, the Python reference book and the
manifest. No network: the recorder is exercised against a local websocket
server on 127.0.0.1."""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import random
from argparse import Namespace
from pathlib import Path

import pytest

from tools.recorder.qp_recorder import qprc, record
from tools.recorder.qp_recorder.compare import compare, load
from tools.recorder.qp_recorder.refbook import (
    L2,
    binance_reference,
    coinbase_level2_reference,
    digest,
    fixed,
    level_hash,
)

REPO = Path(__file__).resolve().parents[3]
REC_DIR = REPO / "tools" / "recorder" / "recordings"
CPP_FIXTURES = REPO / "cpp" / "ull" / "feed" / "tests" / "fixtures"


# ---- format ---------------------------------------------------------------------


def test_worked_example_bytes_match_cpp_and_rust_layout() -> None:
    out = io.BytesIO()
    w = qprc.Writer(out)
    w.write(1, qprc.LINE_B, qprc.KIND_FRAME, b"{}")
    b = out.getvalue()
    assert b[:8] == b"QPRC\x01\x00\x00\x00"
    # Same record bytes as rust/ull/feed/src/recorder.rs worked_example_bytes.
    assert list(b[8:]) == [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, ord("{"), ord("}")]
    assert qprc.to_qprr1(b) == b"QPRR1\n" + b[8:]
    assert qprc.read(b) == ([qprc.Record(1, 1, 0, b"{}")], False)


def test_every_truncation_point_reads_whole_records_only() -> None:
    out = io.BytesIO()
    w = qprc.Writer(out)
    w.write(5, qprc.LINE_A, qprc.KIND_FRAME, b"abc")
    w.write(6, qprc.LINE_REST, qprc.KIND_SNAPSHOT, b"hello")
    b = out.getvalue()
    first_end = 8 + 14 + 3
    for cut in range(8, len(b)):
        recs, trunc = qprc.read(b[:cut])
        assert len(recs) == (1 if cut >= first_end else 0)
        assert trunc == (cut not in (8, first_end))
    assert qprc.read(b) == (
        [qprc.Record(5, 0, 0, b"abc"), qprc.Record(6, 255, 1, b"hello")],
        False,
    )
    with pytest.raises(ValueError):
        qprc.read(b"QPRR1\n")
    with pytest.raises(ValueError):
        w.write(1, 2, 0, b"")


@pytest.mark.parametrize("name", ["coinbase_full_11", "binance_depth_21"])
def test_reads_the_cpp_fixtures(name: str) -> None:
    recs, trunc = qprc.read((CPP_FIXTURES / f"{name}.qprc").read_bytes())
    assert not trunc and len(recs) > 500


# ---- reference book -------------------------------------------------------------


def test_fixed_point_parse_refuses_to_round() -> None:
    assert fixed("0.1") + fixed("0.2") == fixed("0.3")
    assert fixed("77582.00000000") == 7_758_200_000_000
    with pytest.raises(ValueError):
        fixed("0.000000001")
    with pytest.raises(ValueError):
        fixed("abc")


def test_incremental_digest_equals_recomputed_digest() -> None:
    rng = random.Random(7)
    b = L2()
    for _ in range(5_000):
        side = rng.choice("BA")
        b.set(side, rng.randint(1, 50) * 10**6, rng.choice([0, rng.randint(1, 10**9)]))
    full = sum(level_hash("B", p, q) for p, q in b.bids.items())
    full += sum(level_hash("A", p, q) for p, q in b.asks.items())
    assert digest(b) == full % 2**64
    # Worked example from refbook's docstring.
    w = L2()
    w.set("B", fixed("100.10"), fixed("2"))
    w.set("B", fixed("100.1"), fixed("0.5"))
    assert w.dump() == "B 100.1 0.5\n"
    assert digest(w) == level_hash("B", 10_010_000_000, 50_000_000)


def _digest_of_expected(text: str) -> int:
    b = L2()
    for ln in text.splitlines():
        parts = ln.split()
        if parts[0] in ("B", "A"):
            b.set(parts[0], fixed(parts[1]), fixed(parts[2]))
    return digest(b)


@pytest.mark.parametrize("name", ["binance_depth_21", "binance_depth_22"])
def test_binance_reference_reproduces_the_cpp_fixture_final_books(name: str) -> None:
    # An independent check of the reference: the fixtures' expected books come
    # from cpp/ull/feed/tools/make_fixtures.py, which shares no code with it.
    recs, _ = qprc.read((CPP_FIXTURES / f"{name}.qprc").read_bytes())
    exp = (CPP_FIXTURES / f"{name}.expected").read_text()
    last_seq = int(exp.splitlines()[0].split()[1])
    ref = binance_reference(recs, "BTCUSDT")
    assert max(ref) == last_seq
    assert ref[last_seq] == _digest_of_expected(exp)


def test_binance_reference_orders_by_id_and_stops_at_a_hole() -> None:
    def ev(u0: int, u1: int, px: str) -> bytes:
        return json.dumps(
            {"e": "depthUpdate", "s": "X", "U": u0, "u": u1, "b": [[px, "1"]], "a": []}
        ).encode()

    snap = json.dumps({"lastUpdateId": 10, "bids": [["1", "1"]], "asks": []}).encode()
    recs = [
        qprc.Record(1, 0, 0, ev(8, 11, "2")),  # straddles the snapshot: kept
        qprc.Record(2, 255, 1, snap),
        qprc.Record(3, 1, 0, ev(14, 15, "4")),  # arrives early on line B
        qprc.Record(4, 0, 0, ev(12, 13, "3")),
        qprc.Record(5, 1, 0, ev(12, 13, "3")),  # A/B duplicate
        qprc.Record(6, 0, 0, ev(17, 18, "5")),  # 16 never arrives
    ]
    ref = binance_reference(recs, "X")
    assert sorted(ref) == [10, 11, 13, 15]


def test_coinbase_level2_reference_is_live_only_after_snapshot() -> None:
    upd = json.dumps({"type": "l2update", "product_id": "P", "changes": [["sell", "2", "1"]]}).encode()
    snap = json.dumps({"type": "snapshot", "product_id": "P", "bids": [["1", "1"]], "asks": []}).encode()
    other = json.dumps({"type": "l2update", "product_id": "Q", "changes": [["sell", "9", "1"]]}).encode()
    ref = coinbase_level2_reference(
        [
            qprc.Record(0, 0, 0, upd),
            qprc.Record(1, 0, 0, snap),
            qprc.Record(2, 0, 0, other),
            qprc.Record(3, 0, 0, upd),
        ],
        "P",
    )
    assert sorted(ref) == [1, 2, 3] and ref[1] == ref[2] != ref[3]


def test_compare_judges_only_fresh_rows() -> None:
    ref = {5: 0xAB, 6: 0xCD}
    dump = "0 1 0 0000000000000000\n1 0 5 00000000000000ab\n2 0 6 00000000000000ff\n3 0 7 0000000000000001\n"
    r = compare("binance", dump, ref)
    assert (r.records, r.fresh, r.checked, r.mismatches, r.final_fresh) == (4, 3, 2, [2], True)


# ---- manifest -------------------------------------------------------------------


def test_manifest_lists_every_recording_with_its_hash() -> None:
    man = json.loads((REC_DIR / "manifest.json").read_text())
    files = {p.name for p in REC_DIR.glob("*.qprc*")}
    assert {e["file"] for e in man["recordings"]} == files
    for e in man["recordings"]:
        buf = load(REC_DIR / e["file"])
        assert hashlib.sha256(buf).hexdigest() == e["sha256"]
        recs, trunc = qprc.read(buf)
        assert len(recs) == e["records"] and not trunc


# ---- recorder against a local server --------------------------------------------


def test_coinbase_recorder_writes_frames_as_received(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    websockets = pytest.importorskip("websockets")
    frames = [
        '{"type":"subscriptions","channels":[]}',
        '{"type":"snapshot","product_id":"BTC-USD","bids":[["1.5","2"]],"asks":[["2.5","1"]]}',
        '{"type":"l2update","product_id":"BTC-USD","changes":[["buy","1.5","0"]],"time":"t"}',
    ]
    got: list[str] = []

    async def handler(ws) -> None:  # type: ignore[no-untyped-def]
        got.append(await ws.recv())
        for f in frames:
            await ws.send(f)
        await asyncio.sleep(5)

    async def go() -> list[Path]:
        async with websockets.serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            monkeypatch.setattr(record, "COINBASE_WS", f"ws://127.0.0.1:{port}")
            a = Namespace(
                product="BTC-USD",
                channel="level2_batch",
                seconds=1.0,
                out=str(tmp_path / "c.qprc"),
                rotate_minutes=0,
            )
            return await record.record_coinbase(a)

    paths = asyncio.run(go())
    assert json.loads(got[0]) == {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channels": ["level2_batch"],
    }
    recs, trunc = qprc.read(paths[0].read_bytes())
    assert not trunc
    assert [r.data.decode() for r in recs] == frames
    assert {(r.line, r.kind) for r in recs} == {(0, 0)}
    ref = coinbase_level2_reference(recs, "BTC-USD")
    assert sorted(ref) == [1, 2]


def test_coinbase_recorder_stops_when_the_venue_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    websockets = pytest.importorskip("websockets")

    async def handler(ws) -> None:  # type: ignore[no-untyped-def]
        await ws.recv()
        await ws.send(
            '{"type":"error","message":"Failed to subscribe","reason":"full channels now require authentication"}'
        )
        await asyncio.sleep(5)

    async def go() -> None:
        async with websockets.serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            monkeypatch.setattr(record, "COINBASE_WS", f"ws://127.0.0.1:{port}")
            a = Namespace(
                product="BTC-USD", channel="full", seconds=5.0, out=str(tmp_path / "c.qprc"), rotate_minutes=0
            )
            await record.record_coinbase(a)

    with pytest.raises(SystemExit, match="require authentication"):
        asyncio.run(go())


def test_binance_recorder_two_lines_one_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    websockets = pytest.importorskip("websockets")
    events = [
        json.dumps(
            {"e": "depthUpdate", "E": 1, "s": "BTCUSDT", "U": u, "u": u, "b": [["1", str(u)]], "a": []}
        )
        for u in range(100, 110)
    ]
    snap = json.dumps({"lastUpdateId": 103, "bids": [["1", "103"]], "asks": []}).encode()

    async def handler(ws) -> None:  # type: ignore[no-untyped-def]
        for e in events:
            await ws.send(e)
            await asyncio.sleep(0.02)
        await asyncio.sleep(5)

    async def go() -> list[Path]:
        async with websockets.serve(handler, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            monkeypatch.setattr(record, "BINANCE_WS", f"ws://127.0.0.1:{port}/{{sym}}")
            monkeypatch.setattr(record, "_get", lambda url: snap)
            a = Namespace(
                symbol="BTCUSDT",
                lines=2,
                depth_limit=1000,
                snapshot_delay=0.05,
                seconds=1.0,
                out=str(tmp_path / "b.qprc"),
                rotate_minutes=0,
            )
            return await record.record_binance(a)

    paths = asyncio.run(go())
    recs, _ = qprc.read(paths[0].read_bytes())
    assert sum(r.kind == qprc.KIND_SNAPSHOT for r in recs) == 1
    assert sum(r.line == 0 for r in recs) == sum(r.line == 1 for r in recs) == len(events)
    ref = binance_reference(recs, "BTCUSDT")
    assert sorted(ref) == list(range(103, 110))


def test_gzip_load_roundtrip(tmp_path: Path) -> None:
    out = io.BytesIO()
    qprc.Writer(out).write(1, 0, 0, b"x")
    p = tmp_path / "a.qprc.gz"
    p.write_bytes(gzip.compress(out.getvalue()))
    assert load(p) == out.getvalue()
