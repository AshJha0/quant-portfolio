"""Rebuild manifest.json from the recordings in this folder.

Run from the repo root: python -m tools.recorder.recordings.make_manifest
The per-file notes below are the only hand-written part.
"""

from __future__ import annotations

from pathlib import Path

from tools.recorder.qp_recorder.compare import manifest_entry, write_manifest

HERE = Path(__file__).resolve().parent
NOTES = {
    "binance_btcusdt_depth_ab_20261103.qprc.gz": (
        "binance",
        "BTCUSDT",
        "btcusdt@depth@100ms + /api/v3/depth?limit=1000",
        2,
        "Live capture, about 90 s, two independent websocket connections (lines A and B) and one REST snapshot.",
    ),
    "binance_btcusdt_depth_ab_2snap_20261103.qprc.gz": (
        "binance",
        "BTCUSDT",
        "btcusdt@depth@100ms + /api/v3/depth?limit=1000",
        2,
        "Live capture, about 60 s. Taken with an early recorder that fetched a second snapshot by mistake; "
        "kept because the second snapshot (same lastUpdateId as the live book) separates the C++ and Rust "
        "handlers: C++ ignores a snapshot not newer than its book, Rust reloads it and loses the levels "
        "beyond the snapshot's 1,000-level depth.",
    ),
    "coinbase_btcusd_level2batch_20261103.qprc.gz": (
        "coinbase-level2",
        "BTC-USD",
        "level2_batch (served as level2_50)",
        1,
        "Live capture, about 45 s, one line; first frames are the subscriptions ack and the in-band snapshot.",
    ),
}


def main() -> None:
    entries = []
    for name, (venue, sym, ch, lines, note) in sorted(NOTES.items()):
        entries.append(manifest_entry(HERE / name, venue, sym, ch, lines, note))
    write_manifest(HERE / "manifest.json", entries)


if __name__ == "__main__":
    main()
