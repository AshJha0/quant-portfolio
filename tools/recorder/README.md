# tools/recorder: live market-data recorder and replay checks

This folder records public crypto order-book feeds into the raw-byte QPRC
format that the C++ (`cpp/ull/feed`) and Rust (`rust/ull/feed`) feed
handlers replay, and checks both handlers against an independent Python
reference book on those recordings.

No account, key or payment is involved. Every endpoint used is public and
unauthenticated.

## The plain idea

A feed handler is only as good as its behaviour on real traffic. The
handler tests in `cpp/ull/feed` and `rust/ull/feed` run on synthetic,
seeded sessions. This tool adds real ones: it keeps every websocket frame
and REST reply exactly as received, with the receive time and the line it
came from. Replaying a file through a handler is then a deterministic test,
and an old capture can be re-run through any later version of a parser.

The check is the one the C++ fault-injection tests already use. **Whenever
a handler says its book is fresh, the book must equal the reference book
at the same point.** When a handler says it is stale, it has told its user
not to trust the book, so it is not judged there.

## What is recorded

| Venue | Stream | Snapshot | Lines | Handlers checked |
|---|---|---|---|---|
| Binance spot | `<sym>@depth@100ms` (diff-depth) | REST `GET /api/v3/depth?limit=1000` after each line-A (re)connect and each file rotation | A and B (two independent connections) | C++ `BinanceDepth`, Rust `Handler(BinanceDiffDepth)` |
| Coinbase Exchange | `level2_batch` (the server names it `level2_50`) | in-band `snapshot` message | A only | C++ `CoinbaseLevel2`, Rust `venue::coinbase_level2` + `CryptoBook` |

### Coinbase `full` and `level2` need keys (finding, 2026-11-03)

Subscribing to `full` or `level2` on `wss://ws-feed.exchange.coinbase.com`
without authentication now returns:

```
{"type":"error","message":"Failed to subscribe","reason":"level2, level3, and full channels now require authentication. ..."}
```

(observed by this recorder on 2026-11-03, evidence class: direct
observation). This project holds no exchange keys and will not create any,
so Coinbase L3 (`full`) cannot be recorded here. The recorder still accepts
`--channel full` or `--channel level2` and stops with the venue's message
(tested against a local server). The REST book `GET /products/BTC-USD/book?level=3`
is still public, but a snapshot without the `full` stream cannot be
replayed into an L3 book. `level2_batch` remains public and uses the same
`snapshot`/`l2update` message shapes as `level2`, so the C++ `CoinbaseLevel2`
handler runs on it unchanged. It is limited to 50 levels a side
(`level2_50`) and batched.

## The format

QPRC v1, documented in `cpp/ull/feed/include/qp/ull/recorder.hpp` and
`qp_recorder/qprc.py`. All integers are little-endian:

```
header : "QPRC", u16 version = 1, u16 reserved = 0
record : u64 recv_ns, u8 line (0 = A, 1 = B, 255 = REST),
         u8 kind (0 = websocket frame, 1 = snapshot body), u32 len, len raw bytes
```

The Rust crate reads the same records behind its own 6-byte magic
`QPRR1\n`; the Rust replay tool swaps the 8-byte QPRC header for it.
`recv_ns` is `time.time_ns()` on the recording machine, taken when the
frame was handed to the recorder, so it includes Python scheduling delay.

Worked example: one frame `{}` at 1 ns on line B is the 8-byte header,
then `01 00 00 00 00 00 00 00 | 01 | 00 | 02 00 00 00 | 7b 7d`.

## The reference book and the comparison

`qp_recorder/refbook.py` shares no code with either handler.

- **Binance.** It collects every distinct event from both lines, orders
  them by final update id `u` whatever their arrival order, starts from the
  first REST snapshot, and applies the chain by Binance's rule (drop
  `u <= lastUpdateId`; the first kept event has
  `U <= lastUpdateId + 1 <= u`; each next one has `U = previous u + 1`). It
  stops at the first id missing from both lines. The key is `u`.
- **Coinbase level2.** No sequence numbers exist, so the key is the record
  index. The book is live from the in-band snapshot on.

Books are compared by a 64-bit digest: the sum, mod 2^64, of
`mix(mix(px) ^ qty ^ tag(side))` over all levels, where `mix` is the
SplitMix64 finaliser and prices and sizes are int64 with 8 decimals. The
Python side updates the sum per level change. The C++ and Rust tools
(`cpp/replay_dump.cpp`, `rust/src/main.rs`) recompute it from the whole
book after every record and print `index stale seq digest`.
`qp_recorder/compare.py` then checks every fresh row.

Before trusting the reference on live data, the tests run it on the
synthetic Binance fixtures in `cpp/ull/feed/tests/fixtures` and require it
to reproduce their expected final books, which come from a separate
generator.

## Recordings in the repo

`recordings/manifest.json` lists each file with its venue, channel, line
count, start time, duration, record and snapshot counts, raw size and the
SHA-256 of the uncompressed bytes. Files are stored gzipped; the tests
decompress them. Rebuild the manifest with
`python -m tools.recorder.recordings.make_manifest`.

| File | Length | Records | Stored |
|---|---|---|---|
| `binance_btcusdt_depth_ab_20261103.qprc.gz` | 89 s, lines A+B | 1,762 | 102 KB |
| `binance_btcusdt_depth_ab_2snap_20261103.qprc.gz` | 60 s, lines A+B, 2 snapshots | 1,174 | 70 KB |
| `coinbase_btcusd_level2batch_20261103.qprc.gz` | 45 s, line A | 684 | 328 KB |

These are short samples. They show that the handlers work on real
traffic, including real A/B duplicates, for a minute or so. They do not
show behaviour across reconnects, venue maintenance or a busy market open.

## Results on these recordings

From `tests/test_replay_live.py` (2026-11-03, local run):

| Recording | Handler | Records | Fresh and checked | Mismatches |
|---|---|---|---|---|
| Binance A+B | C++ | 1,762 | 1,715 | 0 |
| Binance A+B | Rust | 1,762 | 1,715 | 0 |
| Binance A+B, 2 snapshots | C++ | 1,174 | 1,130 | 0 |
| Binance A+B, 2 snapshots | Rust | 1,174 | see below | 0 (was **1,083** before the fix below) |
| Coinbase level2_batch | C++ | 684 | 683 | 0 |
| Coinbase level2_batch | Rust | 684 | 683 | 0 |

### Finding: Rust reloads a snapshot that is not newer than its book

An early version of the recorder fetched a second Binance snapshot by
mistake, at the same `lastUpdateId` the books had already reached. The
C++ `SeqSync` ignores a snapshot that is not newer than the book it holds
(`seqsync.hpp`, "older than the book we already hold"). The Rust
`SyncEngine::on_snapshot` always reloads. The REST book holds only the top
1,000 levels a side, while the diff stream had also updated levels further
out, so the reload drops those levels. Rust then reports fresh with a book
that differs from the reference at the same update id, from that record to
the end. Whether a not-newer snapshot should be ignored, or whether
comparisons should be limited to the snapshot's depth, is for the owner of
`rust/ull/feed` to decide. The recording is kept and the Rust case is a
strict xfail, so a fix shows up as an unexpected pass.

Fixed 2026-11-03 (lane r18-mm-feed): `SyncEngine::on_snapshot` now ignores
a snapshot that is not newer than the live book, as C++ does, and counts it
in `SyncStats::snapshots_not_newer`. The xfail marker is removed; the case
passes with zero mismatches.

## Running

```
# record (from the repo root; needs the `recorder` dependency group)
uv sync --group recorder
python -m tools.recorder.qp_recorder.record binance --symbol BTCUSDT --lines 2 --seconds 60 --out rec/binance.qprc
python -m tools.recorder.qp_recorder.record coinbase --product BTC-USD --seconds 60 --out rec/coinbase.qprc

# check
python -m pytest -q tools/recorder/tests
```

The replay tests need g++ with C++20 (through WSL on Windows) and cargo.
A missing toolchain skips that half of the tests; set
`QP_RECORDER_REQUIRE_TOOLCHAINS=1` to make it a failure instead, as CI
should.

## Recording full days later

1. Run on a machine that stays up, with a stable clock (chrony or
   Windows time sync), from a folder outside the repo:
   `python -m tools.recorder.qp_recorder.record binance --symbol BTCUSDT --lines 2 --seconds 86400 --rotate-minutes 60 --out /data/qprc/binance_btcusdt.qprc`.
   Each hour goes to its own file with a UTC suffix, and each file gets
   its own REST snapshot (Binance) or a fresh subscription with an in-band
   snapshot (Coinbase), so every file replays alone.
2. For runs over 1.5 hours on Windows, start the command detached (a
   scheduled task or a WMI-launched `.cmd` with CRLF line endings) rather
   than from an interactive shell that can kill it.
3. Expect reconnects. Binance closes connections after 24 hours, and both
   venues drop idle or slow clients. The recorder reconnects after 1 s and
   fetches a new Binance snapshot after every line-A reconnect. The gap
   shows in the file as a stale period in the handlers, which is correct.
4. Size: about 0.75 MB per minute raw for Binance BTCUSDT on two lines
   (about 1.1 GB a day, about 100 MB gzipped), and about 1.8 MB per minute raw
   for Coinbase `level2_batch` (the 45 s sample includes its large in-band snapshot, so this overstates the steady rate), measured on the samples above at a quiet
   hour. Do not commit full days. Gzip them, store them outside git, and
   add one manifest entry per file with `"stored_in_repo": false` and the
   SHA-256, so a later run can prove it replays the same bytes.
5. Replay a day with the same tools: `qp_recorder.replay.build_cpp` /
   `build_rust` and `qp_recorder.compare` work on any QPRC file.

## Sources

- Binance Spot API documentation, "WebSocket Streams: Diff. Depth Stream"
  and "How to manage a local order book correctly" (sync rule, REST
  `/api/v3/depth`).
- Coinbase Exchange WebSocket feed documentation, "Level2 channel" and
  "Level2 batch channel" (message shapes), and the authentication error
  quoted above (observed 2026-11-03).
- Steele, Lea and Flood (2014), "Fast splittable pseudorandom number
  generators", OOPSLA, for the SplitMix64 finaliser used in the digest.
- `websockets` (PyPI, BSD-3-Clause), pinned at 17.4 in the `recorder`
  dependency group and hashed in `uv.lock`.
