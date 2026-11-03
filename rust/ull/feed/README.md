# rust/ull/feed: crypto market-data feed handler (Rust)

Part of W30 (low-latency track), part 2; the Rust twin of `cpp/ull/feed`.
std only, plus the part-1 crates `qp-ull-book` and `qp-ull-qpbf`. Tests run
offline and never open a socket.

## What it does

The handler turns an exchange's stream of book updates into a local copy of
the order book and says when that copy cannot be trusted (`is_stale()`;
W26 pulls quotes while it is true).

| Venue / channel | Book | Sequencing | Parser |
|---|---|---|---|
| Binance spot `<sym>@depth` + REST `/api/v3/depth` | L2 | update-id range `[U, u]` | `fast::binance_into`, `venue::binance` |
| Coinbase Exchange `full` + REST `book?level=3` | L3 | `sequence` per message | `venue::coinbase_full` |
| Coinbase Exchange `level2_batch` (or `level2`) | L2 | none | `venue::coinbase_level2` |

Access. Binance diff-depth and Coinbase `level2_batch` are public and need
no account. Coinbase `full` and `level2` now require an authenticated
connection (Coinbase server message seen 2026-11-03 by the recorder lane,
`tools/recorder`); this project uses no accounts, so the `full` path is
tested on synthetic messages only. `level2_batch` carries the same
`snapshot` / `l2update` messages, batched.

## Pipeline

raw bytes from line A or B -> parser -> `sync::Arbiter` (first arrival wins,
bounded hold for reordering) -> `sync::SyncEngine` (buffer while out of
sync, bridge the REST snapshot, then strict continuity; any gap marks the
book stale and asks for a new snapshot) -> `book::CryptoBook`.

A REST snapshot that is **not newer** than the live book is ignored
(counted in `SyncStats::snapshots_not_newer`). Reloading it would rewind the
book and drop the levels the stream had added beyond the snapshot's depth;
the recorder lane found exactly that on a live two-snapshot Binance
recording (1,083 fresh states differed from the reference before the fix,
none after). `cpp/ull/feed` already behaved this way.

## Books and the hot path

`CryptoBook` keeps the touch in the part-1 flat-array book
(`qp_ull_book::Book`):

- `ProductSpec` (tick, lot, window width, order capacity) maps a
  fixed-point price to an integer tick and a size to whole lots. Default: a
  0.01 tick, a 1e-8 lot, a window of 65,536 ticks.
- The window is centred on the touch when a snapshot loads. A bid above it
  or an ask below it means the touch has moved out: `ApplyError::OutOfWindow`,
  the engine goes stale, and the recovery snapshot re-centres the window.
- Deep levels, prices off the tick grid and sizes above 2^32 lots go to an
  exact side store (ordered maps), so the book always equals the venue's.
- Coinbase UUIDs are interned to u64 handles in a fixed-capacity table.
- Binance events are read by `fast::binance_into` straight into a reused
  `Delta`. It returns `Fallback` for anything it does not fully understand,
  and the generic parser decides; a differential test checks every message
  it accepts against the generic parser, on generated days and fuzzed bytes.

In sync and in order, a Binance message is decoded, arbitrated and applied
with **zero heap allocations** (`tests/no_alloc.rs`, a counting global
allocator, per thread). Coinbase messages still go through the generic JSON
reader, which allocates; their book operations near the touch do not.

## Tests

`cargo test --release -p qp-ull-feed` from `rust/`: unit tests in each
module, `tests/inject.rs` (drop, duplicate, reorder, burst on lines A/B,
fuzzing, recorder replay) and `tests/no_alloc.rs` (zero allocations, fast
versus generic decoder, a touch move out of the window going stale and
recovering, order-feed book operations). Live recordings are replayed by
`tools/recorder/tests/test_replay_live.py`.

## Sources

- Binance spot API documentation, "Diff. Depth Stream" and "How to manage a
  local order book correctly".
- Coinbase Exchange websocket documentation, `full`, `level2` and
  `level2_batch` channels; REST "Get product book".
- Knuth, TAOCP vol. 3, section 6.4, Algorithm R (deletion with linear
  probing), used by the id interner.
