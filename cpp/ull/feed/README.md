# cpp/ull/feed: crypto market-data feed handler

Part of W30 (low-latency track), part 2. Header-only C++20; no third-party
dependencies. Tests run offline on recorded fixtures and never open a socket.

## What it does

A feed handler turns an exchange's stream of book updates into a local copy
of the order book, and says honestly when that copy cannot be trusted.

| Venue / channel | Book | Sequencing | Header |
|---|---|---|---|
| Coinbase Exchange `full` + REST `book?level=3` | L3 (every order) | `sequence` per message | `handlers.hpp` `CoinbaseFull` |
| Binance spot `<sym>@depth` + REST `/api/v3/depth` | L2 (levels) | update-id range `[U, u]` | `handlers.hpp` `BinanceDepth` |
| Coinbase Exchange `level2_batch` (or `level2`) | L2 | none | `handlers.hpp` `CoinbaseLevel2` |

Access. Binance diff-depth and Coinbase `level2_batch` are public and need
no account. Coinbase `full` and `level2` now require an authenticated
connection (Coinbase server message seen 2026-11-03 by the recorder lane,
`tools/recorder`); this project uses no accounts, so `CoinbaseFull` is tested
on synthetic fixtures only, and live recordings use `level2_batch`, which
carries the same `snapshot` / `l2update` messages batched over 50 ms. The
handler does not own the transport:
the caller passes raw websocket text and REST bodies in, which keeps tests
offline and lets the recorder sit in front of the parser.

## The plain idea

Updates carry sequence numbers. We remember the last one applied. An update
we have already applied is a duplicate (usually the second copy from the
other line of an A/B pair) and is dropped. The next one is applied. One
from further ahead means something is missing: we hold it and mark the book
**stale**. If the missing update turns up (reordering, or the other line has
it), the held updates drain and the book is fresh again. If it does not turn
up within a timeout or a window, we ask for a REST snapshot, keep buffering,
rebuild from the snapshot, discard buffered updates the snapshot already
covers, and apply the rest.

## Formal statement

`seqsync.hpp` holds the whole rule as one state machine, `SeqSync<Msg>`.
Each update has a range `[first, last]` (`first == last` for Coinbase).
With `next = applied + 1`:

- `last < next`: duplicate, dropped.
- `first <= next <= last`: applied; then held updates drain while they apply.
- `first > next`: held; the book is stale.

Invariants checked by the tests:

- (I1) Each applied update satisfies `first <= applied_before + 1 <= last`.
  This is Binance's documented rule. For single sequence numbers it means
  exact succession.
- (I2) `stale() == !synced || !held.empty()`.
- (I3) No update is applied twice.

A hole is declared lost when it is older than `gap_timeout_ns` (default
50 ms) or more than `reorder_window` updates (default 64) are held. Then
exactly one snapshot request is outstanding. A snapshot older than the book
already held is ignored, because loading it would silently lose updates.
This case happens when the hole fills while the request is in flight.

The stale flag is exposed as `stale()` and as an `on_stale(bool)` callback.
The gateway (`cpp/ull/gateway`) uses it to cancel working orders on the
symbol and refuse new ones until the book is fresh ("pull quotes", the W26
seam).

Coinbase `level2` has no sequence numbers, so it cannot detect a lost
update. It is stale until its in-band snapshot arrives, and after every
reset. A/B copies are dropped by exact payload, using a 64-bit hash over
the last 4,096 frames. A hash collision would drop a genuine frame; with
4,096 frames in the window the chance is about 2^-52 per frame. Sizes are
absolute, so applying a duplicate twice would do no harm anyway. Two
reordered updates to the same level do harm, and this channel cannot detect
them. Use `full` where gap safety matters.

## Worked example

From `tests/test_feed.cpp`, case `seqsync_worked_example`:

1. Updates 9 and 11 arrive before any snapshot. Both are buffered, and one
   snapshot request goes out.
2. A snapshot at sequence 10 loads. Update 9 is discarded because the
   snapshot covers it. Update 11 is applied. The book is fresh.
3. Update 13 arrives. Update 12 is missing, so 13 is held and the book is
   stale.
4. Update 12 arrives. 12 applies, then 13 drains. The book is fresh again.
5. Line B delivers 12 again. It is counted as a duplicate.

## Books and the hot path

`BinanceDepth` and `CoinbaseFull` keep their books in the part-1 flat-array
L3 book (`cpp/ull/book/l3_book.hpp`) through `flatbook.hpp`:

- A per-product `ProductSpec` (tick, lot, window width, order capacity) maps
  a fixed-point price to an integer tick and a size to whole lots. The
  default is a 0.01 tick, a 1e-8 lot and a window of 16,384 ticks.
- The window is centred on the touch when a snapshot loads. A bid above it
  or an ask below it means the touch has moved out: the update throws
  `OutOfWindow`, the handler resets (stale, snapshot requested), and the
  snapshot re-centres the window.
- Anything else the window cannot hold - deep levels, prices off the tick
  grid, sizes above 2^32 lots - goes to an exact side store (a map-based
  book), so the book is always complete. The map-based `L2Book` and
  `L3Book` stay as the test oracle: the flat books match them call for call
  on 50,000 and 60,000 random operations.
- Coinbase order ids (strings, at most 48 bytes) are interned into a
  fixed-capacity open-addressing table (`IdInterner`).
- Binance frames go through `fastdepth.hpp` first, a decoder that reads the
  event straight into a reused `DepthDelta`. It falls back to `json::parse`
  for anything it does not fully understand (escapes, a wrapper it does not
  expect, duplicate keys, a snapshot body), so it can only be faster, never
  different; a differential test checks every frame it accepts against the
  generic path on the fixtures and on fuzzed bytes.

In sync and in order, a Binance frame is decoded, sequenced and applied
with **zero heap allocations** (`tests/test_noalloc.cpp` replaces `operator
new` and counts). The same frames through the generic `json::parse` path
allocate on every frame. Coinbase frames still go through `json::parse`;
their book operations do not allocate.

## Prices

Prices and sizes are parsed from the exchange's decimal strings into int64
fixed point with 8 decimal places (`decimal.hpp`). They never pass through a
double, so `"0.1" + "0.2"` is the same level as `"0.3"`. More than 8
non-zero decimals, or an overflow, throws instead of rounding.

## Recorder

`recorder.hpp` writes every frame exactly as received: receive time, line
(A, B or REST), kind (frame or snapshot) and the raw bytes, in the
little-endian QPRC v1 format documented in the header. Replay feeds the
bytes back through the same handler. A file cut short by a crash replays up
to the last whole record. Keeping the raw bytes is the plan's answer to
exchange format changes: when a parser changes, old sessions can be replayed
through it.

## Tests

`tests/test_feed.cpp`, 17 cases, and `tests/test_noalloc.cpp`, 2 cases:

- Fixture replay to the expected final book, on 2 Coinbase `full` sessions
  and 2 Binance sessions.
- Fault injection on lines A and B: drop on one line, drop on both,
  duplicate, reorder and a 200-frame burst. Each pattern runs with 3 seeds
  per venue. After every delivery the test checks one thing: whenever the
  handler reports *not stale*, its book equals the reference book at the
  same sequence. Every run must converge to the final book.
- 20,000 mutated frames, none of which may throw out of the handler.
- Recorder round trip and every truncation point.
- Flat books against the map books; the id interner against a hash map;
  the fast Binance decoder against the generic one; a touch move out of the
  window going stale and recovering.
- Zero allocations on the Binance hot path, and in flat L3 book operations.

The fixtures are **synthetic**. See `tests/fixtures/README.md`.

Build and run:

```
cmake -S cpp/ull/feed -B build/ull-feed && cmake --build build/ull-feed && ctest --test-dir build/ull-feed
```

## Not done here

- Recorded days. The acceptance was re-scoped by the owner (2026-11-03) to 3
  recorded days each of Binance diff-depth and Coinbase `level2_batch`;
  `tools/recorder` holds the capture tool and the recordings so far.
- A non-allocating Coinbase decoder; Coinbase frames still parse to a DOM.

## Sources

- Coinbase Exchange websocket feed documentation: the `full` and `level2`
  channels, and sequence numbers ("drop messages with sequence <= the
  snapshot's").
- Coinbase Exchange REST API, "Get product book", level 3.
- Binance spot API documentation, "How to manage a local order book
  correctly" (diff-depth stream and `/api/v3/depth`).
