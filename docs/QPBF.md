# QPBF: the quant-portfolio binary feed

QPBF is a small market-data feed format written for this repository. It
exists so the low-latency code (`cpp/ull/`, and the Rust lane's matching
crate) has a realistic binary feed to decode, arbitrate and recover without
any licensed exchange data. It is modelled on the general shape of
order-by-order exchange feeds such as CME MDP 3.0 and the US equity venues' direct feeds
(fixed-size binary messages, per-message sequence numbers, two redundant
multicast lines and a separate recovery channel). It does not copy any
format, and nothing in it claims to match a real venue.

This page is the specification. The C++ reference implementation is
`cpp/ull/qpbf/qpbf.hpp`; any other implementation must decode the golden
bytes in section 10 to the same values and must re-encode every accepted
packet to identical bytes.

Version: 1. Status: stable for v3.0.0. A change to any byte layout bumps the
version byte.

## 1. The idea in one paragraph

An exchange tells you about every order: added, changed, cancelled,
executed. Each such event is one fixed-size message. Messages are packed
into UDP packets. Every message has a sequence number, so a receiver knows
when it missed one. The same packets go out on two lines (A and B); the
receiver takes each message from whichever line delivers it first. If both
lines lose a message, the receiver asks a TCP snapshot server for the whole
book as of some sequence number, then carries on from there.

## 2. Conventions

- All integers are little-endian, unsigned unless marked signed.
- `u8`, `u16`, `u32`, `u64`: unsigned of that many bits; `i64`: two's
  complement signed.
- Prices are `i64` integer ticks. The tick size is a property of the
  instrument and is not on the wire.
- Timestamps are `u64` nanoseconds since the Unix epoch, exchange clock.
- Every padding byte must be zero. A decoder must reject non-zero padding.
- Offsets are in bytes from the start of the structure.

## 3. Packet

A packet is one UDP datagram: a 16-byte header followed by `msg_count`
messages back to back. A packet is at most 1472 bytes (a 1500-byte Ethernet
MTU minus 20 bytes of IPv4 and 8 bytes of UDP header), and holds at most 255
messages.

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 2 | magic | `0x5051`; on the wire the bytes `51 50` (`'Q' 'P'`) |
| 2 | 1 | version | `1` |
| 3 | 1 | msg_count | number of messages in this packet (0-255) |
| 4 | 4 | session_id | identifies one run of the feed; a new value means sequence numbers restart |
| 8 | 8 | seq | sequence number of the first message in the packet |

Message `i` (0-based) in the packet has sequence number `seq + i`. Sequence
numbers start at 1 for each session and every message consumes one,
heartbeats included. A packet with `msg_count = 0` is legal and consumes
none.

The packet must end exactly after the last message. Trailing bytes, a short
packet, or a message that runs past the end make the whole packet invalid.
A decoder validates the whole packet before acting on any message in it,
so a malformed packet changes no state.

## 4. Messages

Every message starts with the same 16 bytes:

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 2 | length | total message size in bytes; must equal the size for the type |
| 2 | 1 | type | see table below |
| 3 | 1 | flags | must be 0 except bit 0 on a Snapshot message (section 6) |
| 4 | 4 | instrument | instrument id; 0 on a heartbeat |
| 8 | 8 | ts_ns | exchange timestamp |

| Type | Value | Size | Meaning |
|---|---:|---:|---|
| Add | 1 | 40 | a new visible order |
| Modify | 2 | 40 | an order's price and/or quantity changed |
| Cancel | 3 | 32 | part or all of an order removed by its owner |
| Execute | 4 | 40 | part or all of an order filled |
| Trade | 5 | 40 | a trade against non-displayed liquidity (no book change) |
| Snapshot | 6 | 48 | one order of a recovery snapshot (TCP channel only) |
| Heartbeat | 7 | 16 | nothing happened; proves the line is alive |

Body fields after the 16-byte prefix:

**Add (40) and Snapshot (48)**

| Offset | Size | Field | Rule |
|---:|---:|---|---|
| 16 | 8 | order_id | non-zero, unique among live orders of the session |
| 24 | 8 | price | `i64` ticks |
| 32 | 4 | qty | non-zero |
| 36 | 1 | side | 1 = bid, 2 = ask |
| 37 | 3 | padding | zero |
| 40 | 8 | as_of_seq | Snapshot only (section 6) |

**Modify (40)**: 16 order_id (non-zero), 24 price `i64`, 32 qty (non-zero,
the new remaining quantity), 36 four bytes of padding.

**Cancel (32)**: 16 order_id (non-zero), 24 qty (non-zero, quantity
removed), 28 four bytes of padding.

**Execute (40)**: 16 order_id (non-zero), 24 qty (non-zero, quantity
filled), 28 four bytes of padding, 32 match_id `u64`.

**Trade (40)**: 16 match_id `u64`, 24 price `i64`, 32 qty (non-zero), 36
aggressor side (1 or 2), 37 three bytes of padding.

**Heartbeat (16)**: the prefix only.

## 5. Book rules

These rules define what a receiver's order book must contain after each
message. The C++ book (`cpp/ull/book/l3_book.hpp`) and the reference book in
its tests implement them; the W25 reference book is expected to match.

1. **Add** puts the order at the back of the queue at its price and side.
2. **Modify** to the same price with the same or a smaller quantity keeps
   the order's queue position. Any other modify (new price, or a larger
   quantity) moves the order to the back of the queue at the new price.
   This is the usual price-time priority rule; venues differ in detail, and
   QPBF fixes this one.
3. **Cancel** and **Execute** reduce the order by `qty`. If that leaves zero
   the order is removed. `qty` larger than the remaining quantity is a
   protocol error.
4. **Trade** and **Heartbeat** do not change the book.
5. An Add whose `order_id` is already live, or a Modify, Cancel or Execute
   naming an unknown order, is a protocol error. The receiver must treat
   its book as no longer trustworthy (stale) and recover with a snapshot.

Worked example. Start empty. `Add(1, bid, 103, 10)`, `Add(2, bid, 103, 5)`,
`Add(3, ask, 105, 7)`: best bid 103 with 15 lots in two orders, 1 ahead of 2;
best ask 105. `Modify(1, 103, 4)` keeps order 1 first (same price, smaller
quantity). `Modify(1, 103, 6)` moves it behind order 2. `Execute(2, 5)`
removes order 2. The bid side is now order 1 alone, 6 lots at 103.

## 6. Snapshot channel (TCP)

The snapshot server listens on TCP. A client connects; the server sends the
book and closes. Each frame on the stream is a `u16` little-endian length
followed by one QPBF packet of that length (same 16-byte header, same
1472-byte limit).

- Every packet in one snapshot has the same `session_id` as the live feed,
  and its header `seq` is the snapshot's as-of sequence `S`.
- Every message is a Snapshot message with `as_of_seq = S`. Snapshot
  messages do not consume sequence numbers.
- Orders are sent price level by price level in queue order, so the receiver
  rebuilds the same queue positions.
- The last message has `flags` bit 0 set (LAST). If it carries `order_id 0`
  it is an end marker only: `price`, `qty` and `side` must then be zero.
  An empty book is a snapshot with just the end marker.
- The book state is that after applying messages 1..S of the session.

## 7. Receiver behaviour (non-normative)

This is how `cpp/ull/mcast/feed_handler.hpp` uses the format; a receiver
may differ, but must never present a book as current when it is not.

- **Arbitration.** Take sequence number `n` from whichever line delivers it
  first; any later copy is a duplicate and is dropped.
- **Gap.** A message with sequence above the next expected one opens a gap.
  The book is flagged stale at once (a quoting strategy pulls its quotes on
  this flag). Later messages are held in a reorder window.
- **Fill.** If the other line delivers the missing messages before a
  timeout, they are applied, the window drains, and the flag clears.
- **Recover.** If the timeout passes, or the hole is wider than the window,
  the receiver requests a snapshot, keeps buffering live messages, applies
  the snapshot as of `S`, discards buffered messages at or below `S`,
  applies the rest in order, and clears the flag once contiguous.
- **New session.** A new `session_id` resets everything. If its first
  packet starts at sequence 1 the book starts empty; otherwise the receiver
  joined late and needs a snapshot.

A tail gap (the last messages lost on both lines) is only noticed when the
next message arrives; this is why the publisher sends heartbeats.

## 8. What QPBF does not do

- No checksum. Integrity of each datagram rests on the UDP checksum, as on
  most exchange feeds. A field corrupted in transit but still well formed
  would decode. The decoder rejects every structural error (section 3-4).
- No retransmission request channel for single messages (MoldUDP64 has
  one); recovery is by snapshot only.
- No instrument definitions, trading status or auction messages. Those
  would be new message types in a later version.
- Nothing here measures or implies wire latency (see `LATENCY_METHOD.md`).

## 9. Recordings

A recording keeps the raw packet bytes so a replay reproduces the feed
exactly: the 8-byte magic `QPBFREC1`, then for each packet a `u16`
little-endian length and the packet bytes. A record longer than 1472 bytes
makes the file invalid. `ull_replay_publisher generate` writes one from the
seeded generator; `ull_replay_publisher replay` publishes one on two
loopback multicast groups (239.255.10.1 and 239.255.10.2 by default; the
239.0.0.0/8 range is administratively scoped, RFC 2365) with a snapshot
server.

## 10. Golden bytes

An Add for instrument 7, timestamp `0x0102030405060708`, order 42, price
-3, quantity 100, ask, in a packet of session `0xAABBCCDD` starting at
sequence 1000:

```
packet header  51 50 01 01 DD CC BB AA E8 03 00 00 00 00 00 00
message        28 00 01 00 07 00 00 00 08 07 06 05 04 03 02 01
               2A 00 00 00 00 00 00 00 FD FF FF FF FF FF FF FF
               64 00 00 00 02 00 00 00
```

The C++ test `wire_layout_is_fixed` in `cpp/ull/qpbf/tests/test_qpbf.cpp`
checks exactly these 56 bytes. The Rust implementation should carry the
same test.

## 11. Generator

`cpp/ull/qpbf/generator.hpp` produces a seeded, valid message stream: every
cancel, execute and modify names a live order and never removes more than
it has. It keeps a ledger of how many messages of each type it produced, so
a decoder run can be checked count for count. The random source is
xoshiro256** seeded by splitmix64 (Blackman and Vigna, "Scrambled linear
pseudorandom number generators", ACM TOMS 47(4), 2021). The same seed gives
the same bytes on every platform.

The order flow is deliberately simple (a random-walk mid, geometric
distance of new orders from the mid). It is a test load, not a market model;
the plan's W25 simulator is meant to drive it later.

## 12. Evidence

| Claim | Evidence class | Where |
|---|---|---|
| 1e8 generated messages decode and re-encode bit-exactly; per-type counts equal the ledger | test, run locally | `ull_qpbf_roundtrip_1e8` (CTest label `long`) |
| Malformed packets never crash the decoder; every accepted packet is canonical | test + fuzzing | mutation fuzz in `test_qpbf`, libFuzzer target `qpbf/fuzz/fuzz_decode.cpp` |
| Wire layout is as in section 10 | golden test | `wire_layout_is_fixed` |
| A/B, gap, snapshot rules converge for drop, duplicate, reorder, burst | test | `mcast/tests/test_recovery.cpp` |
