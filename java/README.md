# qp-ull-java: ring buffer, garbage-free gateway and OMS core

Part of W30, the low-latency track. Decision D2 scopes Java to "an
LMAX-Disruptor-style ring buffer plus a garbage-free order gateway and OMS
core, with JMH benchmarks and GC-pause tracking, in microseconds, zero GC in
steady state". C++ and Rust hold the nanosecond hot path. This module shows
how the same order path is built on the JVM, and what it costs.

Build: JDK 21 or later, through the Maven wrapper. Nothing needs installing
beyond a JDK, and there are no paid dependencies.

```
cd java
./mvnw -B verify                          # tests, then target/benchmarks.jar
java -jar target/benchmarks.jar -prof gc  # JMH, microseconds per operation
java -cp target/classes io.quantportfolio.ull.bench.LatencyMain 100000 10
```

## What is here

| Package | What it does |
|---|---|
| `ring` | `RingBuffer`: single producer, pre-allocated slots, padded `Sequence` counters, release/acquire publication, busy-spin `BatchEventProcessor` |
| `kill` | `KillSwitch`: the shared kill contract (`docs/KILL_CONTRACT.md`, rules K1-K7) |
| `risk` | `PreTradeRisk`: throttle, duplicate ClOrdID, restricted list, fat finger, notional, price band, position and credit limits (both counting working orders), self-trade prevention |
| `oms` | `OmsCore`: order rows stored as parallel arrays, the FIX OrdStatus transitions, per-symbol position and working exposure |
| `gateway` | `OrderGateway`: one thread owns all of the above and turns `Command`s into `Report`s. Every report carries a clock source and an error bound |
| `metrics` | `LatencyHistogram` (log-linear, garbage-free) and `GcPauseTracker` (HotSpot GC notifications) |
| `bench` | JMH benchmarks and the three-thread `LatencyMain` harness |

## The ring buffer

**Plain idea.** The ring is an array of event objects, all created at start-up.
The producer overwrites a slot and moves a counter forward. The consumer
reads up to that counter. Because nothing is created after start-up, the
garbage collector has nothing to do.

**Formal statement.** The producer may claim slot `s` only when
`s - size <= min(consumer sequences)`. A consumer reads slot `s` only when
`s <= cursor`. The cursor is written with release semantics after the slot's
fields, and read with acquire semantics, so the consumer sees those fields
(JLS 17.4; `VarHandle` access modes). Counters are padded onto their own
64-byte cache lines. This is the design of the LMAX Disruptor (Thompson,
Farley, Barker, Gee and Stewart, "Disruptor", LMAX technical paper, 2011),
re-implemented here without the dependency.

**Worked example.** Take a ring of size 4 with its consumer at -1. The
producer publishes 0 to 3. To claim slot 4 it needs `4 - 4 = 0 <= consumer`,
so it spins until the consumer has read slot 0 (`RingBufferTest.tryNextRefusesToLapAConsumer`).

## Kill contract parity

`KillContractParityTest` reads the Python reference's golden vectors
(`python/oms/kill/vectors.json`) and requires the same canonical JSON event
lines, byte for byte. Current result: 3 of 3 cases identical (worked example,
nesting and duplicate, cancel on disconnect). The gateway then applies the
contract to real orders. A kill sends `SEND_CANCEL` for every covered working
order in ascending id order (K2). New orders in that scope are rejected with
`KILLED`. The kill reaches KILLED only after every one of those orders is
terminal, and a fill that races the cancel counts as terminal (K3). Reset
needs two different people (K4). A disconnect engages a STRATEGY kill for
every strategy on the session (K6). See `GatewayTest`.

## Zero garbage in steady state: how it is checked

`AllocationTest` drives the gateway through 500,000 cycles after warm-up,
which is 5.5 million gateway events: new, ack, partial and full fill, cancel,
cancel ack, plus kill, duplicate and price-band rejects. It reads the
thread's allocated bytes from `com.sun.management.ThreadMXBean`, which is
HotSpot's per-thread TLAB accounting. **Result: 0 bytes, and 0 GC
collections in the window.** A negative-control test shows that the meter
does see allocation, so the zero is not a dead meter. JMH `-prof gc` agrees,
at about 0 B/op (`gc.alloc.rate.norm` 0.016 B/op for the round trip, which is
JMH's own infrastructure amortised over the operations).

Not garbage-free, by design: interning symbols at session set-up,
`cancelOnDisconnect` (it runs once per session drop), and rendering the
contract's JSON wire form (used only in parity tests).

## Measured numbers (honest labels)

All figures are **software in-process latency, not wire-to-wire**. Machine:
the owner's Windows 11 laptop, 12 logical CPUs, HotSpot 27.0.1 (the code is
compiled for release 21). The machine was **heavily shared**: about twenty
other build and test jobs were running at the same time. Read these as an
upper bound under contention, not as what the code does on an isolated
core. Threads are not pinned (the JDK has no portable affinity API).

JMH, single thread, average time (`java -jar target/benchmarks.jar -bm avgt -prof gc`, 3x1 s warm-up, 5x1 s measurement, 1 fork), 2026-11-03:

| Benchmark | us/op | Allocation |
|---|---|---|
| `ringPublishPoll` (one publish and one poll, same thread) | 0.048 ± 0.005 | ≈ 0 B/op |
| `killCheck` (K1 against 8 active kills) | 0.106 ± 0.062 | ≈ 0 B/op |
| `orderRoundTrip` (NEW + ACK + full FILL: 3 events, 3 reports) | 2.63 ± 1.89 | ≈ 0 B/op |

The wide error bars on the last two rows come from the contention, not from
the code.

`LatencyMain`, three busy-spinning threads (client to gateway to venue), two
ring hops, with latency measured from each order's *intended* send time, so
stalls are charged (coordinated-omission correction, after Gil Tene):

| Rate | Orders | p50 us | p99 us | p99.9 us | max us | GC in window |
|---|---|---|---|---|---|---|
| 10,000/s (one run) | 50,000 | 1.70 | 5,374 | 11,403 | 16,258 | 0 |
| 10,000/s (a later run) | 100,000 | 1,343 | 8,126 | 12,452 | 16,738 | 0 |
| 100,000/s | 1,000,000 | 2,015 | 14,287 | 25,690 | 31,801 | 0 |

Those runs shared the machine with other work. On an idle machine
(2026-11-04, [docs/BENCHMARKS.md](../docs/BENCHMARKS.md)) the p50 was 2.88 us at
10,000/s and 2.69 us at 100,000/s, with p99 near 65 us and no collections.

How to read this. The 1.7 us median is what the path costs when its three
threads keep their cores. The millisecond figures are the OS taking those
cores away from spinning threads on a machine with about 20 other busy
processes. With coordinated-omission correction, every order queued behind
such a stall is charged for it. GC is not the cause: the window shows zero
collections. A fair figure needs an idle machine, or a Linux runner with
`isolcpus` and pinned threads. CI can produce shared-VM indicative numbers
(record the CPU model), and nothing here should be quoted as a production
latency.

## Limits

- Single producer per ring. The gateway polls the client ring and the venue
  ring in turn, rather than using a multi-producer ring.
- Fixed capacities (orders, kills, symbols, dedupe window) are set at
  construction. Going past one throws, and does not resize.
- Duplicate ClOrdID detection covers live orders plus the last N terminal
  ids (N is configurable). It is not a whole-session set.
- No FIX codec and no persistence here. The Python W28 path owns those.
