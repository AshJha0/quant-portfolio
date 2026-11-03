# rust/ull: the low-latency track in Rust (W30)

Plain idea. A trading system's fast path is a short pipeline: packets come
in, are decoded, update a local order book, and an order may go out
through pre-trade checks. These six crates build that pipeline in safe,
std-only Rust, each piece small enough to test on its own, and measure it
call by call. The C++ twin is in [`cpp/ull`](../../cpp/ull).

| Crate | What it holds |
|---|---|
| `ring` | Bounded single-producer single-consumer ring; each side owns one counter, Release/Acquire publication; checked under `loom` (`--cfg loom`) |
| `qpbf` | QPBF, the repository's binary feed format ([docs/QPBF.md](../../docs/QPBF.md)): strict decoder, encoder and a seeded generator, with the golden bytes shared with C++ |
| `book` | L3 (order-by-order) book on flat price-indexed arrays inside a fixed tick window |
| `feed` | Crypto feed handler: line A/B arbitration, gap detection with a timer, snapshot recovery, a stale flag, recorded-session replay (QPRR1 format) |
| `gateway` | Order gateway: kill switch first (the Rust implementation of [docs/KILL_CONTRACT.md](../../docs/KILL_CONTRACT.md), replaying `python/oms/kill/vectors.json` byte for byte), then a stale-data gate, then the pre-trade checks ported from `qp_oms.risk` |
| `bench` | Per-call latency harness: HdrHistogram layout, coordinated-omission-aware fixed-rate runs, timer overhead reported; method in [`bench/METHODOLOGY.md`](bench/METHODOLOGY.md) |

Formal statement. Correctness is shown by agreement, not by inspection:
the decoder and book against the QPBF golden bytes and the C++ book; the
gateway against the kill vectors and the pre-trade vectors generated from
the Python reference (same decisions, same order); the ring under `loom`
model checking.

Worked example. Running the kill vectors through `gateway::kill::replay`
must print exactly the expected lines in `vectors.json`; one changed byte
fails the test (`cargo test -p qp-ull-gateway`).

**Every latency figure from these crates is software in-process latency,
not wire-to-wire.** No figure includes a NIC, kernel network stack, switch
or wire, and figures from one laptop under WSL2 are upper bounds on
jitter, not a hardware claim ([docs/LATENCY_METHOD.md](../../docs/LATENCY_METHOD.md)).

Not done: live recording of exchange sessions (a TLS websocket client
needs crates outside std, and tests have no network), so feed replay runs
on fixture sessions only.

```
cd rust
cargo test --release --locked -p qp-ull-ring -p qp-ull-qpbf -p qp-ull-book -p qp-ull-feed -p qp-ull-gateway -p qp-ull-bench
```
