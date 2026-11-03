# cpp/ull: the low-latency track in C++ (W30)

Plain idea. The fast path of a trading system is short: packets arrive,
are decoded, update a local order book, and an order may leave through
pre-trade checks. These header-mostly components build that path in
C++23 with no allocation in the steady state, each tested on its own and
timed call by call. The Rust twin is in [`rust/ull`](../../rust/ull).

| Folder | What it holds |
|---|---|
| `ring/` | Bounded SPSC ring (`spsc_ring.hpp`), run under ThreadSanitizer, with a GenMC model-checking harness in `ring/genmc/` |
| `qpbf/` | QPBF binary feed ([docs/QPBF.md](../../docs/QPBF.md)): strict codec, seeded generator, fuzz target; golden bytes shared with Rust |
| `book/` | Flat-array L3 book (`l3_book.hpp`) |
| `mcast/` | Multicast A/B feed handler on QPBF: gap timer, snapshot recovery, stale flag (`FeedHandler::stale()`), recording and replay publisher |
| `feed/` | Crypto feed handler on recorded Coinbase and Binance sessions; see [`feed/README.md`](feed/README.md) |
| `gateway/` | Order gateway: kill switch per [docs/KILL_CONTRACT.md](../../docs/KILL_CONTRACT.md), stale gate, pre-trade checks ported bit for bit from `qp_oms.risk`; see [`gateway/README.md`](gateway/README.md) |
| `bench/` | Per-call latency harness (invariant TSC with fences, HdrHistogram layout, coordinated-omission-aware runs, system info in the output) |

Formal statement. Each component is trusted only through an independent
check: the codec against golden bytes and a fuzzer; the book against the
Rust book and the Python reference book; the ring under TSan and GenMC;
the gateway against `python/oms/kill/vectors.json` and the pre-trade
vectors generated from the Python reference.

Worked example (from the Round 13 tests). The steady-state loop ring ->
decode -> A/B arbitration with gaps -> book -> histogram performs 0
allocations. With 200 injected 1 ms stalls, about 200 samples land above
0.9 ms and p99.9 stays under 1 us; at a 0.2 % stall rate the stall moves
into p99.9, which is the point of reporting the tail and not the mean.

**Every latency figure from `cpp/ull` is software in-process latency, not
wire-to-wire** ([docs/LATENCY_METHOD.md](../../docs/LATENCY_METHOD.md)). No
NIC, kernel network stack, switch or wire is in any number. Figures from
one laptop under WSL2 are a record of that machine; current figures come
from the CI job artifacts.

Build and test: each of `feed/`, `gateway/` and `bench/` is a standalone
CMake project (the ring, codec and book are header-only and are built by
their tests there):

```
cmake -S cpp/ull/bench -B build/ull_bench -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/ull_bench && ctest --test-dir build/ull_bench -LE long
```

`.github/scripts/cpp_extra.sh` runs the same lines for all three in CI.
