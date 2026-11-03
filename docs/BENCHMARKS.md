# Benchmarks: how latency is measured

## The idea

A single fast call (100 ns) is close to what the clock can resolve, and a
single slow call is easy to time. So every bench, in C++ and in Rust, times
a small batch of back-to-back calls, sized so that one batch lasts at least
2 microseconds, and divides by the batch size. It repeats that many times and
reports the median with the 10th and 90th percentiles as the spread.

To decide whether a change made the code slower, the CI bench job does not
compare with numbers from another machine. It builds the old commit and the
new one on the same runner and runs them in turns.

This page covers the engine and kernel benches. The low-latency track
(`cpp/ull`, `rust/ull`) times single calls with coordinated-omission
correction instead; its method is [LATENCY_METHOD.md](LATENCY_METHOD.md),
and the Java figures (JMH and `LatencyMain`) are in
[`../java/README.md`](../java/README.md). All of them are software,
in-process latency on one machine, not wire-to-wire.

## The method, stated exactly

Shared code: `cpp/core/include/qp/bench/latency.hpp` (C++) and
`rust/core/src/bench.rs` (Rust, `qp_core::bench`).

1. Warm up with the caller's number of untimed calls.
2. Batch size B: start at 1 and double while the fastest of 5 trial
   batches of B calls takes less than 2 us, up to B = 1024. Clock:
   `std::chrono::steady_clock` / `std::time::Instant`. B is written as
   `calls_per_sample`.
3. N = max(31, call budget / B) samples, each one batch divided by B (ns per
   call). Nothing is subtracted.
4. Percentiles by nearest rank (element ceil(qN) - 1 of the sorted
   samples): p10, p50, p90, p99, p99.9, plus max and mean. When B > 1 these
   are percentiles of B-call batch means, and the tails are labelled
   "p99 of B-call batch means".

Each record in the JSON output has `kernel`, `calls_per_sample`, `unit`
(`ns`), `p10`, `p50`, `p90`, `p99`, `p999`, `max` and `tail_basis`. The last
field is `per-call` when B = 1 and otherwise reads, for example,
`p99 of 32-call batch means`. The C++ harness
(`cpp/core/include/qp/bench/latency.hpp`) and the Rust one
(`rust/core/src/bench.rs`) write the same wording.

Worked example. `bs_price` takes about 120 ns. One call is under 2 us, and
so are batches of 2, 4 and 8; a batch of 16 takes about 1.9 us and one of 32
about 3.8 us, so B = 32 (or 16 on a faster machine). With a budget of
100,000 calls that gives 3,125 samples, each a 32-call average.

Percentiles of B-call averages are narrower than percentiles of single
calls: a rare 10 us stall is spread over B calls. The p99, p99.9 and max of
a fast kernel are therefore the p99 of B-call batch means, not single-call
tails, and must not be quoted as per-call tail latency or used for an SLO. Slow kernels (Monte Carlo,
binomial trees, VaR over long histories) run with B = 1 and their tails are
single calls.

## The CI gate (same runner, base against head)

`.github/scripts/bench_ab_run.sh` checks out the base commit (the pull
request's target, or the previous commit on a push) into a worktree, builds
both commits' C++ (GCC 15, Release) and Rust (release) benches, and runs
them in three alternating rounds: base, head, base, head, base, head. Every
run is pinned to CPU 1. `.github/scripts/bench_ab.py` then takes, per kernel
and side, the median over the rounds of p10, p50 and p90.

Both sides are timed with the head's harness: before building, the script
copies the head's bench sources and timing code into the base worktree, so
the gate compares library code, never two timing methods. This matters. On
the first run of the gate, rust/fx-var-engine `portfolio_sigma_50f` looked
1.49x slower (4,599 ns against 3,085 ns) with identical library code; timed
with one harness, base and head both gave a median of 2,342 ns on a laptop.

Both sides are also compiled from one staging directory with the same build
directories. Source paths are baked into the executables (Rust panic
locations, C++ `__FILE__`), so identical code built from two different
directories gives differently laid-out binaries, and code alignment alone
made rust/equity-options-engine `crr_price n=1000` look 1.45x slower on a
docs-only change. Staged at one path, identical code builds to identical
bytes (checked for all eight benches).

A kernel fails the job only when both hold:

- the head median is more than 25% above the base median, and
- the bands do not overlap: head p10 is above base p90.

Example: base 100 ns [90, 110], head 150 ns [140, 160]: +50% and 140 > 110,
so it fails. Head 150 ns [105, 400]: +50%, but 105 is inside the base band,
so it is reported as `slower` and does not fail. Head 120 ns [115, 125]:
separate bands, but +20% is under the threshold; reported, not failed. New
and removed kernels are reported only.

The job summary shows the table; the `bench-results` artifact holds every
round's JSON, `summary.md` and `ab.json`. Those artifacts are the source for
current figures. The tables in the engine READMEs are an older laptop run
with the previous harness and say so.

Limits. Three rounds on a shared runner catch large regressions, not 5%
ones. A base whose bench does not build (for example a new engine) is
reported without a gate, and so is a base library that does not build under
the head's harness (the job prints a warning naming the engine). Pinning
to one CPU also serialises the multi-threaded Monte Carlo kernels; both sides
pay that equally.

## Per-call tails: the specification (W12, not yet implemented)

The batch-mean tails above cannot support a latency objective
(`docs/SLO.md`, S6). W12 replaces them for SLOs and the gate with per-call
timing. This section is the contract that implementation must meet; until
it lands, every tail figure in this repository is a batch-mean figure and
is labelled so.

The plain idea. Time every call on its own, with a clock cheap and fine
enough that its own cost is small next to a 100 ns call, and report how
much the clock itself costs so the reader can subtract it in their head.

The method, stated exactly (x86-64; other targets keep the batch method
and say so in `tail_basis`):

1. Check the TSC is invariant (CPUID leaf 0x80000007, EDX bit 8) and read
   its frequency once against `steady_clock` over 100 ms. Without an
   invariant TSC, fall back to the batch method and say so.
2. Each sample is `lfence; rdtscp` before the call and `rdtscp; lfence`
   after (Intel, "How to Benchmark Code Execution Times on Intel IA-32 and
   IA-64 Instruction Set Architectures", 2010), converted to ns.
3. Timer overhead: the same pair around an empty body, 10,000 times; its
   median and p99 are reported as `timer_overhead_p50` and
   `timer_overhead_p99` and are not subtracted.
4. `calls_per_sample` = 1 and `tail_basis` = `per-call`; p50, p90, p99,
   p99.9 and max by nearest rank as above.
5. Coordinated omission: the bench calls back-to-back, so a stall delays
   later calls rather than hiding them; the stall is counted once, in the
   call that suffered it (Tene, 2015). The figures are service times, not
   response times under an arrival process, and are labelled so.

Acceptance (from the plan): a test bench that stalls for 1 ms on one call
in every 500 must show a per-call p99.9 of at least 1 ms, and one that
stalls one call in 100,000 must not. The rate is 1 in 500, not 1 in 1,000,
on purpose: with N samples and exactly N/1,000 stalls, nearest-rank p99.9
is element ceil(0.999 N) - 1, the largest call that did not stall.

Fuzzing is not benchmarking, but the two share CI time on free runners:
the nightly fuzz workflow (`.github/workflows/fuzz.yml`) spends 100 s per
target and runs at 03:17 UTC, away from the bench jobs.
