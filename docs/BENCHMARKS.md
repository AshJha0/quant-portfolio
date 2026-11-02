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
   samples): p10, p50, p90, p99, p99.9, plus max and mean.

Each record in the JSON output has `kernel`, `calls_per_sample`, `unit`
(`ns`), `p10`, `p50`, `p90`, `p99`, `p999` and `max`.

Worked example. `bs_price` takes about 120 ns. One call is under 2 us, and
so are batches of 2, 4 and 8; a batch of 16 takes about 1.9 us and one of 32
about 3.8 us, so B = 32 (or 16 on a faster machine). With a budget of
100,000 calls that gives 3,125 samples, each a 32-call average.

Percentiles of B-call averages are narrower than percentiles of single
calls: a rare 10 us stall is spread over B calls. The p99.9 and max of a
fast kernel are therefore not single-call tails. Slow kernels (Monte Carlo,
binomial trees, VaR over long histories) run with B = 1 and their tails are
single calls.

## The CI gate (same runner, base against head)

`.github/scripts/bench_ab_run.sh` checks out the base commit (the pull
request's target, or the previous commit on a push) into a worktree, builds
both commits' C++ (GCC 15, Release) and Rust (release) benches, and runs
them in three alternating rounds: base, head, base, head, base, head. Every
run is pinned to CPU 1. `.github/scripts/bench_ab.py` then takes, per kernel
and side, the median over the rounds of p10, p50 and p90.

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
reported without a gate. Benches from a base older than this harness have no
p10/p90; their band is read as the median alone, so for that one comparison
the gate fails when head p10 is above the base median, a looser test. Pinning
to one CPU also serialises the multi-threaded Monte Carlo kernels; both sides
pay that equally.
