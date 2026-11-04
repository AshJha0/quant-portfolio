# W30-rs latency methodology

**Software in-process latency, not wire-to-wire.** No figure here includes a
NIC, a kernel network stack, a switch or a wire. Kernel-bypass NICs, PTP
hardware and co-location are out of scope (plan section 2).

## What is measured

Plain idea: time one call at a time, keep every sample, and report the tail.
A mean hides the slow calls a trading system cares about, and a batch average
(the old `thread_pool.hpp` timings) hides them completely.

| Bench | One sample is | Notes |
|---|---|---|
| `timer_overhead` | an empty timed region | subtract mentally from every other row; it is not subtracted in the output |
| `qpbf_decode_msg` | one `decode_msg` call | packet already validated and in cache |
| `decode_plus_book_apply` | decode one message and apply it to the L3 book | the tick-to-book step |
| `book_apply_fixed_rate_co_aware` | apply at a fixed offered rate, `end - intended start` | coordinated-omission aware |
| `book_apply_fixed_rate_naive` | the same run, `end - actual start` | what a closed loop reports |
| `spsc_ring_round_trip` | push to thread B, B echoes on a second ring, pop | two cross-core hops |

## How it is measured

- **Timer.** x86-64: start = `lfence; rdtsc; lfence`, end = `rdtscp; lfence`
  (Paoloni, Intel white paper 324264, 2010, for the fencing discipline).
  Ticks are converted with a ratio calibrated against `std::time::Instant`
  over 200 ms. Other targets use `Instant`. The output names the timer.
- **Histogram.** HdrHistogram layout (Gil Tene): exact below 2,048 ns, then
  1,024 sub-buckets per power of two, so relative error is under 0.1%.
  Percentiles report the bucket's highest equivalent value, as HdrHistogram's
  `getValueAtPercentile` does. Allocated once; recording never allocates.
- **Coordinated omission.** Tene, "How NOT to Measure Latency" (2015). Call
  `i` is due at `t0 + i * period`; a late call is issued at once and its wait
  is charged to it. Formal statement and worked example (1 ms stall every
  10,000 calls at 10 us spacing gives CO-aware p99.9 = 0.9 ms while the naive
  p99.9 shows only the service time) are in `src/lib.rs` and are checked
  exactly on a simulated clock by `stall_shows_at_the_right_rate_sim`.
- **Saturation flag.** If the CO-aware median exceeds ten times the naive
  p99, the offered rate is above the host's capacity and the CO figures
  measure the backlog, not the code. The output says so
  (`fixed_rate_saturated`).
- **Environment readback.** CPU model, `Cpus_allowed_list`, transparent huge
  pages, `HugePages_Total`, cpufreq governor and isolated CPUs are read from
  `/proc` and `/sys` at run time, never taken from what the operator meant to
  set. On non-Linux hosts each says "unavailable" with the reason. Pinning is
  applied from outside (`taskset -c`), and the readback shows whether it took.
- **Host label.** `--host-class` defaults to "shared-VM indicative". Hosted CI
  runners are shared VMs; their figures are indicative only and are always
  printed with the CPU model.

## Like-for-like with the C++ lane

`ull-bench --write-day day.qpbf` writes the generated day as a QPBF
recording (`docs/QPBF.md` section 9), and `ull-bench --input day.qpbf` runs
on any recording, including one written by the C++ `ull_replay_publisher
generate`. Running both harnesses on
the same file, on the same host, in the same session, with the same
`--rate-mps` and the same pinning is the comparison the plan asks for. The two
lanes' generators are independent, so comparing generated days without a
shared file is not like-for-like.

## Recorded run (2026-11-03, development laptop)

Windows 11, Intel family 6 model 186 (12 logical CPUs), unpinned, not
isolated, release build, 2,000,000 generated messages, seed 20261103.
Indicative only: Windows does not expose the readback fields and the
scheduler adds millisecond-scale preemption.

| Bench | p50 ns | p99 ns | p99.9 ns | p99.99 ns |
|---|---|---|---|---|
| timer_overhead | 27 | 34 | 38 | 145 |
| qpbf_decode_msg | 61 | 86 | 176 | 347 |
| decode_plus_book_apply | 155 | 337 | 579 | 7,227 |
| book_apply fixed 0.2 M/s, CO-aware | 596 | 313,343 | 1,484,799 | 2,439,167 |
| book_apply fixed 0.2 M/s, naive | 387 | 939 | 3,879 | 29,407 |
| spsc_ring_round_trip | 518 | 1,228 | 3,051 | 132,223 |

Idle-machine run (2026-11-04, WSL2, pinned to CPUs 2 and 4, same message count):

| Bench | p50 ns | p99 ns | p99.9 ns | p99.99 ns |
|---|---|---|---|---|
| timer_overhead | 14 | 16 | 17 | 18 |
| qpbf_decode_msg | 30 | 68 | 185 | 368 |
| decode_plus_book_apply | 56 | 192 | 325 | 4,475 |
| book_apply fixed 0.2 M/s, CO-aware | 175 | 5,847 | 1,392,639 | 3,602,431 |
| book_apply fixed 0.2 M/s, naive | 121 | 364 | 3,783 | 14,631 |
| spsc_ring_round_trip | 184 | 234 | 270 | 11,103 |

Reading the first table: the naive and CO-aware rows come from one run. The
naive p99 of about 1 us says the code is fast; the CO-aware p99 of about
0.3 ms and p99.9 of about 1.5 ms say that on this host about 1% of the calls
offered at 200,000 per second were delayed by a few hundred microseconds or
more because something (here, the OS scheduler)
stopped the loop. Both are true; only the second is what a caller would see.
The same host's uninjected CO-aware p99.9 at 100,000 calls per second was
4.7 ms in the `stall_real_clock` run, which is why that test checks only a
lower bound on real clocks.

Run-to-run spread on this host is large: re-running the same day from the
recording file straight afterwards gave a `qpbf_decode_msg` p50 of 43 ns
against 61 ns above. Single runs on an unpinned laptop are not evidence of a
difference smaller than that; a like-for-like Rust/C++ comparison should
interleave several runs of each on the same recording.
