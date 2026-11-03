// Per-call latency harness, open loop, coordinated-omission aware.
//
// Plain idea. A closed-loop benchmark ("call, time it, call again") hides
// stalls: while one call is stuck for 1 ms, the calls that a real feed would
// have sent during that millisecond are simply never issued, so they never
// record the wait. Gil Tene calls this coordinated omission ("How NOT to
// Measure Latency", 2013-2016 talks; HdrHistogram documentation). The fix is
// an open-loop schedule: call i is *intended* to start at t0 + i * interval,
// whether or not call i - 1 has finished. We then record two numbers per
// call:
//
//   service  = end - actual start      (what the code took)
//   response = end - intended start    (what a sender at that rate saw)
//
// Formal statement. With interval T, a single stall of length D >> T at call
// k delays calls k+1 .. k + floor(D / T) by D - (j - k) T each, so response
// time has about D / T samples above 0 for that one event, while service
// time has exactly one. Recording only service time understates every
// percentile above 1 - 1/n_stalls_per_window.
//
// Worked example (tests/test_bench.cpp): T = 2 us, D = 1 ms, one stall per
// 2000 calls. Service: 1 in 2000 = 0.05 % of samples are >= 1 ms, so p99.9
// stays small and p99.99 is about 1 ms. Response: about 500 of every 2000
// calls wait behind the stall (25 %), so p99 is several hundred
// microseconds.
#pragma once

#include "histogram.hpp"
#include "tsc.hpp"

#include <cstdint>

namespace ull::bench {

struct RunConfig {
    std::uint64_t calls{1'000'000};
    std::uint64_t warmup{10'000};
    std::uint64_t interval_ticks{0};  // 0 = closed loop (back to back)
    std::uint64_t stall_every{0};     // inject a stall on every k-th call (0 = never)
    std::uint64_t stall_ticks{0};
};

struct RunResult {
    std::uint64_t calls{0};
    std::uint64_t late_starts{0};  // calls that started after their intended time
    double elapsed_ticks{0};
};

// op(i) is the call under test. Histograms are in TSC ticks.
template <class Op>
RunResult run_open_loop(Op&& op, const RunConfig& c, Histogram& service, Histogram& response) {
    for (std::uint64_t i = 0; i < c.warmup; ++i) op(i);
    RunResult r;
    const std::uint64_t t0 = tsc_start();
    std::uint64_t intended = t0;
    for (std::uint64_t i = 0; i < c.calls; ++i) {
        std::uint64_t s = tsc_start();
        if (c.interval_ticks != 0) {
            while (s < intended) s = tsc_start();
            if (s > intended + c.interval_ticks) ++r.late_starts;
        } else {
            intended = s;
        }
        op(c.warmup + i);
        if (c.stall_every != 0 && (i % c.stall_every) == c.stall_every - 1) spin_ticks(c.stall_ticks);
        const std::uint64_t e = tsc_stop();
        service.record(e - s);
        response.record(e - intended);
        intended += c.interval_ticks;
    }
    r.calls = c.calls;
    r.elapsed_ticks = static_cast<double>(tsc_stop() - t0);
    return r;
}

}  // namespace ull::bench
