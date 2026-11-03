// Time-stamp counter reads for per-call latency.
//
// start(): lfence; rdtsc; lfence   -- earlier instructions have completed
//                                     and later ones have not started.
// stop():  rdtscp; lfence          -- rdtscp waits for earlier instructions;
//                                     the lfence keeps later ones from
//                                     starting before the read.
// This is the pattern in Intel's "How to Benchmark Code Execution Times on
// Intel IA-32 and IA-64 Instruction Set Architectures" (Paoloni 2010), with
// lfence in place of cpuid (cheaper, and serialising for this purpose on
// Intel and on AMD when lfence is dispatch-serialising).
//
// Ticks are converted to nanoseconds with a frequency calibrated against
// std::chrono::steady_clock. That is only meaningful with an invariant TSC
// (cpuinfo flags constant_tsc and nonstop_tsc), which sysinfo.hpp reads back.
// Non-x86 builds fall back to steady_clock (ticks are then nanoseconds).
#pragma once

#include <chrono>
#include <cstdint>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#define ULL_HAVE_TSC 1
#else
#define ULL_HAVE_TSC 0
#endif

namespace ull::bench {

inline std::uint64_t tsc_start() noexcept {
#if ULL_HAVE_TSC
    _mm_lfence();
    const std::uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
#else
    return static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

inline std::uint64_t tsc_stop() noexcept {
#if ULL_HAVE_TSC
    unsigned aux = 0;
    const std::uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
#else
    return static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

// Ticks per nanosecond, measured over `ms` milliseconds of wall time.
inline double calibrate_ticks_per_ns(int ms = 100) {
#if ULL_HAVE_TSC
    using clk = std::chrono::steady_clock;
    const auto w0 = clk::now();
    const std::uint64_t t0 = tsc_start();
    while (clk::now() - w0 < std::chrono::milliseconds(ms)) {
    }
    const std::uint64_t t1 = tsc_stop();
    const auto w1 = clk::now();
    const double ns = std::chrono::duration<double, std::nano>(w1 - w0).count();
    return static_cast<double>(t1 - t0) / ns;
#else
    (void)ms;
    return 1.0;
#endif
}

// Busy-wait for `ticks` TSC ticks (used for stall injection).
inline void spin_ticks(std::uint64_t ticks) noexcept {
    const std::uint64_t t0 = tsc_start();
    while (tsc_start() - t0 < ticks) {
    }
}

}  // namespace ull::bench
