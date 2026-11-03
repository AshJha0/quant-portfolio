// Harness tests: histogram accuracy against exact percentiles, and stall
// injection showing up at the right rate (service) and with the queueing
// effect (response) that coordinated omission would hide.
#include "../../qpbf/generator.hpp"
#include "../harness.hpp"
#include "../histogram.hpp"
#include "../sysinfo.hpp"
#include "../ull_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace ull::bench;

static void histogram_index_roundtrip() {
    // Every value lies inside its bucket, and buckets are contiguous.
    bool ok = true;
    for (std::uint64_t v : {0ull, 1ull, 2047ull, 2048ull, 2049ull, 4095ull, 4096ull, 123456789ull,
                            (1ull << 40) + 7, ~0ull}) {
        const auto i = Histogram::index_of(v);
        const auto hi = Histogram::highest_equivalent(i);
        const auto lo = i == 0 ? 0 : Histogram::highest_equivalent(i - 1) + 1;
        ok &= (lo <= v && v <= hi);
        ok &= (hi - lo) <= std::max<std::uint64_t>(1, v / 1024);  // <= 0.1 % wide
    }
    ULL_CHECK(ok);
    ULL_CHECK(Histogram::index_of(~0ull) == Histogram::kBuckets - 1);
}

static void histogram_matches_exact_percentiles() {
    ull::qpbf::Xoshiro256ss r(3);
    Histogram h;
    std::vector<std::uint64_t> v;
    for (int i = 0; i < 200000; ++i) {
        // Log-uniform over 1 .. 2^30 ticks.
        const auto x = static_cast<std::uint64_t>(std::exp2(30.0 * (r() >> 11) * 0x1.0p-53));
        v.push_back(x);
        h.record(x);
    }
    std::sort(v.begin(), v.end());
    double worst = 0;
    for (double p : {50.0, 90.0, 99.0, 99.9, 99.99}) {
        const auto k = static_cast<std::size_t>(std::ceil(p / 100.0 * v.size())) - 1;
        const double exact = static_cast<double>(v[k]);
        const double got = static_cast<double>(h.value_at_percentile(p));
        ULL_CHECK(got >= exact);  // highest-equivalent never under-reports
        worst = std::max(worst, (got - exact) / exact);
    }
    std::printf("histogram worst relative error vs exact: %.5f\n", worst);
    ULL_CHECK(worst <= 1.0 / 1024 + 1e-12);
    ULL_CHECK(h.max() == v.back() && h.min() == v.front());
}

static void stall_injection() {
    const double tpn = calibrate_ticks_per_ns(50);
    const auto ns = [&](double x) { return static_cast<std::uint64_t>(x * tpn); };
    volatile std::uint64_t sink = 0;
    auto op = [&](std::uint64_t i) { sink = sink + i; };

    RunConfig c;
    c.calls = 400'000;
    c.warmup = 1000;
    c.interval_ticks = ns(2'000);   // 2 us
    c.stall_every = 2000;           // 0.05 % of calls
    c.stall_ticks = ns(1'000'000);  // 1 ms
    Histogram svc, rsp;
    run_open_loop(op, c, svc, rsp);
    const std::uint64_t stalls = c.calls / c.stall_every;
    const std::uint64_t svc_slow = svc.count_at_or_above(ns(900'000));
    const double rsp_slow_frac = static_cast<double>(rsp.count_at_or_above(ns(100'000))) / c.calls;
    std::printf(
        "stall test: %llu stalls injected; service >= 0.9 ms: %llu; "
        "service p99.9 %.1f us p99.99 %.1f us; response p99 %.1f us, >= 0.1 ms: %.1f %%\n",
        static_cast<unsigned long long>(stalls), static_cast<unsigned long long>(svc_slow),
        svc.value_at_percentile(99.9) / tpn / 1e3, svc.value_at_percentile(99.99) / tpn / 1e3,
        rsp.value_at_percentile(99.0) / tpn / 1e3, 100 * rsp_slow_frac);
    // Service time sees each stall once: the exact injected count, plus at
    // most a few OS preemptions on a shared machine.
    ULL_CHECK(svc_slow >= stalls);
    ULL_CHECK(svc_slow <= stalls + stalls / 10 + 5);
    ULL_CHECK(svc.value_at_percentile(99.99) >= ns(1'000'000));
    ULL_CHECK(svc.value_at_percentile(99.9) < ns(900'000));
    // Response time sees the queue behind each stall: call j after the
    // stall waits D - j T, so j <= (D - 0.1 ms) / T = 450 calls of every
    // 2000 wait >= 0.1 ms: 22.5 % from the injected stalls alone. Any OS
    // preemption on a shared machine adds more queueing, never less, so
    // only a lower bound is checked (a loaded
    // laptop has measured 25-93 %).
    ULL_CHECK(rsp_slow_frac > 0.18);
    ULL_CHECK(rsp_slow_frac > 100.0 * svc_slow / c.calls);  // CO: response >> service

    // Same stall at 0.2 %: now it lands inside p99.9 of service time too.
    c.stall_every = 500;
    c.calls = 100'000;
    svc.reset();
    rsp.reset();
    run_open_loop(op, c, svc, rsp);
    ULL_CHECK(svc.value_at_percentile(99.9) >= ns(1'000'000));
    ULL_CHECK(svc.value_at_percentile(99.0) < ns(900'000));
}

static void sysinfo_reads_back() {
    const SysInfo s = read_sysinfo();
    std::printf("sysinfo: %s\n", s.to_json().c_str());
#if defined(__linux__)
    ULL_CHECK(s.cpu_model != "unavailable");
    ULL_CHECK(s.affinity != "unavailable" && !s.affinity.empty());
    ULL_CHECK(s.online_cpus > 0);
#endif
}

int main() {
    histogram_index_roundtrip();
    histogram_matches_exact_percentiles();
    sysinfo_reads_back();
    stall_injection();
    ULL_TEST_MAIN_END();
}
