// Per-call latency harness shared by the engine benchmarks.
//
// Each kernel is called `iters` times after `warmup` untimed calls. Every
// call is timed on its own with the time-stamp counter (x86-64: lfence,
// rdtscp, lfence; calibrated against std::chrono::steady_clock at start-up)
// or with steady_clock elsewhere. The cost of an empty timed call is
// measured the same way and its median is subtracted from every sample, so
// the reported numbers are the kernel cost alone (clamped at zero). Samples
// go into a buffer allocated before the timed loop and are sorted afterwards
// for the percentiles.
//
// Output: one line per kernel on stdout and, with --out PATH, a JSON array
// with fields engine, kernel, label, compiler, flags, cpu, date, iters,
// unit, timer, overhead_ns, p50, p90, p99, p999, max, mean.
// Options: --out PATH, --label NAME (e.g. before/after), --quick (10% iters),
// --only TEXT (run only kernels whose name contains TEXT).

#ifndef QP_BENCH_LATENCY_HPP
#define QP_BENCH_LATENCY_HPP

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <x86intrin.h>
#define QP_BENCH_TSC 1
#endif

#ifndef QP_BENCH_FLAGS
#define QP_BENCH_FLAGS "-O3 -DNDEBUG"
#endif

namespace qpbench {

inline volatile double g_sink = 0.0;  // defeats dead-code elimination

/// Deterministic input generator (splitmix64), uniform in [lo, hi).
class InputGen {
public:
    explicit InputGen(std::uint64_t seed) : state_(seed) {}
    double uniform(double lo, double hi) {
        state_ += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        return lo + (hi - lo) * (static_cast<double>(z >> 11) * 0x1.0p-53);
    }

private:
    std::uint64_t state_;
};

struct Stats {
    double p50 = 0, p90 = 0, p99 = 0, p999 = 0, max = 0, mean = 0;
};

class Timer {
public:
    Timer() {
#ifdef QP_BENCH_TSC
        const auto c0 = std::chrono::steady_clock::now();
        const std::uint64_t t0 = now();
        std::chrono::steady_clock::time_point c1;
        do {
            c1 = std::chrono::steady_clock::now();
        } while (c1 - c0 < std::chrono::milliseconds(50));
        const std::uint64_t t1 = now();
        ns_per_tick_ = std::chrono::duration<double, std::nano>(c1 - c0).count() /
                       static_cast<double>(t1 - t0);
#endif
    }
    static std::uint64_t now() {
#ifdef QP_BENCH_TSC
        unsigned aux;
        _mm_lfence();
        const std::uint64_t t = __rdtscp(&aux);
        _mm_lfence();
        return t;
#else
        return static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
#endif
    }
    double to_ns(double ticks) const { return ticks * ns_per_tick_; }
    static const char* name() {
#ifdef QP_BENCH_TSC
        return "rdtscp calibrated vs steady_clock";
#else
        return "steady_clock";
#endif
    }

private:
#ifdef QP_BENCH_TSC
    double ns_per_tick_ = 1.0;
#else
    double ns_per_tick_ =
        1e9 * static_cast<double>(std::chrono::steady_clock::period::num) /
        static_cast<double>(std::chrono::steady_clock::period::den);
#endif
};

inline std::string read_cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto pos = line.find(':');
            if (pos != std::string::npos) return line.substr(pos + 2);
        }
    }
    return "unknown";
}

inline std::string compiler_id() {
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#else
    return "unknown";
#endif
}

inline std::string today() {
    const std::time_t t = std::time(nullptr);
    char buf[16];
    std::strftime(buf, sizeof buf, "%Y-%m-%d", std::gmtime(&t));
    return buf;
}

class Bench {
public:
    Bench(int argc, char** argv, std::string engine) : engine_(std::move(engine)) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
                out_ = argv[++i];
            } else if (std::strcmp(argv[i], "--label") == 0 && i + 1 < argc) {
                label_ = argv[++i];
            } else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
                only_ = argv[++i];
            } else if (std::strcmp(argv[i], "--quick") == 0) {
                scale_ = 0.1;
            }
        }
        cpu_ = read_cpu_model();
        overhead_ = run_raw([](std::size_t) {}, 10000, 200000).p50;
        std::printf("engine %s | %s | %s | timer %s | empty-call overhead %.1f ns "
                    "(subtracted)\n",
                    engine_.c_str(), compiler_id().c_str(), cpu_.c_str(), Timer::name(),
                    overhead_);
        std::printf("%-36s %9s %9s %9s %9s %11s %11s\n", "kernel (ns per call)", "p50",
                    "p90", "p99", "p99.9", "max", "mean");
    }
    Bench(const Bench&) = delete;
    Bench& operator=(const Bench&) = delete;

    bool quick() const { return scale_ < 1.0; }

    /// Time `f(i)` for i in [0, iters) after `warmup` untimed calls.
    template <class F>
    void run(const std::string& kernel, F&& f, std::size_t warmup, std::size_t iters) {
        if (!only_.empty() && kernel.find(only_) == std::string::npos) return;
        iters = std::max<std::size_t>(
            20, static_cast<std::size_t>(static_cast<double>(iters) * scale_));
        const Stats s = run_raw(f, warmup, iters, overhead_);
        std::printf("%-36s %9.1f %9.1f %9.1f %9.1f %11.1f %11.1f\n", kernel.c_str(),
                    s.p50, s.p90, s.p99, s.p999, s.max, s.mean);
        std::fflush(stdout);
        char rec[1536];
        std::snprintf(rec, sizeof rec,
                      "  {\"engine\": \"%s\", \"kernel\": \"%s\", \"label\": \"%s\", "
                      "\"compiler\": \"%s\", \"flags\": \"%s\", \"cpu\": \"%s\", "
                      "\"date\": \"%s\", \"iters\": %zu, \"unit\": \"ns\", "
                      "\"timer\": \"%s\", \"overhead_ns\": %.2f, \"p50\": %.2f, "
                      "\"p90\": %.2f, \"p99\": %.2f, \"p999\": %.2f, \"max\": %.2f, "
                      "\"mean\": %.2f}",
                      engine_.c_str(), kernel.c_str(), label_.c_str(),
                      compiler_id().c_str(), QP_BENCH_FLAGS, cpu_.c_str(),
                      today().c_str(), iters, Timer::name(), overhead_, s.p50, s.p90,
                      s.p99, s.p999, s.max, s.mean);
        records_.emplace_back(rec);
    }

    ~Bench() {
        if (out_.empty()) return;
        std::ofstream f(out_);
        f << "[\n";
        for (std::size_t i = 0; i < records_.size(); ++i) {
            f << records_[i] << (i + 1 < records_.size() ? ",\n" : "\n");
        }
        f << "]\n";
    }

private:
    template <class F>
    Stats run_raw(F&& f, std::size_t warmup, std::size_t iters, double subtract = 0.0) {
        for (std::size_t i = 0; i < warmup; ++i) f(i);
        std::vector<double> ns(iters);
        for (std::size_t i = 0; i < iters; ++i) {
            const std::uint64_t t0 = Timer::now();
            f(i);
            const std::uint64_t t1 = Timer::now();
            ns[i] = std::max(0.0, timer_.to_ns(static_cast<double>(t1 - t0)) - subtract);
        }
        Stats s;
        double sum = 0.0;
        for (const double v : ns) sum += v;
        s.mean = sum / static_cast<double>(iters);
        std::sort(ns.begin(), ns.end());
        auto pct = [&](double p) {
            return ns[static_cast<std::size_t>(p * static_cast<double>(iters - 1))];
        };
        s.p50 = pct(0.50);
        s.p90 = pct(0.90);
        s.p99 = pct(0.99);
        s.p999 = pct(0.999);
        s.max = ns.back();
        return s;
    }

    std::string engine_, out_, label_ = "current", cpu_, only_;
    double scale_ = 1.0;
    double overhead_ = 0.0;
    Timer timer_;
    std::vector<std::string> records_;
};

}  // namespace qpbench

#endif  // QP_BENCH_LATENCY_HPP
