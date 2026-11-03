// Per-call latency harness shared by the engine benchmarks.
//
// The timing method is the same in the C++ and Rust benches (Rust:
// qp_core::bench, rust/core/src/bench.rs); docs/BENCHMARKS.md states it.
//
//   1. Warm up: `warmup` untimed calls.
//   2. Batch size B: the smallest power of two, at most 1024, for which one
//      batch of B back-to-back calls takes at least 2 us on the monotonic
//      clock (std::chrono::steady_clock here, std::time::Instant in Rust),
//      judged on the fastest of 5 trial batches.
//      Fast kernels are therefore timed as B-call averages, slow kernels
//      one call at a time; B is recorded as `calls_per_sample`.
//   3. N samples, N = max(31, iters / B): each sample is one batch, divided
//      by B to give ns per call. Nothing is subtracted.
//   4. Report the median (p50) with p10 and p90 as the spread, plus p99,
//      p99.9, max and mean of the same per-call samples.
//
// What the tail columns are. When B > 1 every sample is the mean of B
// back-to-back calls, so p99, p99.9 and max are percentiles of B-call
// batch means, not of single calls. Averaging hides a rare slow call: one
// 1 ms stall inside a batch of 1024 adds only about 1 us to that sample.
// They are therefore labelled "p99 of B-call batch means" (stdout header,
// the JSON field `tail_basis`, docs/BENCHMARKS.md) and must not be quoted
// as per-call tail latency or used for an SLO. Per-call tail timing is a
// separate harness (W12, later round). When B == 1 the tails are per call.
//
// Output: one line per kernel on stdout and, with --out PATH, a JSON array
// with fields engine, kernel, label, compiler, flags, cpu, date, iters
// (samples), calls_per_sample, unit, timer, p10, p50, p90, p99, p999, max,
// mean, tail_basis ("per-call" when B == 1, else "p99 of B-call batch
// means" with B written out). Options: --out PATH, --label NAME,
// --quick (10% of the samples),
// --only TEXT (run only kernels whose name contains TEXT).

#ifndef QP_BENCH_LATENCY_HPP
#define QP_BENCH_LATENCY_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

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
    double p10 = 0, p50 = 0, p90 = 0, p99 = 0, p999 = 0, max = 0, mean = 0;
    std::size_t samples = 0, batch = 1;
};

inline constexpr double kMinSampleNs = 2000.0;
inline constexpr std::size_t kMaxBatch = 1024;
inline constexpr std::size_t kMinSamples = 31;
inline constexpr std::size_t kCalibrationTrials = 5;

/// Percentile of sorted samples: element ceil(q * n) - 1 (nearest rank),
/// the same rule as the Rust harness.
inline double nearest_rank(const std::vector<double>& sorted, double q) {
    const std::size_t n = sorted.size();
    std::size_t i = static_cast<std::size_t>(std::ceil(q * static_cast<double>(n)));
    i = std::clamp<std::size_t>(i, 1, n) - 1;
    return sorted[i];
}

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
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_FULL_VER);
#else
    return "unknown";
#endif
}

inline std::string today() {
    const std::time_t t = std::time(nullptr);
    char buf[16];
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::strftime(buf, sizeof buf, "%Y-%m-%d", &tm);
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
        std::printf("engine %s | %s | %s | timer steady_clock, batch to >= 2 us\n", engine_.c_str(),
                    compiler_id().c_str(), cpu_.c_str());
        std::printf("%-36s %6s %9s %9s %9s %9s %11s\n", "kernel (ns per call)", "B", "p10", "p50",
                    "p90", "p99*", "max*");
        std::printf("* p99 and max are of B-call batch means when B > 1, not per-call tails\n");
    }
    Bench(const Bench&) = delete;
    Bench& operator=(const Bench&) = delete;

    bool quick() const { return scale_ < 1.0; }

    /// Time `f(i)` (i counts calls) after `warmup` untimed calls; `iters`
    /// is the call budget, split into samples of B calls (see the header).
    template <class F>
    void run(const std::string& kernel, F&& f, std::size_t warmup, std::size_t iters) {
        if (!only_.empty() && kernel.find(only_) == std::string::npos) return;
        iters = std::max<std::size_t>(
            kMinSamples, static_cast<std::size_t>(static_cast<double>(iters) * scale_));
        const Stats s = run_raw(f, warmup, iters);
        std::printf("%-36s %6zu %9.1f %9.1f %9.1f %9.1f %11.1f\n", kernel.c_str(), s.batch, s.p10,
                    s.p50, s.p90, s.p99, s.max);
        std::fflush(stdout);
        const std::string tail_basis =
            s.batch == 1 ? std::string("per-call")
                         : "p99 of " + std::to_string(s.batch) + "-call batch means";
        char rec[1536];
        std::snprintf(rec, sizeof rec,
                      "  {\"engine\": \"%s\", \"kernel\": \"%s\", \"label\": \"%s\", "
                      "\"compiler\": \"%s\", \"flags\": \"%s\", \"cpu\": \"%s\", "
                      "\"date\": \"%s\", \"iters\": %zu, \"calls_per_sample\": %zu, "
                      "\"unit\": \"ns\", \"timer\": \"steady_clock\", \"p10\": %.2f, "
                      "\"p50\": %.2f, \"p90\": %.2f, \"p99\": %.2f, \"p999\": %.2f, "
                      "\"max\": %.2f, \"mean\": %.2f, \"tail_basis\": \"%s\"}",
                      engine_.c_str(), kernel.c_str(), label_.c_str(), compiler_id().c_str(),
                      QP_BENCH_FLAGS, cpu_.c_str(), today().c_str(), s.samples, s.batch, s.p10,
                      s.p50, s.p90, s.p99, s.p999, s.max, s.mean, tail_basis.c_str());
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
    static Stats run_raw(F&& f, std::size_t warmup, std::size_t iters) {
        using clock = std::chrono::steady_clock;
        std::size_t i = 0;
        for (; i < warmup; ++i) f(i);
        const auto batch_ns = [&](std::size_t b) {
            const auto t0 = clock::now();
            for (std::size_t k = 0; k < b; ++k) f(i++);
            return std::chrono::duration<double, std::nano>(clock::now() - t0).count();
        };
        std::size_t b = 1;
        // The fastest of a few trial batches decides, so one preempted trial
        // cannot stop the doubling early.
        const auto fastest = [&](std::size_t bb) {
            double best = batch_ns(bb);
            for (std::size_t t = 1; t < kCalibrationTrials; ++t)
                best = std::min(best, batch_ns(bb));
            return best;
        };
        while (b < kMaxBatch && fastest(b) < kMinSampleNs) b *= 2;
        const std::size_t n = std::max(kMinSamples, iters / b);
        std::vector<double> ns(n);
        for (double& v : ns) v = batch_ns(b) / static_cast<double>(b);
        Stats s;
        s.samples = n;
        s.batch = b;
        double sum = 0.0;
        for (const double v : ns) sum += v;
        s.mean = sum / static_cast<double>(n);
        std::sort(ns.begin(), ns.end());
        s.p10 = nearest_rank(ns, 0.10);
        s.p50 = nearest_rank(ns, 0.50);
        s.p90 = nearest_rank(ns, 0.90);
        s.p99 = nearest_rank(ns, 0.99);
        s.p999 = nearest_rank(ns, 0.999);
        s.max = ns.back();
        return s;
    }

    std::string engine_, out_, label_ = "current", cpu_, only_;
    double scale_ = 1.0;
    std::vector<std::string> records_;
};
}  // namespace qpbench

#endif  // QP_BENCH_LATENCY_HPP
