// Micro-benchmark for the eqopt engine: prices/sec for analytic
// Black-Scholes, one 1000-step CRR tree (European + American), and a
// 1M-path Monte Carlo (single-threaded and 4-thread).
//
// Methodology: simple std::chrono steady_clock timing around hot loops; a
// volatile accumulator prevents the optimiser from deleting the work.
// Numbers are indicative (single run, shared machine) — rerun locally with
//   ./build/eqopt_bench

#include <chrono>
#include <cstdio>
#include <vector>

#include "eqopt/binomial.hpp"
#include "eqopt/black_scholes.hpp"
#include "eqopt/greeks.hpp"
#include "eqopt/monte_carlo.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

volatile double g_sink = 0.0;  // defeat dead-code elimination

void bench_black_scholes() {
    constexpr int n = 1'000'000;
    // Pre-generate varied inputs so the loop measures pricing, not caching.
    std::vector<double> spots(n), vols(n);
    for (int i = 0; i < n; ++i) {
        spots[i] = 80.0 + 0.00004 * i;   // 80 .. 120
        vols[i] = 0.10 + 0.0000003 * i;  // 0.10 .. 0.40
    }
    const auto t0 = Clock::now();
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += eqopt::bs_price(spots[i], 100.0, 1.0, 0.05, vols[i], 0.02,
                               eqopt::OptionType::Call);
    }
    const double dt = seconds_since(t0);
    g_sink = acc;
    std::printf("BS analytic        : %d prices in %7.1f ms  -> %10.0f prices/sec\n",
                n, dt * 1e3, n / dt);

    const auto t1 = Clock::now();
    for (int i = 0; i < n / 10; ++i) {
        const auto g = eqopt::bs_greeks(spots[i], 100.0, 1.0, 0.05, vols[i],
                                        0.02, eqopt::OptionType::Call);
        acc += g.delta + g.vega;
    }
    const double dt1 = seconds_since(t1);
    g_sink = acc;
    std::printf("BS full greeks     : %d evals  in %7.1f ms  -> %10.0f evals/sec\n",
                n / 10, dt1 * 1e3, (n / 10) / dt1);
}

void bench_binomial() {
    constexpr int n_steps = 1000;
    constexpr int reps = 20;
    const auto t0 = Clock::now();
    double acc = 0.0;
    for (int i = 0; i < reps; ++i) {
        acc += eqopt::crr_price(100.0 + i * 1e-6, 100.0, 1.0, 0.05, 0.2, 0.02,
                                eqopt::OptionType::Call,
                                eqopt::ExerciseStyle::European, n_steps);
    }
    const double dt = seconds_since(t0) / reps;
    g_sink = acc;
    std::printf("CRR tree, n=1000 EU: 1 tree   in %7.2f ms  -> %10.1f trees/sec\n",
                dt * 1e3, 1.0 / dt);

    const auto t1 = Clock::now();
    for (int i = 0; i < reps; ++i) {
        acc += eqopt::crr_price(100.0 + i * 1e-6, 100.0, 1.0, 0.05, 0.2, 0.02,
                                eqopt::OptionType::Put,
                                eqopt::ExerciseStyle::American, n_steps);
    }
    const double dt1 = seconds_since(t1) / reps;
    g_sink = acc;
    std::printf("CRR tree, n=1000 AM: 1 tree   in %7.2f ms  -> %10.1f trees/sec\n",
                dt1 * 1e3, 1.0 / dt1);
}

void bench_monte_carlo() {
    constexpr std::int64_t n_paths = 1'000'000;
    for (unsigned threads : {1u, 4u}) {
        const auto t0 = Clock::now();
        const auto r = eqopt::mc_price(100.0, 105.0, 1.0, 0.04, 0.25, 0.015,
                                       eqopt::OptionType::Call, n_paths, true,
                                       true, 42, threads);
        const double dt = seconds_since(t0);
        g_sink = r.price;
        std::printf(
            "MC 1M paths (%u thr): price %.4f +/- %.4f in %7.1f ms  -> %10.0f paths/sec\n",
            threads, r.price, r.std_error, dt * 1e3,
            static_cast<double>(r.n_paths) / dt);
    }
}

}  // namespace

int main() {
    std::printf("eqopt micro-benchmark (g++ -O2, single run)\n");
    std::printf("-------------------------------------------\n");
    bench_black_scholes();
    bench_binomial();
    bench_monte_carlo();
    return 0;
}
