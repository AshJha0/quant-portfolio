// Throughput benchmarks for the FX options engine.
//
//   1. 1,000,000 Garman-Kohlhagen vanilla prices (varying strike/vol so the
//      compiler cannot hoist the computation).
//   2. Strike-from-delta solves across the four quoting conventions
//      (analytic for spot/forward, Brent for premium-adjusted).
//   3. One 1,000,000-path Monte Carlo price (antithetic + control variate,
//      mt19937_64, single-threaded).
//
// Build: cmake --build build -j && ./build/fxopt_bench

#include <chrono>
#include <cstdio>

#include "fxopt/deltas.hpp"
#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/monte_carlo.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

}  // namespace

int main() {
    using namespace fxopt;
    const double S = 1.10, T = 0.5, rd = 0.0425, rf = 0.0290;

    // ---- 1M GK prices ---------------------------------------------------
    {
        constexpr int n = 1'000'000;
        double acc = 0.0;
        const auto t0 = Clock::now();
        for (int i = 0; i < n; ++i) {
            const double K = 0.90 + 0.40 * (i % 1000) / 1000.0;
            const double sig = 0.06 + 0.10 * (i % 97) / 97.0;
            acc += gk_price(S, K, T, rd, rf, sig,
                            i % 2 == 0 ? OptionType::Call : OptionType::Put);
        }
        const double dt = seconds_since(t0);
        std::printf("GK vanilla pricing : %d prices in %.3f s -> "
                    "%.2f M prices/s (checksum %.6f)\n",
                    n, dt, n / dt / 1e6, acc);
    }

    // ---- strike-from-delta solves ---------------------------------------
    {
        constexpr int n_per_conv = 25'000;
        const DeltaConvention convs[] = {
            DeltaConvention::Spot, DeltaConvention::Forward,
            DeltaConvention::SpotPa, DeltaConvention::ForwardPa};
        const char* names[] = {"spot", "forward", "spot_pa", "forward_pa"};
        for (int c = 0; c < 4; ++c) {
            double acc = 0.0;
            const auto t0 = Clock::now();
            for (int i = 0; i < n_per_conv; ++i) {
                const double d = 0.05 + 0.40 * (i % 100) / 100.0;
                const double sig = 0.07 + 0.08 * (i % 53) / 53.0;
                acc += strike_from_delta(d, S, T, rd, rf, sig,
                                         OptionType::Call, convs[c]);
                acc += strike_from_delta(-d, S, T, rd, rf, sig,
                                         OptionType::Put, convs[c]);
            }
            const double dt = seconds_since(t0);
            std::printf("strike-from-delta  : %-10s %d solves in %.3f s -> "
                        "%.2f k solves/s (checksum %.4f)\n",
                        names[c], 2 * n_per_conv, dt,
                        2 * n_per_conv / dt / 1e3, acc);
        }
    }

    // ---- 1M-path Monte Carlo --------------------------------------------
    {
        constexpr std::int64_t n_paths = 1'000'000;
        const auto t0 = Clock::now();
        const MCResult r = mc_price(S, 1.12, T, rd, rf, 0.0925,
                                    OptionType::Call, n_paths, 0);
        const double dt = seconds_since(t0);
        std::printf("Monte Carlo        : %lld paths in %.3f s -> "
                    "%.2f M paths/s (price %.8f, SE %.2e, %s)\n",
                    static_cast<long long>(n_paths), dt, n_paths / dt / 1e6,
                    r.price, r.std_error, r.method.c_str());
    }
    return 0;
}
