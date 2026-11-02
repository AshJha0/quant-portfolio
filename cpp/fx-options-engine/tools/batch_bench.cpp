// FX batch throughput: gk_price_batch / gk_greeks_batch (bit-exact) against
// the _fast variants (vectorised) on the same 1024 quotes.
//
// Method: each run prices the batch `reps` times and divides the elapsed
// std::chrono::steady_clock time by reps * 1024. The program does 21 runs
// and prints the median with the min and max, so run-to-run noise is
// visible. Build with the Release preset; run on a quiet machine.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "fxopt/batch.hpp"

namespace {

using fxopt::OptionType;
volatile double g_sink = 0.0;

template <class F>
void measure(const char* name, F&& f, std::size_t n) {
    constexpr int kRuns = 21;
    constexpr int kReps = 2000;
    std::vector<double> ns(kRuns);
    for (int i = 0; i < 50; ++i) f();  // warm-up
    for (int run = 0; run < kRuns; ++run) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < kReps; ++k) f();
        const auto t1 = std::chrono::steady_clock::now();
        ns[run] = std::chrono::duration<double, std::nano>(t1 - t0).count() /
                  (static_cast<double>(kReps) * static_cast<double>(n));
    }
    std::sort(ns.begin(), ns.end());
    std::printf("%-22s median %6.2f ns/option  (min %6.2f, max %6.2f, %d runs)\n", name,
                ns[kRuns / 2], ns.front(), ns.back(), kRuns);
}

}  // namespace

int main() {
    constexpr std::size_t n = 1024;
    std::mt19937_64 g(7);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> S(n), K(n, 1.10), T(n), rd(n), rf(n), sig(n), out(n);
    std::vector<OptionType> ty(n);
    for (std::size_t i = 0; i < n; ++i) {
        S[i] = 1.10 * std::exp(0.2 * u(g) - 0.1);
        T[i] = 0.02 + 2.0 * u(g);
        rd[i] = 0.05 * u(g);
        rf[i] = 0.04 * u(g);
        sig[i] = 0.04 + 0.2 * u(g);
        ty[i] = u(g) < 0.5 ? OptionType::Call : OptionType::Put;
    }
    const fxopt::GkBatchInputs in{S, K, T, rd, rf, sig, ty};
    measure(
        "scalar try_gk_price",
        [&] {
            for (std::size_t i = 0; i < n; ++i)
                out[i] = fxopt::try_gk_price(S[i], K[i], T[i], rd[i], rf[i], sig[i], ty[i])
                             .value_or(0.0);
            g_sink = g_sink + out[n / 2];
        },
        n);
    measure(
        "gk_price_batch",
        [&] {
            (void)fxopt::gk_price_batch(in, out);
            g_sink = g_sink + out[n / 2];
        },
        n);
    measure(
        "gk_price_batch_fast",
        [&] {
            (void)fxopt::gk_price_batch_fast(in, out);
            g_sink = g_sink + out[n / 2];
        },
        n);
    std::vector<double> o[10];
    for (auto& x : o) x.resize(n);
    const fxopt::GkGreeksBatchOutputs go{o[0], o[1], o[2], o[3], o[4],
                                         o[5], o[6], o[7], o[8], o[9]};
    measure(
        "gk_greeks_batch",
        [&] {
            (void)fxopt::gk_greeks_batch(in, go);
            g_sink = g_sink + o[1][n / 2];
        },
        n);
    measure(
        "gk_greeks_batch_fast",
        [&] {
            (void)fxopt::gk_greeks_batch_fast(in, go);
            g_sink = g_sink + o[1][n / 2];
        },
        n);
    return 0;
}
