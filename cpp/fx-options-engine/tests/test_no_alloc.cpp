// Zero heap allocations on the kernel success path.
//
// This binary replaces the global operator new/delete with counting
// versions, so it is built as its own test executable. Each test warms the
// kernel once, then counts allocations across many calls.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <new>
#include <vector>

#include "fxopt/batch.hpp"
#include "fxopt/binomial.hpp"
#include "fxopt/black76.hpp"
#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/greeks.hpp"
#include "fxopt/implied_vol.hpp"
#include "fxopt/monte_carlo.hpp"
#include "fxopt/thread_pool.hpp"

namespace {
std::atomic<long> g_allocs{0};
}  // namespace

// GCC flags malloc/free inside replacement operators as mismatched once it
// inlines them into new-expressions; the pairing is correct by construction.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

void* operator new(std::size_t n) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t n) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using namespace fxopt;

namespace {

template <class F>
long allocations(F&& f) {
    f(0);  // warm-up
    const long before = g_allocs.load();
    for (int i = 1; i < 200; ++i) f(i);
    return g_allocs.load() - before;
}

constexpr OptionType kCall = OptionType::Call;
constexpr OptionType kPut = OptionType::Put;

}  // namespace

TEST(NoAlloc, AnalyticKernels) {
    double acc = 0.0;
    EXPECT_EQ(allocations([&](int i) {
                  const double S = 1.05 + 0.001 * i;
                  acc += *try_gk_price(S, 1.1, 1, 0.03, 0.01, 0.1, kPut);
                  acc += try_analytic_greeks(S, 1.1, 1, 0.03, 0.01, 0.1, kCall)->vega;
                  acc += *try_black76_price(S, 1.1, 1, 0.03, 0.1, kCall);
                  acc += *try_implied_vol(0.04 + 0.0001 * i, 1.1, 1.1, 1, 0.03, 0.01, kCall);
                  // The throwing API allocates only on its error path.
                  acc += gk_price(S, 1.1, 1, 0.03, 0.01, 0.1, kCall);
                  acc += analytic_greeks(S, 1.1, 1, 0.03, 0.01, 0.1, kCall).delta_spot;
                  acc += implied_vol(0.04 + 0.0001 * i, 1.1, 1.1, 1, 0.03, 0.01, kCall);
              }),
              0);
    EXPECT_GT(acc, 0.0);
}

TEST(NoAlloc, ErrorPathOfNoexceptApi) {
    EXPECT_EQ(allocations([](int) {
                  (void)try_gk_price(-1.0, 1.0, 1, 0, 0, 0.2, kCall);
                  (void)try_implied_vol(1e9, 1.0, 1.0, 1, 0, 0, kCall);
              }),
              0);
}

TEST(NoAlloc, BinomialWithWorkspace) {
    std::vector<double> ws(binomial_workspace_size(500));
    double acc = 0.0;
    EXPECT_EQ(allocations([&](int i) {
                  acc += *try_binomial_price(1.05 + 0.001 * i, 1.1, 1, 0.03, 0.01, 0.15, kPut,
                                             500, Exercise::American, ws);
              }),
              0);
    EXPECT_GT(acc, 0.0);
}

TEST(NoAlloc, BatchKernels) {
    constexpr std::size_t n = 256;
    std::vector<double> S(n), K(n, 1.1), T(n, 0.75), rd(n, 0.02), rf(n, 0.01), sig(n, 0.12);
    std::vector<OptionType> ty(n, kCall);
    for (std::size_t i = 0; i < n; ++i) S[i] = 0.9 + 0.001 * static_cast<double>(i);
    std::vector<double> o[10];
    for (auto& v : o) v.resize(n);
    const GkBatchInputs in{S, K, T, rd, rf, sig, ty};
    const GkGreeksBatchOutputs out{o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9]};
    EXPECT_EQ(allocations([&](int) {
                  ASSERT_TRUE(gk_price_batch(in, o[0]).has_value());
                  ASSERT_TRUE(gk_greeks_batch(in, out).has_value());
              }),
              0);
}

TEST(NoAlloc, MonteCarloWithPoolAndWorkspace) {
    ThreadPool pool(3);
    McWorkspace ws;
    ws.reserve(50000, true);
    double acc = 0.0;
    EXPECT_EQ(allocations([&](int i) {
                  acc += try_mc_price(1.1, 1.1, 1, 0.03, 0.01, 0.1, kCall, 50000,
                                      Seed{static_cast<std::uint64_t>(i)}, true, true, 4, ws,
                                      &pool)
                             ->price;
                  acc += try_mc_price(1.1, 1.1, 1, 0.03, 0.01, 0.1, kCall, 50000,
                                      Seed{static_cast<std::uint64_t>(i)}, true, true, 1, ws,
                                      nullptr)
                             ->price;
              }),
              0);
    EXPECT_GT(acc, 0.0);
}
