// Zero heap allocations on the kernel success path.
//
// This binary replaces the global operator new/delete with counting
// versions, so it is built as its own test executable. Each test warms the
// kernel once (lazy one-time set-up such as libm tables is not the
// question), then counts allocations across many calls. The legacy
// throwing API is checked too where it should not allocate.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <new>
#include <vector>

#include "eqopt/batch.hpp"
#include "eqopt/binomial.hpp"
#include "eqopt/black76.hpp"
#include "eqopt/black_scholes.hpp"
#include "eqopt/greeks.hpp"
#include "eqopt/implied_vol.hpp"
#include "eqopt/monte_carlo.hpp"
#include "eqopt/thread_pool.hpp"

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

using namespace eqopt;

namespace {

template <class F>
long allocations(F&& f) {
    f(0);  // warm-up
    const long before = g_allocs.load();
    for (int i = 1; i < 200; ++i) f(i);
    return g_allocs.load() - before;
}

}  // namespace

TEST(NoAlloc, AnalyticKernels) {
    double acc = 0.0;
    EXPECT_EQ(allocations([&](int i) {
                  const double S = 90.0 + 0.1 * i;
                  acc += *try_bs_price(S, 100, 1, 0.03, 0.2, 0.01, OptionType::Put);
                  acc += try_bs_greeks(S, 100, 1, 0.03, 0.2, 0.01)->vega;
                  acc += *try_black76_price(S, 100, 1, 0.03, 0.2);
                  acc += try_black76_greeks(S, 100, 1, 0.03, 0.2)->gamma;
                  acc += *try_implied_vol(10.0 + 0.01 * i, 100, 100, 1, 0.03, 0.01);
                  // The throwing API allocates only on its error path.
                  acc += bs_price(S, 100, 1, 0.03, 0.2, 0.01);
                  acc += bs_greeks(S, 100, 1, 0.03, 0.2, 0.01).delta;
                  acc += implied_vol(10.0 + 0.01 * i, 100, 100, 1, 0.03, 0.01);
              }),
              0);
    EXPECT_GT(acc, 0.0);
}

TEST(NoAlloc, ErrorPathOfNoexceptApi) {
    // Building an Error never allocates: it holds a static string and numbers.
    long n = allocations([](int) {
        (void)try_bs_price(-1.0, 100, 1, 0, 0.2);
        (void)try_implied_vol(1e9, 100, 100, 1, 0);
    });
    EXPECT_EQ(n, 0);
}

TEST(NoAlloc, CrrWithWorkspace) {
    std::vector<double> ws(crr_workspace_size(500));
    double acc = 0.0;
    EXPECT_EQ(allocations([&](int i) {
                  acc += *try_crr_price(95.0 + 0.05 * i, 100, 1, 0.03, 0.25, 0.01, OptionType::Put,
                                        ExerciseStyle::American, 500, ws);
              }),
              0);
    EXPECT_GT(acc, 0.0);
}

TEST(NoAlloc, BatchKernels) {
    constexpr std::size_t n = 256;
    std::vector<double> S(n), K(n, 100), T(n, 0.75), r(n, 0.02), sig(n, 0.3), q(n, 0.0);
    std::vector<OptionType> ty(n, OptionType::Call);
    for (std::size_t i = 0; i < n; ++i) S[i] = 70.0 + 0.25 * static_cast<double>(i);
    std::vector<double> o[8];
    for (auto& v : o) v.resize(n);
    const BsBatchInputs in{S, K, T, r, sig, q, ty};
    const BsGreeksBatchOutputs out{o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]};
    EXPECT_EQ(allocations([&](int) {
                  ASSERT_TRUE(bs_price_batch(in, o[0]).has_value());
                  ASSERT_TRUE(bs_greeks_batch(in, out).has_value());
              }),
              0);
}

TEST(NoAlloc, MonteCarloWithPoolAndWorkspace) {
    ThreadPool pool(3);
    McWorkspace ws;
    ws.reserve(50000, true);
    double acc = 0.0;
    EXPECT_EQ(allocations([&](int i) {
                  acc += try_mc_price(100, 100, 1, 0.03, 0.2, 0.0, OptionType::Call, 50000, true,
                                      true, static_cast<std::uint64_t>(i), 4, ws, &pool)
                             ->price;
              }),
              0);
    EXPECT_GT(acc, 0.0);
}
