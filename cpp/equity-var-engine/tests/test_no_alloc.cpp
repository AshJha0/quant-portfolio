// Zero heap allocations on the kernel success path.
//
// This binary replaces the global operator new/delete with counting
// versions, so it is its own test executable. Each test warms the kernel
// once, then counts allocations across many calls.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <new>
#include <vector>

#include "eqvar/kernels.hpp"

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

using namespace eqvar;

namespace {

template <class F>
long allocations(F&& f) {
    f(0);  // warm-up
    const long before = g_allocs.load();
    for (int i = 1; i < 50; ++i) f(i);
    return g_allocs.load() - before;
}

Matrix cov_of(std::size_t n) {
    Matrix c(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) c(i, j) = 1e-4 * (i == j ? 1.0 : 0.3);
    return c;
}

}  // namespace

TEST(NoAlloc, HistoricalAndParametric) {
    std::vector<double> pnl(1000), ws(1000), w(20, 1e5), y(20);
    for (std::size_t i = 0; i < pnl.size(); ++i) {
        pnl[i] = static_cast<double>((i * 7919) % 1000) - 480.0;
    }
    const Matrix cov = cov_of(20);
    std::vector<double> l(400), scratch(400);
    double acc = 0.0;
    EXPECT_EQ(allocations([&](int i) {
                  const double a = 0.01 + 0.001 * i;
                  acc += try_var_es_order_statistic(pnl, a, ws)->es;
                  acc += *try_historical_var(pnl, a, ws);
                  acc += *try_expected_shortfall(pnl, a, ws);
                  acc += *try_parametric_var(w, cov, a, Dist::StudentT, 5.0);
                  acc += *try_cholesky_lower(cov, l, scratch);
                  (void)try_matvec(cov, w, y);
                  acc += y[0];
                  // The error path does not allocate either.
                  (void)try_validate_alpha(0.9);
              }),
              0);
    EXPECT_NE(acc, 0.0);
}

TEST(NoAlloc, MonteCarloWithWorkspace) {
    const std::size_t n = 10, paths = 2000;
    const Matrix cov = cov_of(n);
    std::vector<double> w(n, 1e5), l(n * n), scratch(n * n);
    ASSERT_TRUE(try_cholesky_lower(cov, l, scratch).has_value());
    McVarWorkspace ws;
    ws.reserve(n, paths);
    double acc = 0.0;
    EXPECT_EQ(allocations([&](int i) {
                  acc += try_monte_carlo_var_factored(w, l, 0.01, paths, Dist::StudentT, 6.0,
                                                      static_cast<std::uint64_t>(i), 20, ws)
                             ->var;
              }),
              0);
    EXPECT_GT(acc, 0.0);
}
