// Zero heap allocations on the kernel success path.
//
// This binary replaces the global operator new/delete with counting
// versions, so it is its own test executable. Each check warms the kernel
// once, then counts allocations across repeated calls with caller-owned
// buffers.

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <new>
#include <vector>

#include "fxvar/book.hpp"
#include "fxvar/kernels.hpp"

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

using namespace fxvar;

namespace {

template <class F>
long allocations(F&& f) {
  f(0);
  const long before = g_allocs.load();
  for (int i = 1; i < 100; ++i) f(i);
  return g_allocs.load() - before;
}

}  // namespace

TEST(NoAlloc, VarEsAndClosedForms) {
  std::vector<double> pnl(10000), ws(10000);
  for (std::size_t i = 0; i < pnl.size(); ++i)
    pnl[i] = std::sin(0.37 * static_cast<double>(i)) * 1e5;
  double acc = 0.0;
  EXPECT_EQ(allocations([&](int i) {
              pnl[static_cast<std::size_t>(i)] += 1.0;  // vary the input
              acc += try_empirical_var_es(pnl, 0.01, ws)->second;
              acc += *try_var_standard_error(pnl, 0.01, ws);
              acc += *try_normal_var(1e5 + i, 0.01) + *try_normal_es(1e5, 0.01);
              acc += *try_t_var(1e5 + i, 0.01, 5.0) + *try_t_es(1e5, 0.01, 5.0);
            }),
            0);
  EXPECT_GT(acc, 0.0);
}

TEST(NoAlloc, LinearAlgebraSimulationAndBook) {
  const std::size_t k = 25;
  Matrix cov(k, k);
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j)
      cov(i, j) = 3.6e-5 * std::pow(0.5, std::abs(static_cast<double>(i) - static_cast<double>(j)));
  std::vector<double> w(k, 1e6), y(k), lower(k * k), scen(1000 * k);

  const Market m({{"EUR", 1.08}, {"JPY", 0.0067}, {"GBP", 1.27}},
                 {{"USD", 0.05}, {"EUR", 0.03}, {"GBP", 0.045}, {"JPY", 0.001}});
  const Book book({SpotPosition{"EURUSD", 1e7, {}}, ForwardPosition{"GBPJPY", 3e6, 0.5, {}}});
  const CompiledBook cb(book, m);
  const std::size_t nf = cb.factors().size();
  std::vector<double> bscen(1000 * nf), out(1000);
  std::vector<double> l2(nf * nf);
  Matrix c2(nf, nf);
  for (std::size_t i = 0; i < nf; ++i) c2(i, i) = 3.6e-5;
  ASSERT_TRUE(try_cholesky_lower(c2, l2).has_value());

  double acc = 0.0;
  EXPECT_EQ(allocations([&](int i) {
              w[0] = 1e6 + i;
              acc += try_matvec(cov, w, y).has_value() ? y[1] : 0.0;
              acc += *try_portfolio_sigma(w, cov);
              acc += try_var_covar(w, cov, 0.01, 10.0)->es;
              acc += try_cholesky_lower(cov, lower).has_value() ? lower[k + 1] : 0.0;
              acc += try_simulate_factor_returns(lower, k, 1000, McDist::kStudentT, 5.0,
                                                 static_cast<std::uint64_t>(i), scen)
                             .has_value()
                         ? scen[3]
                         : 0.0;
              (void)try_simulate_factor_returns(l2, nf, 1000, McDist::kNormal, 5.0,
                                                static_cast<std::uint64_t>(i), bscen);
              acc += cb.try_pnl_batch(bscen, out).has_value() ? out[5] : 0.0;
              acc += *cb.try_pnl(std::span<const double>(bscen.data(), nf));
            }),
            0);
  EXPECT_NE(acc, 0.0);
}
