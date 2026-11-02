// The noexcept kernel API (fxvar/kernels.hpp, CompiledBook::try_pnl*) must
// reproduce the throwing API bit for bit, and its errors must carry the same
// message text.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "fxvar/book.hpp"
#include "fxvar/expected_shortfall.hpp"
#include "fxvar/kernels.hpp"
#include "fxvar/matrix.hpp"
#include "fxvar/monte_carlo.hpp"
#include "fxvar/parametric.hpp"

using namespace fxvar;

namespace {

static_assert(noexcept(try_empirical_var_es({}, 0.01, {})));
static_assert(noexcept(try_var_standard_error({}, 0.01, {})));
static_assert(noexcept(try_normal_var(1.0)));
static_assert(noexcept(try_normal_es(1.0)));
static_assert(noexcept(try_t_var(1.0)));
static_assert(noexcept(try_t_es(1.0)));
static_assert(noexcept(try_matvec(std::declval<const Matrix&>(), {}, {})));
static_assert(noexcept(try_quad_form({}, std::declval<const Matrix&>())));
static_assert(noexcept(try_portfolio_sigma({}, std::declval<const Matrix&>())));
static_assert(noexcept(try_var_covar({}, std::declval<const Matrix&>())));
static_assert(noexcept(try_cholesky_lower(std::declval<const Matrix&>(), {})));
static_assert(noexcept(try_simulate_factor_returns({}, 0, 0, McDist::kNormal, 6.0, 0, {})));
static_assert(noexcept(std::declval<const CompiledBook&>().try_pnl({})));
static_assert(noexcept(std::declval<const CompiledBook&>().try_pnl_batch({}, {})));
static_assert(std::is_trivially_copyable_v<Error>);

bool same(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

std::vector<double> sample(std::size_t n, std::uint64_t seed) {
  Rng rng(seed);
  std::vector<double> v(n);
  for (auto& x : v) x = 1e5 * rng.normal() + (rng.uniform() < 0.02 ? -4e5 : 0.0);
  return v;
}

Matrix ar1_cov(std::size_t k) {
  Matrix c(k, k);
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j)
      c(i, j) = 3.6e-5 * std::pow(0.5, std::abs(static_cast<double>(i) - static_cast<double>(j)));
  return c;
}

}  // namespace

TEST(LatencyApi, EmpiricalVarEsAndStandardErrorEqualThrowingApi) {
  for (std::size_t n : {10UL, 11UL, 250UL, 1000UL, 10007UL}) {
    const auto pnl = sample(n, n);
    std::vector<double> ws(n);
    for (double alpha : {0.001, 0.01, 0.025, 0.05, 0.3}) {
      const auto r = try_empirical_var_es(pnl, alpha, ws);
      ASSERT_TRUE(r.has_value());
      const auto legacy = empirical_var_es(pnl, alpha);
      EXPECT_TRUE(same(r->first, legacy.first) && same(r->second, legacy.second));
      EXPECT_TRUE(same(*try_var_standard_error(pnl, alpha, ws), var_standard_error(pnl, alpha)));
    }
  }
}

TEST(LatencyApi, ErrorsCarryTheThrowingMessage) {
  std::vector<double> ws(3);
  const std::vector<double> bad{1.0, std::numeric_limits<double>::quiet_NaN(), 2.0};
  const auto r = try_empirical_var_es(bad, 0.01, ws);
  ASSERT_FALSE(r.has_value());
  try {
    (void)empirical_var(bad, 0.01);
    FAIL();
  } catch (const std::invalid_argument& e) {
    EXPECT_EQ(message(r.error()), e.what());
  }
  const auto a = try_normal_var(1.0, 0.7);
  try {
    (void)normal_var(1.0, 0.7);
    FAIL();
  } catch (const std::invalid_argument& e) {
    EXPECT_EQ(message(a.error()), e.what());
  }
  std::vector<double> small(2);
  const std::vector<double> ok{1, 2, 3};
  EXPECT_EQ(try_empirical_var_es(ok, 0.01, small).error().code, Errc::kWorkspaceTooSmall);
}

TEST(LatencyApi, ClosedFormAndLinearAlgebraEqualThrowingApi) {
  for (double sig : {0.0, 1.0, 12345.6})
    for (double alpha : {0.001, 0.01, 0.2}) {
      EXPECT_TRUE(same(*try_normal_var(sig, alpha, 3.0), normal_var(sig, alpha, 3.0)));
      EXPECT_TRUE(same(*try_normal_es(sig, alpha, 3.0), normal_es(sig, alpha, 3.0)));
      EXPECT_TRUE(same(*try_t_var(sig, alpha, 5.0, 3.0), t_var(sig, alpha, 5.0, 3.0)));
      EXPECT_TRUE(same(*try_t_es(sig, alpha, 5.0, 3.0), t_es(sig, alpha, 5.0, 3.0)));
    }
  const Matrix cov = ar1_cov(40);
  std::vector<double> w(40), y(40);
  for (std::size_t i = 0; i < 40; ++i)
    w[i] = (i % 3 ? -1.0 : 1.0) * (1e6 + 1e4 * static_cast<double>(i));
  ASSERT_TRUE(try_matvec(cov, w, y).has_value());
  const auto y0 = matvec(cov, w);
  for (std::size_t i = 0; i < 40; ++i) EXPECT_TRUE(same(y[i], y0[i]));
  EXPECT_TRUE(same(*try_quad_form(w, cov), quad_form(w, cov)));
  EXPECT_TRUE(same(*try_portfolio_sigma(w, cov), portfolio_sigma(w, cov)));
  for (TailDist d : {TailDist::kNormal, TailDist::kStudentT}) {
    const VarEs a = *try_var_covar(w, cov, 0.01, 10.0, d, 5.0, 0.0);
    const VarEs b = var_covar(w, cov, 0.01, 10.0, d, 5.0, 0.0);
    EXPECT_TRUE(same(a.var, b.var) && same(a.es, b.es));
  }
}

TEST(LatencyApi, CholeskyAndSimulationEqualThrowingApi) {
  const std::size_t k = 30;
  FactorCov fc;
  for (std::size_t i = 0; i < k; ++i) fc.factors.push_back("FX:C" + std::to_string(i));
  fc.cov = ar1_cov(k);
  const double h = 10.0;
  Matrix scaled = fc.cov;
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j) scaled(i, j) *= h;
  std::vector<double> lower(k * k);
  ASSERT_TRUE(try_cholesky_lower(scaled, lower).has_value());
  const CholeskyResult ch = cholesky(scaled);
  ASSERT_FALSE(ch.jittered);
  for (std::size_t i = 0; i < k * k; ++i) EXPECT_TRUE(same(lower[i], ch.lower.data()[i]));
  for (McDist d : {McDist::kNormal, McDist::kStudentT}) {
    std::vector<double> out(2000 * k);
    ASSERT_TRUE(try_simulate_factor_returns(lower, k, 2000, d, 5.0, 99, out).has_value());
    const ReturnsMatrix ref = simulate_factor_returns(fc, 2000, d, 5.0, {}, 99, h);
    for (std::size_t i = 0; i < out.size(); ++i) ASSERT_TRUE(same(out[i], ref.data.data()[i])) << i;
  }
  Matrix sing(2, 2, 1.0);
  std::vector<double> l2(4);
  EXPECT_FALSE(try_cholesky_lower(sing, l2).has_value());
}

TEST(BatchApi, BookRevaluationEqualsLegacyBitForBit) {
  const Market m({{"EUR", 1.08}, {"JPY", 0.0067}, {"GBP", 1.27}, {"CHF", 1.12}},
                 {{"USD", 0.05}, {"EUR", 0.03}, {"JPY", 0.001}, {"GBP", 0.045}, {"CHF", 0.01}});
  const Book book({SpotPosition{"EURUSD", 1e7, {}}, SpotPosition{"EURJPY", -5e6, {}},
                   ForwardPosition{"GBPCHF", 3e6, 0.5, {}}, CashPosition{"JPY", 1e8}},
                  "EUR");
  const CompiledBook cb(book, m);
  const std::size_t k = cb.factors().size();
  FactorCov fc{cb.factors(), ar1_cov(k)};
  const ReturnsMatrix scen = simulate_factor_returns(fc, 5000, McDist::kStudentT, 4.0, {}, 3);
  std::vector<double> out(5000);
  ASSERT_TRUE(cb.try_pnl_batch(scen.data.data(), out).has_value());
  const auto legacy = cb.pnl(scen);
  for (std::size_t s = 0; s < out.size(); ++s) {
    ASSERT_TRUE(same(out[s], legacy[s])) << s;
    ASSERT_TRUE(
        same(*cb.try_pnl(std::span<const double>(scen.data.row(s), k)), cb.pnl(scen.data.row(s))));
  }
  std::vector<double> poisoned(scen.data.data());
  poisoned[7] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(cb.try_pnl_batch(poisoned, out).has_value());
  std::vector<double> short_out(4999);
  EXPECT_EQ(cb.try_pnl_batch(scen.data.data(), short_out).error().code, Errc::kSizeMismatch);
}
