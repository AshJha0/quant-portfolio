// Pins the shared VaR-engine conventions on the FX engine: alpha as the
// tail probability, order-statistic VaR / fractional ES, Kupiec and Basel on
// any window, Acerbi-Szekely Z2, alpha-aware Cornish-Fisher validity,
// capped Cholesky with PSD repair, EWMA seed, Poisson jumps over the
// horizon and the bootstrap SE. Expected values are computed by hand or by
// an independent route in each test.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "fxvar/backtest.hpp"
#include "fxvar/expected_shortfall.hpp"
#include "fxvar/matrix.hpp"
#include "fxvar/monte_carlo.hpp"
#include "fxvar/parametric.hpp"
#include "fxvar/returns.hpp"
#include "fxvar/stats.hpp"

using namespace fxvar;

TEST(Alpha, IsTheTailProbability) {
  // 0.99 used to mean 99 % VaR; it is now an invalid tail probability.
  EXPECT_THROW(normal_var(1.0, 0.99), std::invalid_argument);
  EXPECT_NEAR(normal_var(1.0, 0.01), 2.3263478740408408, 1e-9);
  EXPECT_THROW(empirical_var({1.0, 2.0, 3.0}, 0.99), std::invalid_argument);
}

TEST(OrderStatistic, FractionalAtomByHand) {
  std::vector<double> pnl;
  for (int i = 1; i <= 10; ++i) pnl.push_back(-static_cast<double>(i));
  // k = 2.5, m = 3: VaR = 8, ES = (10 + 9 + 0.5 * 8) / 2.5 = 9.2.
  const auto [v, e] = empirical_var_es(pnl, 0.25);
  EXPECT_EQ(v, 8.0);
  EXPECT_NEAR(e, 9.2, 1e-15);
  const auto [vw, ew] = empirical_var_es(pnl, 0.25, std::vector<double>(10, 7.0));
  EXPECT_EQ(vw, 8.0);
  EXPECT_NEAR(ew, 9.2, 1e-14);
}

TEST(Kupiec, StructInputAndTailAlpha) {
  const LrTest r = kupiec_pof({.n_obs = 250, .n_exceptions = 5, .alpha = 0.01});
  const double lr = -2.0 * (245.0 * std::log(0.99) + 5.0 * std::log(0.01) -
                            245.0 * std::log(0.98) - 5.0 * std::log(0.02));
  EXPECT_NEAR(r.lr, lr, 1e-12);
  EXPECT_THROW(kupiec_pof({.n_obs = 250, .n_exceptions = 251, .alpha = 0.01}),
               std::invalid_argument);
  EXPECT_THROW(kupiec_pof({.n_obs = 250, .n_exceptions = 5, .alpha = 0.99}),
               std::invalid_argument);
}

TEST(Basel, PlusFactorMappedThrough250DayTable) {
  // n = 1000, x = 15: F = 0.9521 -> yellow, x_eq = 5 -> +0.40. The old code
  // took the add-on from the raw count (15 > 9 -> 0.85).
  const TrafficLight y = basel_traffic_light(BaselInput{.n_obs = 1000, .n_exceptions = 15});
  EXPECT_EQ(y.zone, Zone::kYellow);
  EXPECT_DOUBLE_EQ(y.plus_factor, 0.40);
  EXPECT_DOUBLE_EQ(y.multiplier, 3.40);
  // n = 500, x = 9: F = 0.9689 -> x_eq = 6 -> +0.50.
  EXPECT_DOUBLE_EQ(basel_traffic_light(BaselInput{.n_obs = 500, .n_exceptions = 9}).plus_factor,
                   0.50);
  // n = 250, alpha = 0.025, x = 12: F(12; 250, 0.025) = 0.9616 -> yellow.
  const TrafficLight a25 =
      basel_traffic_light(BaselInput{.n_obs = 250, .n_exceptions = 12, .alpha = 0.025});
  EXPECT_EQ(a25.zone, Zone::kYellow);
  const double table[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.40, 0.50, 0.65, 0.75, 0.85, 1.0};
  for (int x = 0; x <= 10; ++x)
    EXPECT_DOUBLE_EQ(basel_traffic_light(BaselInput{.n_obs = 250, .n_exceptions = x}).plus_factor,
                     table[x])
        << x;
}

TEST(AcerbiSzekely, Z2ByHand) {
  const std::vector<double> pnl = {0.5, -3.0, 0.2, -1.5};
  // Z2 = (-3 - 1.5) / (4 * 0.25 * 2) + 1 = -1.25.
  EXPECT_NEAR(acerbi_szekely_z2(pnl, {1.0}, {2.0}, 0.25), -1.25, 1e-15);
  EXPECT_THROW(acerbi_szekely_z2(pnl, {1.0}, {0.0}, 0.25), std::invalid_argument);
}

TEST(CornishFisher, NonFiniteMomentsThrow) {
  // The old predicate returned false and cornish_fisher_var with the check
  // off returned NaN.
  EXPECT_THROW(cornish_fisher_valid(std::nan(""), 0.0, 0.01), std::invalid_argument);
  EXPECT_THROW(cornish_fisher_var(1.0, std::nan(""), 0.0, 0.01, 0.0, 1.0, false),
               std::invalid_argument);
  EXPECT_TRUE(cornish_fisher_valid(1.0, 1.0, 0.01));
  EXPECT_FALSE(cornish_fisher_valid(1.0, 1.0, 0.001));
}

TEST(Cholesky, JitterIsCappedAt1e6MeanDiag) {
  // lambda_min = -5e-6: the old ladder (no cap, up to 1e-5) factored it
  // silently; above the 1e-6 x mean(diag) cap it now throws.
  const Matrix a = Matrix::from_rows({{1.0, 1.0 + 5e-6}, {1.0 + 5e-6, 1.0}});
  EXPECT_THROW(cholesky(a), std::runtime_error);
  const CholeskyResult c = cholesky(a, Repair::kClip);
  EXPECT_TRUE(c.repaired);
  // Clip keeps the unit variances.
  EXPECT_NEAR(c.lower(0, 0), 1.0, 1e-12);
}

TEST(Cholesky, HighamReferenceCase) {
  const Matrix a = Matrix::from_rows({{1.0, 0.9, 0.9}, {0.9, 1.0, -0.9}, {0.9, -0.9, 1.0}});
  EXPECT_THROW(cholesky(a), std::runtime_error);
  const Matrix h = repair_covariance(a, Repair::kHigham);
  EXPECT_NEAR(h(0, 1), 0.5, 1e-8);  // numpy eigh reference
  EXPECT_NEAR(h(0, 2), 0.5, 1e-8);
  EXPECT_NEAR(h(1, 2), -0.5, 1e-8);
  const EigenResult e = symmetric_eigen(a);
  EXPECT_NEAR(e.values[0], -0.8, 1e-14);
  EXPECT_NEAR(e.values[2], 1.9, 1e-14);
}

TEST(Ewma, SeedIsMeanSquareOverFirst30) {
  ReturnsMatrix r;
  r.factors = {"FX:EUR"};
  r.data = Matrix(40, 1);
  double s = 0.0;
  for (int i = 0; i < 40; ++i) {
    r.data(i, 0) = 0.001 * (i + 1);
    if (i < 30) s += r.data(i, 0) * r.data(i, 0);
  }
  const EwmaVolatility ev = ewma_volatility(r, 0.94);
  // Old seed: full-sample ddof=1 variance (uses rows 30..39: look-ahead).
  EXPECT_NEAR(ev.sigma(0, 0), std::sqrt(s / 30.0), 1e-15);
}

TEST(Jumps, PoissonCountScalesWithHorizon) {
  // Per-day p = 0.05 over h = 10 days: P(at least one jump) = 1 - 0.95^10
  // = 0.4013. The old Bernoulli(p) overlay ignored the horizon (0.05).
  FactorCov cov;
  cov.factors = {"FX:TRY"};
  cov.cov = Matrix::from_rows({{1e-14}});
  JumpSpec j;
  j.prob = 0.05;
  j.mean = {{"FX:TRY", -0.1}};
  const std::size_t n = 200000;
  const ReturnsMatrix sc = simulate_factor_returns(cov, n, McDist::kJump, 6.0, j, 9, 10.0);
  std::size_t hit = 0;
  double total = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    hit += sc.data(i, 0) < -0.05 ? 1 : 0;
    total += sc.data(i, 0);
  }
  const double p_hit = static_cast<double>(hit) / static_cast<double>(n);
  const double expect = 1.0 - std::pow(0.95, 10.0);
  // Binomial SE = sqrt(p (1-p) / n) = 0.0011; 5 SE.
  EXPECT_NEAR(p_hit, expect, 5.0 * std::sqrt(expect * (1 - expect) / n));
  // Mean jump return = -0.1 * lam h, lam = -ln(0.95). Poisson SD of the
  // count sqrt(lam h) = 0.716 -> SE of the mean 0.1 * 0.716 / sqrt(n) = 1.6e-4.
  const double lam_h = -std::log(0.95) * 10.0;
  EXPECT_NEAR(total / static_cast<double>(n), -0.1 * lam_h, 5.0 * 0.1 * std::sqrt(lam_h / n));
  j.prob = 1.0;
  EXPECT_THROW(simulate_factor_returns(cov, 10, McDist::kJump, 6.0, j, 9), std::invalid_argument);
}

TEST(MonteCarloSe, BootstrapPrimaryKdeDiagnostic) {
  FactorCov cov;
  cov.factors = {"FX:EUR"};
  cov.cov = Matrix::from_rows({{1.0}});
  const ReturnsMatrix sc = simulate_factor_returns(cov, 100000, McDist::kNormal, 6.0, {}, 11);
  std::vector<double> pnl(sc.data.data().begin(), sc.data.data().end());
  const double a = 0.01;
  const double se_true = std::sqrt(a * (1 - a) / 1e5) / norm_pdf(norm_ppf(1 - a));
  // 400 replicates: relative sampling error ~ 3.5 %; 20 % also covers the
  // O(n^-1/2) bias of the quantile bootstrap.
  EXPECT_NEAR(var_standard_error_bootstrap(pnl, a, 400, 3), se_true, 0.20 * se_true);
  EXPECT_NEAR(var_standard_error(pnl, a), se_true, 0.10 * se_true);
  EXPECT_THROW(var_standard_error_bootstrap(pnl, a, 1, 3), std::invalid_argument);
}

TEST(LogGamma, LanczosMatchesClosedForms) {
  EXPECT_NEAR(log_gamma(0.5), 0.5 * std::log(3.14159265358979323846), 1e-14);
  EXPECT_NEAR(log_gamma(10.0), std::log(362880.0), 1e-13);
  EXPECT_THROW(log_gamma(0.0), std::invalid_argument);
}

TEST(EmptyBook, Throws) {
  const Market m({{"EUR", 1.1}});
  EXPECT_THROW(CompiledBook(Book{}, m), std::invalid_argument);
  EXPECT_THROW(portfolio_sigma({0.0, 0.0}, Matrix::identity(2)), std::invalid_argument);
}
