// Empirical VaR/ES (hand-exact), closed-form normal/t identities.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "fxvar/expected_shortfall.hpp"
#include "fxvar/monte_carlo.hpp"
#include "fxvar/stats.hpp"

using namespace fxvar;

TEST(EmpiricalVar, HandExactQuantiles) {
  // 10 P&Ls; losses sorted desc: 50, 40, 30, 20, 10, 0, -5, -15, -25, -35.
  const std::vector<double> pnl{-50, -40, -30, -20, -10, 0, 5, 15, 25, 35};
  // alpha=0.90: m = ceil(10*0.1) = 1 -> worst loss.
  EXPECT_DOUBLE_EQ(empirical_var(pnl, 0.90), 50.0);
  // alpha=0.80: m = 2 -> second-worst loss.
  EXPECT_DOUBLE_EQ(empirical_var(pnl, 0.80), 40.0);
  // alpha=0.75: 10*(0.25) = 2.5 -> third-worst loss.
  EXPECT_DOUBLE_EQ(empirical_var(pnl, 0.75), 30.0);
  // alpha=0.99 with only 10 obs: still the worst loss.
  EXPECT_DOUBLE_EQ(empirical_var(pnl, 0.99), 50.0);
}

TEST(EmpiricalEs, AcerbiTascheTailSplitting) {
  const std::vector<double> pnl{-50, -40, -30, -20, -10, 0, 5, 15, 25, 35};
  // alpha=0.80: exactly the 2 worst losses averaged.
  EXPECT_DOUBLE_EQ(empirical_es(pnl, 0.80), 45.0);
  // alpha=0.75: tail mass 0.25 = 0.1 + 0.1 + 0.05: fractional atom share
  // ES = (0.1*50 + 0.1*40 + 0.05*30) / 0.25 = 42.
  EXPECT_DOUBLE_EQ(empirical_es(pnl, 0.75), 42.0);
  // ES >= VaR always.
  for (double a : {0.5, 0.75, 0.9, 0.99}) {
    const auto [v, e] = empirical_var_es(pnl, a);
    EXPECT_GE(e, v);
  }
}

TEST(EmpiricalVar, WeightedScenarios) {
  // Weights emphasise the recent (last) scenario: BRW-style.
  const std::vector<double> pnl{-100.0, -10.0, 0.0, -50.0};
  const std::vector<double> w{0.05, 0.15, 0.30, 0.50};
  // Losses desc: 100 (w .05), 50 (w .50), 10 (w .15), 0 (w .30).
  // alpha=0.90 -> target 0.10: cum 0.05 < 0.10 at loss 100, then 0.55 at
  // loss 50 -> VaR = 50.
  EXPECT_DOUBLE_EQ(empirical_var(pnl, 0.90, w), 50.0);
  // ES = (0.05*100 + 0.05*50)/0.10 = 75.
  EXPECT_DOUBLE_EQ(empirical_es(pnl, 0.90, w), 75.0);
}

TEST(EmpiricalVar, InputValidation) {
  EXPECT_THROW(empirical_var({}, 0.99), std::invalid_argument);
  EXPECT_THROW(empirical_var({1.0, std::nan("")}, 0.99), std::invalid_argument);
  EXPECT_THROW(empirical_var({1.0, 2.0}, 1.0), std::invalid_argument);
  EXPECT_THROW(empirical_var({1.0, 2.0}, 0.0), std::invalid_argument);
  EXPECT_THROW(empirical_var({1.0, 2.0}, 0.99, {0.5}), std::invalid_argument);
  EXPECT_THROW(empirical_var({1.0, 2.0}, 0.99, {-0.1, 1.1}),
               std::invalid_argument);
  EXPECT_THROW(empirical_var({1.0, 2.0}, 0.99, {0.0, 0.0}),
               std::invalid_argument);
}

TEST(ClosedForm, NormalVarEsIdentity) {
  // ES = sigma * phi(z_a)/(1-a); check the identity to 1e-10 and the
  // textbook 99% numbers on sigma = 1.
  const double sigma = 2.5e6, alpha = 0.99, mu = 1.0e4;
  const double z = norm_ppf(alpha);
  EXPECT_NEAR(normal_var(sigma, alpha, mu), -mu + sigma * z, 1e-10);
  EXPECT_NEAR(normal_es(sigma, alpha, mu),
              -mu + sigma * norm_pdf(z) / (1.0 - alpha), 1e-10);
  // sigma=1, alpha=0.99: VaR = 2.3263478740, ES = 2.6652142306 (classic).
  EXPECT_NEAR(normal_var(1.0, 0.99), 2.3263478740408408, 1e-9);
  EXPECT_NEAR(normal_es(1.0, 0.99), 2.665214220345808, 1e-9);
  EXPECT_GT(normal_es(1.0, 0.95), normal_var(1.0, 0.95));
}

TEST(ClosedForm, StudentTFatterThanNormalAtEqualSigma) {
  // Standardised t: same sigma, fatter 99% tail; df -> inf recovers normal.
  const double sigma = 1.0;
  EXPECT_GT(t_var(sigma, 0.99, 4.0), normal_var(sigma, 0.99));
  EXPECT_GT(t_es(sigma, 0.99, 4.0), normal_es(sigma, 0.99));
  EXPECT_NEAR(t_var(sigma, 0.99, 1e7), normal_var(sigma, 0.99), 1e-5);
  EXPECT_THROW(t_var(sigma, 0.99, 2.0), std::invalid_argument);
  EXPECT_THROW(normal_var(-1.0, 0.99), std::invalid_argument);
}

TEST(ClosedForm, EmpiricalConvergesToNormalIdentity) {
  // Deterministic equiprobable normal grid (inverse-CDF stratification):
  // empirical ES on the grid approaches the closed form.
  const int n = 200000;
  std::vector<double> pnl(n);
  for (int i = 0; i < n; ++i)
    pnl[i] = norm_ppf((i + 0.5) / static_cast<double>(n));
  const auto [v, e] = empirical_var_es(pnl, 0.975);
  EXPECT_NEAR(v, normal_var(1.0, 0.975), 2e-4);
  EXPECT_NEAR(e, normal_es(1.0, 0.975), 2e-3);
}


TEST(EmpiricalVar, SingleAndTwoElementSamples) {
  // A one-scenario sample is degenerate but must not read out of bounds:
  // both VaR and ES collapse onto that scenario's loss for every alpha.
  const std::vector<double> one{-250.0};
  for (const double a : {0.5, 0.95, 0.99, 0.999}) {
    EXPECT_DOUBLE_EQ(empirical_var(one, a), 250.0);
    EXPECT_DOUBLE_EQ(empirical_es(one, a), 250.0);
  }
  // A profit-only single scenario reports a negative "loss" (a gain floor).
  EXPECT_DOUBLE_EQ(empirical_var(std::vector<double>{40.0}, 0.99), -40.0);
  // Two scenarios: the 99% level picks the worse one; ES >= VaR always.
  const std::vector<double> two{-100.0, 20.0};
  EXPECT_DOUBLE_EQ(empirical_var(two, 0.99), 100.0);
  EXPECT_DOUBLE_EQ(empirical_es(two, 0.99), 100.0);
  EXPECT_DOUBLE_EQ(empirical_var(two, 0.5), 100.0);
  EXPECT_GE(empirical_es(two, 0.5), empirical_var(two, 0.5));
  // ... and the standard-error machinery refuses samples this small.
  EXPECT_THROW(var_standard_error(one, 0.99), std::invalid_argument);
  EXPECT_THROW(var_standard_error(two, 0.99), std::invalid_argument);
}
