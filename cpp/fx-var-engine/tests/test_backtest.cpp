// Backtesting: Kupiec / Christoffersen hand values, chi2 p spot checks,
// exact Basel traffic-light boundaries.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "fxvar/backtest.hpp"
#include "fxvar/stats.hpp"

using namespace fxvar;

TEST(Kupiec, HandComputedLikelihoodRatio) {
  // x=5, n=250, alpha=0.99 -> p=0.01, pi_hat=0.02:
  // LR = -2[ (245 ln .99 + 5 ln .01) - (245 ln .98 + 5 ln .02) ].
  const double ll0 = 245.0 * std::log(0.99) + 5.0 * std::log(0.01);
  const double ll1 = 245.0 * std::log(0.98) + 5.0 * std::log(0.02);
  const double lr_hand = -2.0 * (ll0 - ll1);
  const LrTest t = kupiec_pof(5, 250, 0.99);
  EXPECT_NEAR(t.lr, lr_hand, 1e-12);
  EXPECT_NEAR(t.p, chi2_sf(lr_hand, 1.0), 1e-14);
  // Exactly the expected count: LR = 0, p = 1 (250 * 0.01 = 2.5 is not
  // integral, so use n=100, x=1 at alpha=0.99).
  const LrTest exact = kupiec_pof(1, 100, 0.99);
  EXPECT_NEAR(exact.lr, 0.0, 1e-12);
  EXPECT_NEAR(exact.p, 1.0, 1e-12);
  // Zero exceptions on a long window is evidence of over-conservatism.
  const LrTest zero = kupiec_pof(0, 1000, 0.99);
  EXPECT_NEAR(zero.lr, -2.0 * 1000.0 * std::log(0.99), 1e-9);
  EXPECT_THROW(kupiec_pof(-1, 250, 0.99), std::invalid_argument);
  EXPECT_THROW(kupiec_pof(5, 0, 0.99), std::invalid_argument);
}

TEST(Christoffersen, HandComputedMarkovTest) {
  // Series 0,0,1,1,0,0,1,1,0,0 (n=10, 9 transitions):
  // n00=3, n01=2, n10=2, n11=2; pi01=2/5, pi11=1/2, pi=4/9.
  const std::vector<int> e{0, 0, 1, 1, 0, 0, 1, 1, 0, 0};
  const double pi01 = 2.0 / 5.0, pi11 = 0.5, pi = 4.0 / 9.0;
  const double ll0 = 5.0 * std::log(1.0 - pi) + 4.0 * std::log(pi);
  const double ll1 = 3.0 * std::log(1.0 - pi01) + 2.0 * std::log(pi01) +
                     2.0 * std::log(1.0 - pi11) + 2.0 * std::log(pi11);
  const double lr_hand = -2.0 * (ll0 - ll1);
  const LrTest t = christoffersen_independence(e);
  EXPECT_NEAR(t.lr, lr_hand, 1e-12);
  EXPECT_NEAR(t.p, chi2_sf(lr_hand, 1.0), 1e-14);
  // A clustered series scores higher LR than an alternating one at equal
  // exception count.
  const std::vector<int> clustered{1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const std::vector<int> spread{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
  EXPECT_GT(christoffersen_independence(clustered).lr,
            christoffersen_independence(spread).lr);
  // Degenerate all-zero series: LR = 0.
  EXPECT_NEAR(christoffersen_independence({0, 0, 0, 0}).lr, 0.0, 1e-15);
  EXPECT_THROW(christoffersen_independence({1}), std::invalid_argument);
  EXPECT_THROW(christoffersen_independence({0, 2, 0}), std::invalid_argument);
}

TEST(ConditionalCoverage, SumsComponents) {
  const std::vector<int> e{0, 0, 1, 1, 0, 0, 1, 1, 0, 0};
  const LrTest uc = kupiec_pof(4, 10, 0.99);
  const LrTest ind = christoffersen_independence(e);
  const LrTest cc = conditional_coverage(e, 0.99);
  EXPECT_NEAR(cc.lr, uc.lr + ind.lr, 1e-12);
  EXPECT_NEAR(cc.p, chi2_sf(cc.lr, 2.0), 1e-14);
}

TEST(Chi2, SurvivalFunctionSpotChecks) {
  // scipy.stats.chi2.sf reference values.
  EXPECT_NEAR(chi2_sf(2.706, 1.0), 0.09997137812525883, 1e-10);
  EXPECT_NEAR(chi2_sf(6.635, 1.0), 0.009999419574042536, 1e-10);
  EXPECT_NEAR(chi2_sf(9.210, 2.0), 0.01000170200470548, 1e-10);
}

TEST(Basel, ExactRegulatoryBoundaries) {
  // 250-day, 99% window: green 0-4, yellow 5-9, red >= 10 - exact zone
  // boundaries from the cumulative binomial, not a lookup table.
  for (int x = 0; x <= 4; ++x)
    EXPECT_EQ(basel_traffic_light(x).zone, Zone::kGreen) << "x=" << x;
  for (int x = 5; x <= 9; ++x)
    EXPECT_EQ(basel_traffic_light(x).zone, Zone::kYellow) << "x=" << x;
  for (int x : {10, 11, 15, 25})
    EXPECT_EQ(basel_traffic_light(x).zone, Zone::kRed) << "x=" << x;
  // 1996 table multipliers.
  EXPECT_DOUBLE_EQ(basel_traffic_light(0).multiplier, 3.0);
  EXPECT_DOUBLE_EQ(basel_traffic_light(4).multiplier, 3.0);
  EXPECT_DOUBLE_EQ(basel_traffic_light(5).multiplier, 3.40);
  EXPECT_DOUBLE_EQ(basel_traffic_light(6).multiplier, 3.50);
  EXPECT_DOUBLE_EQ(basel_traffic_light(7).multiplier, 3.65);
  EXPECT_DOUBLE_EQ(basel_traffic_light(8).multiplier, 3.75);
  EXPECT_DOUBLE_EQ(basel_traffic_light(9).multiplier, 3.85);
  EXPECT_DOUBLE_EQ(basel_traffic_light(10).multiplier, 4.0);
}

TEST(EvaluateBacktest, EndToEndCountsAndConvention) {
  // Exception iff loss (-pnl) strictly exceeds the VaR forecast.
  const std::vector<double> pnl{-120, 50, -80, -101, 30, -100};
  const std::vector<double> var(6, 100.0);
  const BacktestResult r = evaluate_var_backtest(pnl, var, 0.99);
  EXPECT_EQ(r.n_exceptions, 2);  // -120 and -101; -100 is not an exception
  const std::vector<int> expect_exc{1, 0, 0, 1, 0, 0};
  EXPECT_EQ(r.exceedances, expect_exc);
  EXPECT_NEAR(r.exception_rate, 2.0 / 6.0, 1e-15);
  EXPECT_NEAR(r.conditional.lr, r.kupiec.lr + r.independence.lr, 1e-12);
  EXPECT_THROW(evaluate_var_backtest({1.0}, {1.0}, 0.99),
               std::invalid_argument);
  EXPECT_THROW(evaluate_var_backtest({1.0, 2.0}, {1.0}, 0.99),
               std::invalid_argument);
  EXPECT_THROW(
      evaluate_var_backtest({1.0, std::nan("")}, {1.0, 1.0}, 0.99),
      std::invalid_argument);
}
