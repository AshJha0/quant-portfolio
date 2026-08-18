// Parametric (variance-covariance) VaR: closed-form identities,
// Cornish-Fisher domain checks, singular covariance handling.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "fxvar/expected_shortfall.hpp"
#include "fxvar/parametric.hpp"
#include "fxvar/stats.hpp"

using namespace fxvar;

namespace {

Market test_market() {
  return Market({{"EUR", 1.10}, {"JPY", 0.0090}},
                {{"USD", 0.050}, {"EUR", 0.030}, {"JPY", 0.001}});
}

}  // namespace

TEST(VarCovar, ClosedFormMatchesHandComputation) {
  // Two factors, hand-computable: sigma_p^2 = w' Sigma w.
  const std::vector<double> w{1e6, -5e5};
  const Matrix cov =
      Matrix::from_rows({{4e-5, 1e-5}, {1e-5, 9e-5}});
  const double sig2 = 1e12 * 4e-5 - 2.0 * 1e6 * 5e5 * 1e-5 + 2.5e11 * 9e-5;
  const double sig = std::sqrt(sig2);
  const VarEs ve = var_covar(w, cov, 0.99, 1.0, TailDist::kNormal);
  EXPECT_NEAR(ve.var, sig * norm_ppf(0.99), 1e-12 * ve.var);
  EXPECT_NEAR(ve.es, sig * norm_pdf(norm_ppf(0.99)) / 0.01, 1e-12 * ve.es);
  EXPECT_NEAR(portfolio_sigma(w, cov), sig, 1e-12 * sig);
}

TEST(VarCovar, SqrtTimeAndDistributionOrdering) {
  const std::vector<double> w{1e6};
  const Matrix cov = Matrix::from_rows({{6.4e-5}});
  const VarEs d1 = var_covar(w, cov, 0.99, 1.0);
  const VarEs d10 = var_covar(w, cov, 0.99, 10.0);
  EXPECT_NEAR(d10.var, d1.var * std::sqrt(10.0), 1e-10 * d10.var);
  EXPECT_NEAR(d10.es, d1.es * std::sqrt(10.0), 1e-10 * d10.es);
  // t overlay is fatter at 99% at equal sigma.
  const VarEs t5 = var_covar(w, cov, 0.99, 1.0, TailDist::kStudentT, 5.0);
  EXPECT_GT(t5.var, d1.var);
  EXPECT_GT(t5.es, d1.es);
}

TEST(VarCovar, SingularCovarianceIsHandledPsd) {
  // Perfectly correlated factors with offsetting exposures: sigma = 0
  // exactly; the quadratic form must clamp tiny negative round-off, not
  // throw.
  const std::vector<double> w{1e6, -1e6};
  const Matrix cov = Matrix::from_rows({{1e-4, 1e-4}, {1e-4, 1e-4}});
  EXPECT_NEAR(portfolio_sigma(w, cov), 0.0, 1e-12);
  const VarEs ve = var_covar(w, cov, 0.99, 1.0);
  EXPECT_DOUBLE_EQ(ve.var, 0.0);
}

TEST(ParametricVar, FullPipelineMatchesExposureClosedForm) {
  // parametric_var(book) must equal var_covar(linear exposures, sample
  // cov) - the driver adds nothing but wiring.
  const Market m = test_market();
  Book book({SpotPosition{"EURUSD", 10e6, {}},
             ForwardPosition{"USDJPY", 5e6, 0.5, {}}});
  ReturnsMatrix rets;
  rets.factors = {"FX:EUR", "FX:JPY", "IR:JPY", "IR:USD"};
  rets.data = Matrix(250, 4);
  for (int i = 0; i < 250; ++i) {
    const double t = static_cast<double>(i);
    rets.data(i, 0) = 0.006 * std::sin(0.11 * t);
    rets.data(i, 1) = 0.007 * std::cos(0.07 * t + 0.3);
    rets.data(i, 2) = 0.0004 * std::sin(0.05 * t + 1.0);
    rets.data(i, 3) = 0.0005 * std::cos(0.03 * t);
  }
  const auto res = parametric_var(book, m, rets, {});
  const CompiledBook cb(book, m);
  const FactorCov cov = sample_cov(rets.select(cb.factors()));
  const VarEs direct = var_covar(cb.linear_exposures(), cov.cov, 0.99, 1.0);
  EXPECT_NEAR(res.var, direct.var, 1e-12 * direct.var);
  EXPECT_NEAR(res.es, direct.es, 1e-12 * direct.es);
  ASSERT_EQ(res.exposures.size(), 4u);
}

TEST(ParametricVar, EmptyBookThrowsFactorlessIsZero) {
  const Market m = test_market();
  ReturnsMatrix rets;
  rets.factors = {"FX:EUR"};
  rets.data = Matrix(100, 1, 0.001);
  EXPECT_THROW(parametric_var(Book{}, m, rets, {}), std::invalid_argument);
  Book cash({CashPosition{"USD", 1e6}});
  const auto res = parametric_var(cash, m, rets, {});
  EXPECT_DOUBLE_EQ(res.var, 0.0);
  EXPECT_DOUBLE_EQ(res.sigma, 0.0);
}

TEST(CornishFisher, ReducesToNormalAtZeroMoments) {
  EXPECT_NEAR(cornish_fisher_z(1.5, 0.0, 0.0), 1.5, 1e-15);
  const double v = cornish_fisher_var(2e6, 0.0, 0.0, 0.99);
  EXPECT_NEAR(v, normal_var(2e6, 0.99), 1e-6);
}

TEST(CornishFisher, NegativeSkewRaisesLossQuantile) {
  // Negative P&L skew (carry-trade profile) must increase VaR.
  const double v0 = cornish_fisher_var(1e6, 0.0, 0.0, 0.99);
  const double vneg = cornish_fisher_var(1e6, -0.6, 0.5, 0.99);
  EXPECT_GT(vneg, v0);
}

TEST(CornishFisher, DomainCheckRejectsNonMonotone) {
  // (S, K) far outside the Maillard validity domain: the expansion is
  // non-monotone and must be refused by default.
  EXPECT_FALSE(cornish_fisher_domain_ok(2.5, 0.0));
  EXPECT_TRUE(cornish_fisher_domain_ok(0.0, 1.0));
  EXPECT_THROW(cornish_fisher_var(1e6, 2.5, 0.0, 0.99), std::invalid_argument);
  // Forcing check_domain=false returns a number (documented escape hatch).
  EXPECT_NO_THROW(cornish_fisher_var(1e6, 2.5, 0.0, 0.99, 0.0, 1.0, false));
}

TEST(ParametricVar, PegFlagSurfacedThroughDriver) {
  const Market m = Market({{"EUR", 1.10}, {"HKD", 0.1282}}, {});
  Book book({SpotPosition{"EURUSD", 10e6, {}}, SpotPosition{"USDHKD", 5e6, {}}});
  ReturnsMatrix rets;
  rets.factors = {"FX:EUR", "FX:HKD"};
  rets.data = Matrix(200, 2);
  for (int i = 0; i < 200; ++i) {
    rets.data(i, 0) = 0.005 * std::sin(0.3 * i);
    rets.data(i, 1) = 0.0001 * std::sin(0.7 * i);  // pegged
  }
  const auto res = parametric_var(book, m, rets, {});
  ASSERT_EQ(res.flagged_peg_factors.size(), 1u);
  EXPECT_EQ(res.flagged_peg_factors[0], "FX:HKD");
}
