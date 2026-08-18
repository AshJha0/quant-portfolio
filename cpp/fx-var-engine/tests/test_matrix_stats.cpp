// Matrix / Cholesky / special-function tests.
//
// Reference values: inverse normal CDF quantiles are standard
// high-precision constants (Wichura AS241 tables / scipy.stats.norm.ppf);
// chi-square and Student-t spot values cross-checked against scipy
// (see tests/test_golden_python.cpp for the systematic cross-language
// constants).

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "fxvar/matrix.hpp"
#include "fxvar/stats.hpp"

using namespace fxvar;

TEST(Matrix, BasicOpsAndMatmul) {
  const Matrix a = Matrix::from_rows({{1.0, 2.0}, {3.0, 4.0}});
  const Matrix b = Matrix::from_rows({{5.0, 6.0}, {7.0, 8.0}});
  const Matrix c = matmul(a, b);
  EXPECT_DOUBLE_EQ(c(0, 0), 19.0);
  EXPECT_DOUBLE_EQ(c(0, 1), 22.0);
  EXPECT_DOUBLE_EQ(c(1, 0), 43.0);
  EXPECT_DOUBLE_EQ(c(1, 1), 50.0);
  const auto y = matvec(a, {1.0, -1.0});
  EXPECT_DOUBLE_EQ(y[0], -1.0);
  EXPECT_DOUBLE_EQ(y[1], -1.0);
  EXPECT_DOUBLE_EQ(quad_form({1.0, 1.0}, a), 10.0);
  EXPECT_THROW(Matrix::from_rows({{1.0}, {1.0, 2.0}}), std::invalid_argument);
}

TEST(Cholesky, ReconstructsPositiveDefinite) {
  const Matrix cov = Matrix::from_rows(
      {{4.0, 2.0, 0.6}, {2.0, 3.0, 0.4}, {0.6, 0.4, 1.0}});
  const CholeskyResult res = robust_cholesky(cov);
  EXPECT_FALSE(res.jittered);
  EXPECT_EQ(res.jitter, 0.0);
  const Matrix rec = matmul(res.lower, res.lower.transpose());
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      EXPECT_NEAR(rec(i, j), cov(i, j), 1e-12);
}

TEST(Cholesky, SingularCovarianceGetsJitterAndWarning) {
  // Two perfectly correlated factors: a pegged pair to the same anchor.
  const Matrix cov = Matrix::from_rows({{1e-4, 1e-4}, {1e-4, 1e-4}});
  const CholeskyResult res = robust_cholesky(cov);
  EXPECT_TRUE(res.jittered);
  EXPECT_GT(res.jitter, 0.0);
  EXPECT_FALSE(res.warning.empty());
  const Matrix rec = matmul(res.lower, res.lower.transpose());
  EXPECT_NEAR(rec(0, 0), cov(0, 0), 1e-8);
  EXPECT_NEAR(rec(0, 1), cov(0, 1), 1e-8);
}

TEST(Cholesky, RejectsAsymmetricAndNonSquare) {
  EXPECT_THROW(robust_cholesky(Matrix::from_rows({{1.0, 0.5}, {0.2, 1.0}})),
               std::invalid_argument);
  EXPECT_THROW(robust_cholesky(Matrix(2, 3)), std::invalid_argument);
}

TEST(NormalDistribution, PdfCdfKnownValues) {
  EXPECT_NEAR(norm_pdf(0.0), 0.3989422804014327, 1e-15);
  EXPECT_NEAR(norm_cdf(0.0), 0.5, 1e-15);
  EXPECT_NEAR(norm_cdf(1.959963984540054), 0.975, 1e-12);
  EXPECT_NEAR(norm_cdf(-2.3263478740408408), 0.01, 1e-12);
}

TEST(NormalDistribution, InverseCdfBelow1eMinus9) {
  // High-precision reference quantiles (AS241 / scipy.stats.norm.ppf).
  EXPECT_NEAR(norm_ppf(0.5), 0.0, 1e-15);
  EXPECT_NEAR(norm_ppf(0.975), 1.959963984540054, 1e-9);
  EXPECT_NEAR(norm_ppf(0.99), 2.3263478740408408, 1e-9);
  EXPECT_NEAR(norm_ppf(0.999), 3.090232306167813, 1e-9);
  EXPECT_NEAR(norm_ppf(0.95), 1.6448536269514722, 1e-9);
  EXPECT_NEAR(norm_ppf(0.025), -1.959963984540054, 1e-9);
  EXPECT_NEAR(norm_ppf(1e-6), -4.753424308822899, 1e-8);
  // Round trip over a wide range.
  for (double p : {1e-8, 1e-4, 0.1, 0.3, 0.7, 0.9, 0.9999, 1.0 - 1e-8})
    EXPECT_NEAR(norm_cdf(norm_ppf(p)), p, 1e-12);
  EXPECT_THROW(norm_ppf(0.0), std::invalid_argument);
  EXPECT_THROW(norm_ppf(1.0), std::invalid_argument);
}

TEST(Gamma, RegularisedIncompleteAndChi2) {
  // P(a, x) at analytic points: P(0.5, x) = erf(sqrt(x)).
  EXPECT_NEAR(reg_lower_gamma(0.5, 1.0), std::erf(1.0), 1e-12);
  EXPECT_NEAR(reg_upper_gamma(0.5, 1.0), std::erfc(1.0), 1e-12);
  // chi2(1) sf at the 5% critical value (scipy chi2.isf(0.05, 1)).
  EXPECT_NEAR(chi2_sf(3.8414588206941285, 1.0), 0.05, 1e-10);
  // chi2(2) sf(x) = exp(-x/2) exactly.
  EXPECT_NEAR(chi2_sf(5.991464547107983, 2.0),
              std::exp(-5.991464547107983 / 2.0), 1e-12);
  EXPECT_NEAR(chi2_sf(5.991464547107983, 2.0), 0.05, 1e-10);
}

TEST(StudentT, CdfQuantileRoundTripAndKnownValues) {
  // t quantiles vs scipy.stats.t.ppf.
  EXPECT_NEAR(t_ppf(0.975, 5.0), 2.5705818356363146, 1e-10);
  EXPECT_NEAR(t_ppf(0.99, 6.0), 3.1426684032910064, 1e-10);
  EXPECT_NEAR(t_ppf(0.5, 7.0), 0.0, 1e-15);
  // t(1) is Cauchy: quantile(0.75) = 1 exactly.
  EXPECT_NEAR(t_ppf(0.75, 1.0), 1.0, 1e-10);
  for (double p : {0.01, 0.2, 0.5, 0.8, 0.99})
    EXPECT_NEAR(t_cdf(t_ppf(p, 4.5), 4.5), p, 1e-11);
  EXPECT_THROW(t_ppf(1.2, 5.0), std::invalid_argument);
}

TEST(Binomial, ExactCdf) {
  // Binomial(4, 0.5): P(X <= 2) = (1+4+6)/16.
  EXPECT_NEAR(binom_cdf(2, 4, 0.5), 11.0 / 16.0, 1e-14);
  EXPECT_DOUBLE_EQ(binom_cdf(-1, 10, 0.3), 0.0);
  EXPECT_DOUBLE_EQ(binom_cdf(10, 10, 0.3), 1.0);
}

TEST(Moments, HandComputedSample) {
  // x = {1, 2, 3, 4}: mean 2.5, ddof=1 sd sqrt(5/3), zero skew,
  // m4/m2^2 - 3 = (2*(1.5^4) + 2*(0.5^4))/4 / (1.25^2) - 3 = -1.36.
  const Moments m = sample_moments({1.0, 2.0, 3.0, 4.0});
  EXPECT_NEAR(m.mean, 2.5, 1e-15);
  EXPECT_NEAR(m.stdev, std::sqrt(5.0 / 3.0), 1e-15);
  EXPECT_NEAR(m.skewness, 0.0, 1e-15);
  EXPECT_NEAR(m.excess_kurtosis, -1.36, 1e-12);
  EXPECT_THROW(sample_moments({1.0}), std::invalid_argument);
}

TEST(Validation, AlphaAndHorizonBounds) {
  EXPECT_THROW(validate_alpha(0.0), std::invalid_argument);
  EXPECT_THROW(validate_alpha(1.0), std::invalid_argument);
  EXPECT_THROW(validate_alpha(-0.1), std::invalid_argument);
  EXPECT_DOUBLE_EQ(validate_alpha(0.99), 0.99);
  EXPECT_THROW(validate_horizon(0.0), std::invalid_argument);
  EXPECT_DOUBLE_EQ(validate_horizon(10.0), 10.0);
}
