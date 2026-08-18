// Parametric (variance-covariance) VaR: closed-form identities to 1e-12,
// Student-t tail behaviour, Cornish-Fisher with validity-domain check.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "eqvar/parametric.hpp"
#include "eqvar/stats.hpp"

using namespace eqvar;

TEST(PortfolioSigma, HandComputedTwoAsset) {
    // sigma^2 = w1^2 s11 + 2 w1 w2 s12 + w2^2 s22
    const Matrix cov(2, 2, {4.0e-4, 1.0e-4, 1.0e-4, 2.5e-4});
    const std::vector<double> w = {1.0e6, -2.0e6};
    const double expected =
        std::sqrt(1e12 * 4e-4 - 2.0 * 1e6 * 2e6 * 1e-4 + 4e12 * 2.5e-4);
    EXPECT_NEAR(portfolio_sigma(w, cov), expected, 1e-12 * expected);
}

TEST(PortfolioSigma, DiversificationNeverHurts) {
    // sigma(portfolio) <= sum of standalone sigmas for long-only exposures.
    const Matrix cov(2, 2, {4.0e-4, 1.0e-4, 1.0e-4, 2.5e-4});
    const std::vector<double> w = {1.0e6, 1.0e6};
    const double standalone = 1e6 * std::sqrt(4e-4) + 1e6 * std::sqrt(2.5e-4);
    EXPECT_LT(portfolio_sigma(w, cov), standalone);
}

TEST(ParametricVar, NormalClosedFormTo1e12) {
    // Single asset: VaR = -z_alpha * |w| * vol exactly.
    const Matrix cov(1, 1, {4.0e-4});  // vol = 2 %
    const std::vector<double> w = {1.0e6};
    const double sigma = 2.0e4;
    const double expected_99 = -normal_ppf(0.01) * sigma;
    EXPECT_NEAR(parametric_var(w, cov, 0.01), expected_99, 1e-12 * expected_99);
    const double expected_95 = 1.6448536269514722 * sigma;  // z_0.95, Wichura
    EXPECT_NEAR(parametric_var(w, cov, 0.05), expected_95, 1e-8);
    // Non-zero mean shifts VaR linearly.
    EXPECT_NEAR(parametric_var(w, cov, 0.01, Dist::Normal, 6.0, 500.0),
                expected_99 - 500.0, 1e-9);
}

TEST(ParametricVar, StudentTFatterThanNormalAndVarianceMatched) {
    const Matrix cov(1, 1, {4.0e-4});
    const std::vector<double> w = {1.0e6};
    const double v_norm = parametric_var(w, cov, 0.01, Dist::Normal);
    const double v_t4 = parametric_var(w, cov, 0.01, Dist::StudentT, 4.0);
    const double v_t100 = parametric_var(w, cov, 0.01, Dist::StudentT, 100.0);
    EXPECT_GT(v_t4, v_norm);          // fat tails at 99 %
    EXPECT_GT(v_t4, v_t100);          // fatness decreases with df
    EXPECT_NEAR(v_t100, v_norm, 0.02 * v_norm);  // df -> inf: normal limit
    // Closed form: z_t = t_ppf(alpha, df) * sqrt((df-2)/df).
    const double expected =
        -student_t_ppf(0.01, 6.0) * std::sqrt(4.0 / 6.0) * 2.0e4;
    EXPECT_NEAR(parametric_var(w, cov, 0.01, Dist::StudentT, 6.0), expected, 1e-12 * expected);
}

TEST(ParametricVar, SqrtTimeHorizonScaling) {
    const Matrix cov(1, 1, {4.0e-4});
    const std::vector<double> w = {1.0e6};
    const double v1 = parametric_var(w, cov, 0.01);
    const double v10 = parametric_var(w, cov, 0.01, Dist::Normal, 6.0, 0.0, 10);
    EXPECT_NEAR(v10, v1 * std::sqrt(10.0), 1e-9);
    EXPECT_THROW(parametric_var(w, cov, 0.01, Dist::Normal, 6.0, 0.0, 0),
                 std::invalid_argument);
}

TEST(ParametricVar, Validation) {
    const Matrix cov(1, 1, {4.0e-4});
    const std::vector<double> w = {1.0e6};
    EXPECT_THROW(parametric_var(w, cov, 0.0), std::invalid_argument);
    EXPECT_THROW(parametric_var(w, cov, 0.5), std::invalid_argument);
    EXPECT_THROW(parametric_var(w, cov, 0.01, Dist::StudentT, 2.0), std::invalid_argument);
    EXPECT_THROW(parametric_var(std::vector<double>{}, cov, 0.01), std::invalid_argument);
    EXPECT_THROW(parametric_var(std::vector<double>{1.0, 2.0}, cov, 0.01),
                 std::invalid_argument);
}

TEST(CornishFisher, ReducesToNormalAtZeroMoments) {
    const double z = normal_ppf(0.01);
    EXPECT_DOUBLE_EQ(cornish_fisher_z(z, 0.0, 0.0), z);
    EXPECT_NEAR(cornish_fisher_var(1.0e4, 0.01, 0.0, 0.0), -z * 1.0e4, 1e-9);
}

TEST(CornishFisher, LeftSkewAndFatTailsRaiseVar) {
    // Moment pairs chosen inside the CF validity domain (S = -0.5, K = 0 is
    // already non-monotone on |z| <= 3.5 — in the Python reference too).
    const double base = cornish_fisher_var(1.0e4, 0.01, 0.0, 0.0);
    EXPECT_GT(cornish_fisher_var(1.0e4, 0.01, -0.2, 0.0), base);  // left skew
    EXPECT_GT(cornish_fisher_var(1.0e4, 0.01, 0.0, 1.0), base);   // excess kurt
    // Values pinned to the Python reference (eq_var.cornish_fisher_var).
    EXPECT_NEAR(cornish_fisher_var(1.0e4, 0.01, -0.2, 0.0), 24583.575119223973, 1e-6);
    EXPECT_NEAR(cornish_fisher_var(1.0e4, 0.01, 0.0, 1.0), 25601.356024614324, 1e-6);
}

TEST(CornishFisher, DomainCheckRejectsNonMonotoneRegion) {
    EXPECT_TRUE(cornish_fisher_domain_ok(0.0, 0.0));
    EXPECT_TRUE(cornish_fisher_domain_ok(-0.5, 1.0));
    // Large skew makes the cubic non-monotone: not a quantile function.
    EXPECT_FALSE(cornish_fisher_domain_ok(3.0, 0.0));
    EXPECT_THROW(cornish_fisher_var(1.0e4, 0.01, 3.0, 0.0), std::invalid_argument);
    // Explicit override for diagnostics only.
    EXPECT_NO_THROW({
        const double v = cornish_fisher_var(1.0e4, 0.01, 3.0, 0.0, 0.0, false);
        (void)v;
    });
    EXPECT_THROW(cornish_fisher_var(-1.0, 0.01), std::invalid_argument);
}
