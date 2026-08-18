// Special functions: inverse normal CDF vs known quantiles, chi-squared
// p-values via the regularized incomplete gamma, Student-t quantiles via the
// incomplete beta, sample moments. Reference values from scipy 1.17.1.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "eqvar/stats.hpp"

using namespace eqvar;

TEST(NormalPpf, KnownQuantiles) {
    // Phi^{-1}(0.975) — the canonical 1.96; spec requires 1e-8, we hold 1e-12.
    EXPECT_NEAR(normal_ppf(0.975), 1.959963984540054, 1e-12);
    EXPECT_NEAR(normal_ppf(0.01), -2.3263478740408408, 1e-12);
    EXPECT_NEAR(normal_ppf(0.995), 2.5758293035489004, 1e-12);
    EXPECT_NEAR(normal_ppf(0.05), -1.6448536269514722, 1e-12);
    EXPECT_DOUBLE_EQ(normal_ppf(0.5), 0.0);
}

TEST(NormalPpf, SymmetryAndRoundTrip) {
    for (double p : {1e-8, 1e-4, 0.01, 0.025, 0.3, 0.5, 0.7, 0.975, 0.9999}) {
        EXPECT_NEAR(normal_ppf(p), -normal_ppf(1.0 - p), 1e-9) << "p=" << p;
        EXPECT_NEAR(normal_cdf(normal_ppf(p)), p, 1e-12) << "p=" << p;
    }
}

TEST(NormalPpf, RejectsOutOfRange) {
    EXPECT_THROW(normal_ppf(0.0), std::invalid_argument);
    EXPECT_THROW(normal_ppf(1.0), std::invalid_argument);
    EXPECT_THROW(normal_ppf(-0.1), std::invalid_argument);
}

TEST(NormalCdfPdf, KnownValues) {
    EXPECT_NEAR(normal_cdf(0.0), 0.5, 1e-15);
    EXPECT_NEAR(normal_cdf(1.959963984540054), 0.975, 1e-14);
    EXPECT_NEAR(normal_pdf(0.0), 0.3989422804014327, 1e-15);  // 1/sqrt(2 pi)
    // pdf is the derivative of cdf (central difference check).
    const double h = 1e-6;
    EXPECT_NEAR((normal_cdf(1.0 + h) - normal_cdf(1.0 - h)) / (2 * h), normal_pdf(1.0), 1e-9);
}

TEST(IncompleteGamma, Chi2PValues) {
    // chi2(1) critical value 3.841... => p = 0.05 (spec tolerance 1e-4; held to 1e-12).
    EXPECT_NEAR(chi2_sf(3.841458820694124, 1.0), 0.05, 1e-12);
    EXPECT_NEAR(chi2_sf(5.991464547107979, 2.0), 0.05, 1e-12);
    EXPECT_NEAR(chi2_sf(2.0, 1.0), 0.15729920705028105, 1e-12);  // scipy chi2.sf(2, 1)
    EXPECT_DOUBLE_EQ(chi2_sf(0.0, 1.0), 1.0);
}

TEST(IncompleteGamma, PPlusQIsOne) {
    for (double a : {0.5, 1.0, 2.5, 10.0}) {
        for (double x : {0.1, 1.0, 5.0, 30.0}) {
            EXPECT_NEAR(regularized_gamma_p(a, x) + regularized_gamma_q(a, x), 1.0, 1e-13)
                << "a=" << a << " x=" << x;
        }
    }
}

TEST(IncompleteBeta, BinomialCdf) {
    // scipy binom.cdf(4, 250, 0.01) and binom.cdf(9, 250, 0.01) — the Basel
    // green / yellow boundary probabilities.
    EXPECT_NEAR(binomial_cdf(4, 250, 0.01), 0.89218762690362508, 1e-12);
    EXPECT_NEAR(binomial_cdf(9, 250, 0.01), 0.99974980993125950, 1e-12);
    EXPECT_DOUBLE_EQ(binomial_cdf(250, 250, 0.01), 1.0);
    EXPECT_DOUBLE_EQ(binomial_cdf(-1, 250, 0.01), 0.0);
}

TEST(StudentT, QuantilesVsScipy) {
    // scipy.stats.t.ppf reference values (bisection documented in stats.hpp).
    EXPECT_NEAR(student_t_ppf(0.01, 6.0), -3.1426684032910068, 1e-9);
    EXPECT_NEAR(student_t_ppf(0.05, 8.0), -1.8595480375308979, 1e-9);
    EXPECT_NEAR(student_t_ppf(0.025, 4.5), -2.6589123472044038, 1e-9);
    EXPECT_DOUBLE_EQ(student_t_ppf(0.5, 6.0), 0.0);
}

TEST(StudentT, CdfPpfRoundTripAndLimits) {
    EXPECT_NEAR(student_t_cdf(-2.0, 5.0), 0.050969739414929174, 1e-12);  // scipy
    for (double p : {0.001, 0.01, 0.1, 0.6, 0.99}) {
        EXPECT_NEAR(student_t_cdf(student_t_ppf(p, 7.0), 7.0), p, 1e-12) << "p=" << p;
    }
    // df -> infinity: t quantile approaches the normal quantile.
    EXPECT_NEAR(student_t_ppf(0.01, 1e6), normal_ppf(0.01), 1e-4);
    // Fat tails: |t quantile| > |normal quantile| in the tail.
    EXPECT_LT(student_t_ppf(0.01, 4.0), normal_ppf(0.01));
}

TEST(Moments, KnownSample) {
    const std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_DOUBLE_EQ(mean(x), 3.0);
    EXPECT_NEAR(stdev(x), std::sqrt(2.5), 1e-15);  // ddof = 1
    EXPECT_NEAR(skewness(x), 0.0, 1e-15);          // symmetric
    EXPECT_NEAR(excess_kurtosis(x), -1.3, 1e-12);  // scipy.stats.kurtosis([1..5])
    const std::vector<double> y = {1.0, 1.0, 1.0, 10.0};
    EXPECT_NEAR(skewness(y), 1.1547005383792515, 1e-12);  // scipy bias=True
}
