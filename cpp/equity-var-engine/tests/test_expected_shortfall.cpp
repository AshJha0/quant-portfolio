// Expected Shortfall: hand-exact empirical tail integral, the closed-form
// normal ES identity against numerical quadrature to 1e-10, ES >= VaR.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "eqvar/expected_shortfall.hpp"
#include "eqvar/historical.hpp"
#include "eqvar/stats.hpp"

using namespace eqvar;

TEST(EmpiricalEs, HandExactTinyArrays) {
    // Sorted: -10 -8 -6 -4 -2 1 2 3 4 5 (n = 10).
    const std::vector<double> pnl = {1.0, -10.0, 2.0, -8.0, 3.0, -6.0, 4.0, -4.0, 5.0, -2.0};
    // alpha = 0.2: an = 2 exactly -> mean of two worst losses.
    EXPECT_DOUBLE_EQ(expected_shortfall(pnl, 0.2), 9.0);
    // alpha = 0.25: an = 2.5 -> fractional weight 0.5 on the 3rd order stat:
    // -( -10 - 8 + 0.5*(-6) ) / 2.5 = 8.4.
    EXPECT_DOUBLE_EQ(expected_shortfall(pnl, 0.25), 8.4);
    // alpha = 0.1: an = 1 -> the single worst loss.
    EXPECT_DOUBLE_EQ(expected_shortfall(pnl, 0.1), 10.0);
}

TEST(EmpiricalEs, DominatesVarOnSameSample) {
    std::vector<double> pnl(250);
    for (int t = 0; t < 250; ++t) {
        pnl[t] = 1.0e4 * std::sin(2.1 * t) + 3.0e3 * std::cos(0.37 * t * t);
    }
    for (double alpha : {0.01, 0.025, 0.05, 0.10}) {
        EXPECT_GE(expected_shortfall(pnl, alpha), historical_var(pnl, alpha))
            << "alpha=" << alpha;
    }
}

TEST(NormalEs, IdentityVsNumericalQuadratureTo1e10) {
    // ES * alpha = -sigma * int_{-inf}^{z} x phi(x) dx (= sigma * phi(z)).
    // Verify the closed form against Simpson quadrature of the tail integral.
    const double sigma = 1.7e4;
    for (double alpha : {0.01, 0.025, 0.05}) {
        const double z = normal_ppf(alpha);
        const double lo = -14.0;  // phi(-14) ~ 5e-44: truncation negligible
        const int n = 40000;      // even
        const double h = (z - lo) / n;
        double integral = 0.0;
        for (int i = 0; i <= n; ++i) {
            const double x = lo + h * i;
            const double f = x * normal_pdf(x);
            const double w = (i == 0 || i == n) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
            integral += w * f;
        }
        integral *= h / 3.0;
        const double es_quad = -sigma * integral / alpha;
        EXPECT_NEAR(normal_es(sigma, alpha) / es_quad, 1.0, 1e-10) << "alpha=" << alpha;
    }
}

TEST(NormalEs, ExceedsNormalVarAndScalesWithSigma) {
    const double sigma = 1.0e4;
    for (double alpha : {0.01, 0.025, 0.05}) {
        EXPECT_GT(normal_es(sigma, alpha), -normal_ppf(alpha) * sigma) << "alpha=" << alpha;
    }
    EXPECT_NEAR(normal_es(2.0 * sigma, 0.01), 2.0 * normal_es(sigma, 0.01), 1e-9);
    EXPECT_NEAR(normal_es(sigma, 0.01, 100.0), normal_es(sigma, 0.01) - 100.0, 1e-12);
    EXPECT_DOUBLE_EQ(normal_es(0.0, 0.01), 0.0);
}

TEST(StudentTEs, FatterThanNormalAndNormalLimit) {
    const double sigma = 1.0e4;
    EXPECT_GT(student_t_es(sigma, 0.01, 4.0), normal_es(sigma, 0.01));
    EXPECT_GT(student_t_es(sigma, 0.025, 6.0), normal_es(sigma, 0.025));
    // df -> infinity recovers the normal ES.
    EXPECT_NEAR(student_t_es(sigma, 0.025, 1.0e5), normal_es(sigma, 0.025),
                1e-3 * normal_es(sigma, 0.025));
    // t ES must dominate the variance-matched t VaR.
    const double t_var = -student_t_ppf(0.01, 6.0) * std::sqrt(4.0 / 6.0) * sigma;
    EXPECT_GT(student_t_es(sigma, 0.01, 6.0), t_var);
}

TEST(Es, Validation) {
    const std::vector<double> tiny = {1.0, -1.0, 2.0};
    EXPECT_THROW(expected_shortfall(tiny, 0.05), std::invalid_argument);
    std::vector<double> pnl(50, 1.0);
    pnl[0] = -1.0;
    EXPECT_THROW(expected_shortfall(pnl, 0.0), std::invalid_argument);
    EXPECT_THROW(expected_shortfall(pnl, 0.5), std::invalid_argument);
    EXPECT_THROW(normal_es(-1.0, 0.01), std::invalid_argument);
    EXPECT_THROW(student_t_es(1.0, 0.01, 2.0), std::invalid_argument);
}
