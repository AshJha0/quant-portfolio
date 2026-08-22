// Monte Carlo VaR: convergence to the parametric closed form within 3 SE,
// Student-t tail fattening, bitwise seed determinism, moment matching.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "eqvar/expected_shortfall.hpp"
#include "eqvar/monte_carlo.hpp"
#include "eqvar/parametric.hpp"
#include "eqvar/returns.hpp"
#include "eqvar/stats.hpp"

using namespace eqvar;

namespace {

Matrix demo_cov() {
    const std::vector<double> vols = {0.010, 0.015, 0.020};
    const Matrix corr(3, 3, {1.0, 0.5, 0.25, 0.5, 1.0, 0.3, 0.25, 0.3, 1.0});
    return covariance_from_vols(vols, corr);
}

const std::vector<double> kExposures = {1.0e6, -5.0e5, 2.0e5};

}  // namespace

TEST(MonteCarlo, BitwiseSeedDeterminism) {
    const Matrix cov = demo_cov();
    const MonteCarloResult a = monte_carlo_var(kExposures, cov, 0.01, 20'000, Dist::Normal, 6.0, 42);
    const MonteCarloResult b = monte_carlo_var(kExposures, cov, 0.01, 20'000, Dist::Normal, 6.0, 42);
    EXPECT_EQ(a.var, b.var);  // bitwise, not approximate
    EXPECT_EQ(a.es, b.es);
    EXPECT_EQ(a.var_se, b.var_se);
    const MonteCarloResult c = monte_carlo_var(kExposures, cov, 0.01, 20'000, Dist::Normal, 6.0, 43);
    EXPECT_NE(a.var, c.var);  // a different seed must move the estimate
}

TEST(MonteCarlo, NormalConvergesToParametricWithin3SE) {
    const Matrix cov = demo_cov();
    const double exact_var = parametric_var(kExposures, cov, 0.01);
    const double exact_es = normal_es(portfolio_sigma(kExposures, cov), 0.01);
    const MonteCarloResult mc =
        monte_carlo_var(kExposures, cov, 0.01, 200'000, Dist::Normal, 6.0, 7);
    ASSERT_GT(mc.var_se, 0.0);
    EXPECT_LT(mc.var_se, 0.02 * exact_var);  // 200k paths: SE well under 2 %
    EXPECT_NEAR(mc.var, exact_var, 3.0 * mc.var_se);
    // ES averages the tail; allow a slightly wider band on the same SE scale.
    EXPECT_NEAR(mc.es, exact_es, 4.0 * mc.var_se);
}

TEST(MonteCarlo, StudentTFatterThanNormalAt99AndConvergesToClosedForm) {
    const Matrix cov = demo_cov();
    const MonteCarloResult n =
        monte_carlo_var(kExposures, cov, 0.01, 200'000, Dist::Normal, 6.0, 11);
    const MonteCarloResult t =
        monte_carlo_var(kExposures, cov, 0.01, 200'000, Dist::StudentT, 5.0, 11);
    EXPECT_GT(t.var, n.var);
    EXPECT_GT(t.es, n.es);
    const double exact_t = parametric_var(kExposures, cov, 0.01, Dist::StudentT, 5.0);
    EXPECT_NEAR(t.var, exact_t, 3.0 * t.var_se);
}

TEST(MonteCarlo, EsDominatesVar) {
    const Matrix cov = demo_cov();
    for (std::uint64_t seed : {0ULL, 1ULL, 2ULL}) {
        const MonteCarloResult r =
            monte_carlo_var(kExposures, cov, 0.025, 50'000, Dist::Normal, 6.0, seed);
        EXPECT_GE(r.es, r.var) << "seed=" << seed;
    }
}

TEST(SimulateFactorReturns, MatchesTargetCovarianceNormalAndT) {
    const Matrix cov = demo_cov();
    for (Dist dist : {Dist::Normal, Dist::StudentT}) {
        const Matrix scen = simulate_factor_returns(cov, 200'000, dist, 6.0, 3);
        const Matrix hat = sample_covariance(scen);
        // Variances within 5 % relative, covariances within 10 %: ~ 3-4 MC SE
        // at 200k paths (t moments converge slower — same target covariance).
        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(hat(i, i) / cov(i, i), 1.0, 0.05) << "i=" << i;
            for (std::size_t j = 0; j < 3; ++j) {
                EXPECT_NEAR(hat(i, j), cov(i, j), 0.10 * std::sqrt(cov(i, i) * cov(j, j)));
            }
        }
    }
}

TEST(RandomStreamTest, UniformInOpenIntervalAndGaussianMoments) {
    RandomStream rng(123);
    double s = 0.0, s2 = 0.0;
    for (int i = 0; i < 100'000; ++i) {
        const double u = rng.uniform();
        EXPECT_GT(u, 0.0);
        EXPECT_LT(u, 1.0);
        const double g = normal_ppf(u);
        s += g;
        s2 += g * g;
    }
    EXPECT_NEAR(s / 1e5, 0.0, 0.015);   // ~ 4.7 SE
    EXPECT_NEAR(s2 / 1e5, 1.0, 0.02);
}

TEST(RandomStreamTest, ChiSquaredMeanAndVariance) {
    RandomStream rng(99);
    const double df = 6.0;
    double s = 0.0, s2 = 0.0;
    const int n = 50'000;
    for (int i = 0; i < n; ++i) {
        const double x = rng.chi_squared(df);
        EXPECT_GT(x, 0.0);
        s += x;
        s2 += x * x;
    }
    const double m = s / n;
    EXPECT_NEAR(m, df, 0.07);                       // E = df, SE ~ 0.015
    EXPECT_NEAR(s2 / n - m * m, 2.0 * df, 0.5);     // Var = 2 df
}

TEST(MonteCarlo, BootstrapSeSameSeedIsBitwiseIdentical) {
    const Matrix cov = demo_cov();
    const Matrix scen = simulate_factor_returns(cov, 20'000, Dist::Normal, 6.0, 5);
    const std::vector<double> pnl = portfolio_pnl(scen, kExposures);
    const double se_a = mc_bootstrap_se(pnl, 0.01, 200, 7);
    const double se_b = mc_bootstrap_se(pnl, 0.01, 200, 7);
    EXPECT_EQ(se_a, se_b);  // bitwise, not approximate
    const double se_c = mc_bootstrap_se(pnl, 0.01, 200, 8);
    EXPECT_NE(se_a, se_c);  // a different seed must move the estimate
}

TEST(MonteCarlo, BootstrapSeThrowsOnFewerThan10Scenarios) {
    const std::vector<double> tiny(9, -1.0);
    EXPECT_THROW(mc_bootstrap_se(tiny, 0.01), std::invalid_argument);
    const std::vector<double> ok(10, -1.0);
    EXPECT_NO_THROW(mc_bootstrap_se(ok, 0.01, 50));
}

TEST(MonteCarlo, BootstrapSeThrowsOnAlphaOutsideValidRange) {
    const std::vector<double> pnl(100, -1.0);
    EXPECT_THROW(mc_bootstrap_se(pnl, 0.0), std::invalid_argument);
    EXPECT_THROW(mc_bootstrap_se(pnl, 0.5), std::invalid_argument);
    EXPECT_THROW(mc_bootstrap_se(pnl, -0.1), std::invalid_argument);
    EXPECT_THROW(mc_bootstrap_se(pnl, 0.6), std::invalid_argument);
}

TEST(MonteCarlo, BootstrapSeWithinSaneMultipleOfOrderStatisticSe) {
    const Matrix cov = demo_cov();
    const Matrix scen = simulate_factor_returns(cov, 50'000, Dist::Normal, 6.0, 21);
    const std::vector<double> pnl = portfolio_pnl(scen, kExposures);
    const MonteCarloResult ref = mc_tail_metrics(pnl, 0.01);
    const double boot_se = mc_bootstrap_se(pnl, 0.01, 300, 3);
    ASSERT_GT(ref.var_se, 0.0);
    ASSERT_GT(boot_se, 0.0);
    // Different estimators, not expected to agree closely; just guard
    // against wild divergence (the bootstrap is known to run a bit higher
    // in deep tails, per the order-statistic estimator's low bias there).
    EXPECT_LT(boot_se, 3.0 * ref.var_se);
    EXPECT_GT(boot_se, ref.var_se / 3.0);
}

TEST(MonteCarlo, BootstrapSeHandlesDegenerateZeroVarianceSampleWithoutNaN) {
    const std::vector<double> flat(200, 42.0);
    const double se = mc_bootstrap_se(flat, 0.01, 100, 1);
    EXPECT_TRUE(std::isfinite(se));
    EXPECT_DOUBLE_EQ(se, 0.0);
}

TEST(MonteCarlo, Validation) {
    const Matrix cov = demo_cov();
    EXPECT_THROW(monte_carlo_var(std::vector<double>{}, cov, 0.01), std::invalid_argument);
    EXPECT_THROW(monte_carlo_var(std::vector<double>{1.0}, cov, 0.01), std::invalid_argument);
    EXPECT_THROW(monte_carlo_var(kExposures, cov, 0.6), std::invalid_argument);
    EXPECT_THROW(monte_carlo_var(kExposures, cov, 0.01, 1000, Dist::StudentT, 2.0),
                 std::invalid_argument);
    EXPECT_THROW(simulate_factor_returns(cov, 0), std::invalid_argument);
    const std::vector<double> tiny(50, 0.0);
    EXPECT_THROW(mc_tail_metrics(tiny, 0.01), std::invalid_argument);
}
