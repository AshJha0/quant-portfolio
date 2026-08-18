// Edge cases from the documentation contract: empty inputs, single-asset
// portfolios, alpha at the boundaries of (0, 0.5), zero-variance assets,
// degenerate P&L series, and returns/pnl-mapping validation.  Each case here
// is also documented in docs/VALIDATION.md.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "eqvar/expected_shortfall.hpp"
#include "eqvar/historical.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/monte_carlo.hpp"
#include "eqvar/parametric.hpp"
#include "eqvar/returns.hpp"
#include "eqvar/stats.hpp"

using namespace eqvar;

TEST(EdgeCases, EmptyInputsThrow) {
    const std::vector<double> empty;
    const Matrix cov(1, 1, {1e-4});
    EXPECT_THROW(historical_var(empty, 0.01), std::invalid_argument);
    EXPECT_THROW(expected_shortfall(empty, 0.01), std::invalid_argument);
    EXPECT_THROW(quantile_linear(empty, 0.5), std::invalid_argument);
    EXPECT_THROW(portfolio_sigma(empty, cov), std::invalid_argument);
    EXPECT_THROW(parametric_var(empty, cov, 0.01), std::invalid_argument);
    EXPECT_THROW(monte_carlo_var(empty, cov, 0.01), std::invalid_argument);
    EXPECT_THROW(portfolio_pnl(Matrix(5, 1, 0.0), empty), std::invalid_argument);
    EXPECT_THROW(mean(empty), std::invalid_argument);
    EXPECT_THROW(log_returns(empty), std::invalid_argument);
}

TEST(EdgeCases, SingleAssetPortfolioIsExact) {
    // n = 1: parametric, MC and ES must all collapse to the scalar formulas.
    const double vol = 0.02, w0 = 1.0e6;
    const Matrix cov(1, 1, {vol * vol});
    const std::vector<double> w = {w0};
    const double sigma = w0 * vol;
    EXPECT_NEAR(portfolio_sigma(w, cov), sigma, 1e-9);
    EXPECT_NEAR(parametric_var(w, cov, 0.01), -normal_ppf(0.01) * sigma, 1e-9);
    const MonteCarloResult mc = monte_carlo_var(w, cov, 0.01, 100'000, Dist::Normal, 6.0, 5);
    EXPECT_NEAR(mc.var, -normal_ppf(0.01) * sigma, 3.0 * mc.var_se);
    EXPECT_GE(mc.es, mc.var);
    // Short single-asset exposure has identical risk (symmetry of the normal).
    const std::vector<double> ws = {-w0};
    EXPECT_NEAR(parametric_var(ws, cov, 0.01), parametric_var(w, cov, 0.01), 1e-9);
}

TEST(EdgeCases, AlphaBoundsRejectedEverywhere) {
    std::vector<double> pnl(100);
    for (int t = 0; t < 100; ++t) pnl[t] = std::sin(1.3 * t) * 100.0;
    const Matrix cov(1, 1, {1e-4});
    const std::vector<double> w = {1.0e6};
    for (double bad : {0.0, -0.01, 0.5, 0.75, 1.0}) {
        EXPECT_THROW(historical_var(pnl, bad), std::invalid_argument) << bad;
        EXPECT_THROW(age_weighted_var(pnl, bad), std::invalid_argument) << bad;
        EXPECT_THROW(filtered_historical_var(pnl, bad), std::invalid_argument) << bad;
        EXPECT_THROW(expected_shortfall(pnl, bad), std::invalid_argument) << bad;
        EXPECT_THROW(parametric_var(w, cov, bad), std::invalid_argument) << bad;
        EXPECT_THROW(normal_es(1.0, bad), std::invalid_argument) << bad;
        EXPECT_THROW(cornish_fisher_var(1.0, bad), std::invalid_argument) << bad;
    }
    // Alpha just inside the boundary must work.
    EXPECT_NO_THROW(historical_var(pnl, 0.499));
    EXPECT_NO_THROW(historical_var(pnl, 0.011));
}

TEST(EdgeCases, ZeroVarianceAssetHandledThroughout) {
    // Asset 2 is riskless: covariance is singular but PSD.
    const Matrix cov(2, 2, {4.0e-4, 0.0, 0.0, 0.0});
    const std::vector<double> w = {1.0e6, 5.0e5};
    // Parametric: the riskless leg contributes nothing.
    const double sigma = portfolio_sigma(w, cov);
    EXPECT_NEAR(sigma, 1.0e6 * 0.02, 1e-9);
    // Cholesky must succeed via the jitter fallback and MC must stay close to
    // the parametric answer (the jitter is ~1e-10 of the mean variance).
    const CholeskyResult ch = cholesky(cov);
    EXPECT_GT(ch.jitter_added, 0.0);
    const MonteCarloResult mc = monte_carlo_var(w, cov, 0.01, 100'000, Dist::Normal, 6.0, 17);
    EXPECT_NEAR(mc.var, -normal_ppf(0.01) * sigma, 3.0 * mc.var_se);
}

TEST(EdgeCases, FullyRisklessPortfolio) {
    // All-zero covariance: sigma = 0, parametric VaR = 0 exactly.
    const Matrix cov(2, 2, 0.0);
    const std::vector<double> w = {1.0e6, -1.0e6};
    EXPECT_DOUBLE_EQ(portfolio_sigma(w, cov), 0.0);
    EXPECT_DOUBLE_EQ(parametric_var(w, cov, 0.01), 0.0);
    EXPECT_DOUBLE_EQ(normal_es(0.0, 0.01), 0.0);
}

TEST(EdgeCases, ConstantPnlSeries) {
    // A constant P&L series has an empirical quantile equal to the constant;
    // historical VaR of a constant profit is a negative "loss" (a gain floor).
    std::vector<double> flat(100, 25.0);
    EXPECT_DOUBLE_EQ(historical_var(flat, 0.05), -25.0);
    EXPECT_DOUBLE_EQ(expected_shortfall(flat, 0.05), -25.0);
    // Filtered HS on zero P&L: vol floor prevents 0/0, result is 0.
    std::vector<double> zeros(100, 0.0);
    EXPECT_DOUBLE_EQ(filtered_historical_var(zeros, 0.05), 0.0);
    // Zero-variance moments are defined as 0, not NaN.
    EXPECT_DOUBLE_EQ(skewness(flat), 0.0);
    EXPECT_DOUBLE_EQ(excess_kurtosis(flat), 0.0);
}

TEST(EdgeCases, NonFinitePnlRejected) {
    std::vector<double> pnl(100, 1.0);
    pnl[3] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(historical_var(pnl, 0.05), std::invalid_argument);
    pnl[3] = std::nan("");
    EXPECT_THROW(expected_shortfall(pnl, 0.05), std::invalid_argument);
    EXPECT_THROW(age_weighted_var(pnl, 0.05), std::invalid_argument);
}

TEST(EdgeCases, ReturnsValidation) {
    EXPECT_THROW(log_returns(std::vector<double>{100.0}), std::invalid_argument);
    EXPECT_THROW(log_returns(std::vector<double>{100.0, -5.0}), std::invalid_argument);
    EXPECT_THROW(log_returns(std::vector<double>{100.0, 0.0}), std::invalid_argument);
    const std::vector<double> prices = {100.0, 110.0, 99.0};
    const std::vector<double> lr = log_returns(prices);
    const std::vector<double> sr = simple_returns(prices);
    ASSERT_EQ(lr.size(), 2u);
    EXPECT_NEAR(lr[0], std::log(1.1), 1e-15);
    EXPECT_NEAR(sr[0], 0.1, 1e-15);
    EXPECT_NEAR(sr[1], 99.0 / 110.0 - 1.0, 1e-15);
    // log return <= simple return (Jensen), equality only at 0.
    EXPECT_LT(lr[0], sr[0]);
}

TEST(EdgeCases, PortfolioPnlShapeMismatch) {
    const Matrix panel(10, 3, 0.01);
    EXPECT_THROW(portfolio_pnl(panel, std::vector<double>{1.0, 2.0}), std::invalid_argument);
    const std::vector<double> ok = {1.0, 2.0, 3.0};
    EXPECT_NO_THROW(portfolio_pnl(panel, ok));
    const std::vector<double> pnl = portfolio_pnl(panel, ok);
    ASSERT_EQ(pnl.size(), 10u);
    EXPECT_NEAR(pnl[0], 0.06, 1e-15);
}

TEST(EdgeCases, MinObsGuardExactBoundary) {
    // Exactly kMinHistObs observations pass; one fewer throws.
    std::vector<double> ok(kMinHistObs);
    for (std::size_t t = 0; t < ok.size(); ++t) ok[t] = std::sin(0.9 * static_cast<double>(t));
    EXPECT_NO_THROW(historical_var(ok, 0.05));
    std::vector<double> short_series(ok.begin(), ok.end() - 1);
    EXPECT_THROW(historical_var(short_series, 0.05), std::invalid_argument);
}

TEST(EdgeCases, DeepTailAlphaOnSmallSample) {
    // alpha far below 1/n: the type-7 quantile interpolates just above the
    // sample minimum (h = (n-1) alpha < 1) — the tail is unresolvable and the
    // estimate is pinned to the worst observed loss, never extrapolated
    // beyond it.  Documented failure mode (docs/VALIDATION.md).
    std::vector<double> pnl(100);
    for (int t = 0; t < 100; ++t) pnl[t] = 10.0 * std::sin(2.3 * t) - (t == 50 ? 500.0 : 0.0);
    const double v = historical_var(pnl, 1e-6);
    std::vector<double> sorted = pnl;
    std::sort(sorted.begin(), sorted.end());
    const double h = 99.0 * 1e-6;  // (n-1) q, entirely inside the first gap
    const double expected = -(sorted[0] + h * (sorted[1] - sorted[0]));
    EXPECT_NEAR(v, expected, 1e-12 * std::abs(expected));
    EXPECT_LE(v, -sorted.front());   // never exceeds the worst observed loss
    EXPECT_GT(v, -sorted[1]);        // ... but stays pinned to it
}
