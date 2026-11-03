// Historical-simulation VaR: hand-exact quantiles on tiny arrays, BRW
// weights, EWMA filtering, sqrt-time scaling, input validation.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include "eqvar/historical.hpp"

using namespace eqvar;

namespace {

// 100-day deterministic P&L series with a fat left tail.
std::vector<double> demo_pnl() {
    std::vector<double> pnl(100);
    for (int t = 0; t < 100; ++t) {
        pnl[t] = 100.0 * std::sin(0.7 * t + 1.0) - (t % 17 == 0 ? 150.0 : 0.0);
    }
    return pnl;
}

}  // namespace

TEST(HistoricalVar, IsOrderStatisticAndIsMonotoneInAlpha) {
    const std::vector<double> pnl = demo_pnl();
    // Order statistic: losses sorted descending, m = ceil(n alpha) -> L(m).
    // The old code returned the type-7 interpolant, which is not an order
    // statistic for non-integer (n-1) alpha, so this assertion failed on it.
    std::vector<double> loss;
    for (double v : pnl) loss.push_back(-v);
    std::sort(loss.begin(), loss.end(), std::greater<double>());
    const std::size_t m =
        static_cast<std::size_t>(std::ceil(0.05 * static_cast<double>(pnl.size()) - 1e-9));
    EXPECT_EQ(historical_var(pnl, 0.05), loss[m - 1]);
    // Deeper tail => larger VaR.
    EXPECT_GE(historical_var(pnl, 0.01), historical_var(pnl, 0.05));
    EXPECT_GE(historical_var(pnl, 0.05), historical_var(pnl, 0.10));
}

TEST(HistoricalVar, Validation) {
    std::vector<double> tiny(kMinHistObs - 1, 0.0);
    EXPECT_THROW(historical_var(tiny, 0.01), std::invalid_argument);
    const std::vector<double> pnl = demo_pnl();
    EXPECT_THROW(historical_var(pnl, 0.0), std::invalid_argument);
    EXPECT_THROW(historical_var(pnl, 0.5), std::invalid_argument);
    std::vector<double> bad = pnl;
    bad[10] = std::nan("");
    EXPECT_THROW(historical_var(bad, 0.01), std::invalid_argument);
}

TEST(BrwWeights, SumToOneAndMonotoneInRecency) {
    for (std::size_t n : {std::size_t{1}, std::size_t{5}, std::size_t{250}}) {
        const std::vector<double> w = brw_weights(n, 0.98);
        ASSERT_EQ(w.size(), n);
        const double sum = std::accumulate(w.begin(), w.end(), 0.0);
        EXPECT_NEAR(sum, 1.0, 1e-12) << "n=" << n;
        for (std::size_t i = 1; i < n; ++i) {
            EXPECT_GT(w[i], w[i - 1]) << "weights must increase with recency";
        }
    }
    // Exact ratio between consecutive weights is 1/lam.
    const std::vector<double> w = brw_weights(10, 0.95);
    EXPECT_NEAR(w[9] / w[8], 1.0 / 0.95, 1e-12);
    EXPECT_THROW(brw_weights(10, 1.0), std::invalid_argument);
    EXPECT_THROW(brw_weights(0, 0.98), std::invalid_argument);
}

TEST(AgeWeightedVar, LambdaNearOneRecoversStepHistorical) {
    const std::vector<double> pnl = demo_pnl();
    // lam -> 1: BRW converges to the equal-weight step-CDF inversion.  Use an
    // alpha safely between the k/n weight boundaries (0.05, 0.06) so the tiny
    // residual age tilt cannot flip the selected order statistic.
    const double brw = age_weighted_var(pnl, 0.053, 0.999999);
    std::vector<double> sorted = pnl;
    std::sort(sorted.begin(), sorted.end());
    // Step inversion at alpha = 0.053 with n = 100 picks the 6th order stat.
    EXPECT_DOUBLE_EQ(brw, -sorted[5]);
}

TEST(AgeWeightedVar, RecentCrashDominatesUnderLowLambda) {
    // A large loss on the most recent day must raise BRW VaR far more than
    // the same loss buried at the start of the window.
    std::vector<double> recent(100, 10.0), old(100, 10.0);
    recent.back() = -500.0;
    old.front() = -500.0;
    const double var_recent = age_weighted_var(recent, 0.05, 0.94);
    const double var_old = age_weighted_var(old, 0.05, 0.94);
    EXPECT_GT(var_recent, var_old);
    EXPECT_DOUBLE_EQ(var_recent, 500.0);  // crash weight (1-l)/(1-l^n) > 5 %
}

TEST(EwmaVolatility, NoLookAheadAndRiskMetricsRecursion) {
    const std::vector<double> x = {1.0, -2.0, 3.0, -1.0, 2.0};
    const double lam = 0.94;
    // Seed window 3: s2_0 = mean(x[0..3)^2) = (1 + 4 + 9) / 3. The old code
    // seeded with the full-sample variance (look-ahead) and failed here.
    const std::vector<double> sig = ewma_volatility(x, lam, 3);
    ASSERT_EQ(sig.size(), x.size());
    double s2 = 14.0 / 3.0;
    EXPECT_NEAR(sig[0], std::sqrt(s2), 1e-14);
    for (std::size_t t = 1; t < x.size(); ++t) {
        s2 = lam * s2 + (1.0 - lam) * x[t - 1] * x[t - 1];
        EXPECT_NEAR(sig[t], std::sqrt(s2), 1e-14) << "t=" << t;
    }
    const double next = lam * s2 + (1.0 - lam) * x[4] * x[4];
    EXPECT_NEAR(ewma_volatility_next(x, lam, 3), std::sqrt(next), 1e-14);
    // Default seed window min(30, n) = 5 here: mean of all five squares.
    EXPECT_NEAR(ewma_volatility(x, lam)[0], std::sqrt(19.0 / 5.0), 1e-14);
    // Changing an observation after the seed window cannot change sigma[0].
    std::vector<double> y = x;
    y[4] = 100.0;
    EXPECT_EQ(ewma_volatility(y, lam, 3)[0], sig[0]);
    EXPECT_THROW(ewma_volatility(std::vector<double>{1.0}, lam), std::invalid_argument);
    EXPECT_THROW(ewma_volatility(x, 0.0), std::invalid_argument);
}

TEST(FilteredHistoricalVar, ScaleEquivariantAndRegimeResponsive) {
    const std::vector<double> pnl = demo_pnl();
    // Positive homogeneity: FHS(c pnl) = c FHS(pnl).
    std::vector<double> scaled = pnl;
    for (double& v : scaled) v *= 3.0;
    EXPECT_NEAR(filtered_historical_var(scaled, 0.05), 3.0 * filtered_historical_var(pnl, 0.05),
                1e-9);
    // A quiet recent regime must cut FHS VaR below plain historical VaR.
    std::vector<double> calm = pnl;
    for (std::size_t t = 60; t < calm.size(); ++t) calm[t] *= 0.05;
    EXPECT_LT(filtered_historical_var(calm, 0.05), historical_var(calm, 0.05));
}

TEST(SqrtTime, ScalingAndValidation) {
    EXPECT_DOUBLE_EQ(scale_var_sqrt_time(100.0, 1), 100.0);
    EXPECT_DOUBLE_EQ(scale_var_sqrt_time(100.0, 4), 200.0);
    EXPECT_NEAR(scale_var_sqrt_time(50.0, 10), 50.0 * std::sqrt(10.0), 1e-12);
    EXPECT_THROW(scale_var_sqrt_time(100.0, 0), std::invalid_argument);
}
