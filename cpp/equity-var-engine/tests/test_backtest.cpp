// Backtesting: Kupiec LR hand-computed, chi2 p-values, Christoffersen
// clustering detection, Basel traffic-light exact zone boundaries.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "eqvar/backtest.hpp"
#include "eqvar/stats.hpp"

using namespace eqvar;

TEST(Exceptions, IndicatorAndBroadcast) {
    const std::vector<double> pnl = {-120.0, 50.0, -99.9, -100.1, 0.0};
    const std::vector<double> var1 = {100.0};  // scalar broadcast
    const std::vector<std::uint8_t> ex = exceptions_from_pnl(pnl, var1);
    const std::vector<std::uint8_t> expected = {1, 0, 0, 1, 0};
    EXPECT_EQ(ex, expected);
    const std::vector<double> vard = {130.0, 100.0, 90.0, 100.0, 100.0};
    EXPECT_EQ(exceptions_from_pnl(pnl, vard), (std::vector<std::uint8_t>{0, 0, 1, 1, 0}));
    EXPECT_THROW(exceptions_from_pnl(pnl, std::vector<double>{-1.0}), std::invalid_argument);
    EXPECT_THROW(exceptions_from_pnl(pnl, std::vector<double>{1.0, 2.0}), std::invalid_argument);
}

TEST(Kupiec, HandComputedLrTo1e12) {
    // T = 250, x = 5, p = 0.01: LR computed from first principles here.
    const double T = 250.0, x = 5.0, p = 0.01, pihat = x / T;
    const double ll0 = (T - x) * std::log(1.0 - p) + x * std::log(p);
    const double ll1 = (T - x) * std::log(1.0 - pihat) + x * std::log(pihat);
    const double lr_hand = -2.0 * (ll0 - ll1);
    const KupiecResult r = kupiec_pof(250, 5, 0.01);
    EXPECT_NEAR(r.lr, lr_hand, 1e-12);
    EXPECT_NEAR(r.pvalue, chi2_sf(lr_hand, 1.0), 1e-14);
    EXPECT_DOUBLE_EQ(r.expected, 2.5);
    EXPECT_DOUBLE_EQ(r.rate, 0.02);
}

TEST(Kupiec, Chi2CriticalValueGivesPointOhFive) {
    // Find the exception count whose LR brackets 3.8415 and confirm the
    // p-value machinery: chi2(1) sf at the critical value is 0.05 to 1e-4.
    EXPECT_NEAR(chi2_sf(3.841458820694124, 1.0), 0.05, 1e-4);
    // Exact-coverage sample: x/T == alpha -> LR = 0, p = 1.
    const KupiecResult exact = kupiec_pof(200, 2, 0.01);
    EXPECT_NEAR(exact.lr, 0.0, 1e-12);
    EXPECT_DOUBLE_EQ(exact.pvalue, 1.0);
}

TEST(Kupiec, DegenerateCountsAndMonotonicity) {
    // Zero exceptions on 250 days at 1 %: LR = -2 * 250 ln(0.99).
    const KupiecResult zero = kupiec_pof(250, 0, 0.01);
    EXPECT_NEAR(zero.lr, -2.0 * 250.0 * std::log(0.99), 1e-12);
    // More excess exceptions -> larger LR.
    EXPECT_GT(kupiec_pof(250, 10, 0.01).lr, kupiec_pof(250, 6, 0.01).lr);
    // A materially bad model is rejected at 5 %.
    EXPECT_LT(kupiec_pof(250, 10, 0.01).pvalue, 0.05);
    EXPECT_THROW(kupiec_pof(0, 0, 0.01), std::invalid_argument);
    EXPECT_THROW(kupiec_pof(250, 251, 0.01), std::invalid_argument);
    EXPECT_THROW(kupiec_pof(250, -1, 0.01), std::invalid_argument);
}

TEST(Christoffersen, TransitionCountsOnTinyPattern) {
    const std::vector<std::uint8_t> ex = {0, 1, 1, 0, 1};
    const ChristoffersenResult r = christoffersen_independence(ex);
    EXPECT_DOUBLE_EQ(r.n00, 0.0);
    EXPECT_DOUBLE_EQ(r.n01, 2.0);
    EXPECT_DOUBLE_EQ(r.n10, 1.0);
    EXPECT_DOUBLE_EQ(r.n11, 1.0);
    EXPECT_DOUBLE_EQ(r.pi01, 1.0);
    EXPECT_DOUBLE_EQ(r.pi11, 0.5);
}

TEST(Christoffersen, DetectsPlantedClustering) {
    // 250 days, 10 exceptions: (a) one solid run of 10 -> heavy clustering,
    // (b) evenly spread every 25 days -> independent-looking.
    std::vector<std::uint8_t> clustered(250, 0), spread(250, 0);
    for (int t = 100; t < 110; ++t) clustered[t] = 1;
    for (int t = 12; t < 250; t += 25) spread[t] = 1;
    const ChristoffersenResult c = christoffersen_independence(clustered);
    const ChristoffersenResult s = christoffersen_independence(spread);
    EXPECT_LT(c.pvalue, 0.001) << "a run of 10 exceptions must reject independence";
    EXPECT_GT(c.pi11, c.pi01);  // exception begets exception in the cluster
    EXPECT_GT(s.pvalue, 0.10) << "evenly spread exceptions must not reject";
    EXPECT_GT(c.lr, s.lr);
    EXPECT_THROW(christoffersen_independence(std::vector<std::uint8_t>{1}),
                 std::invalid_argument);
}

TEST(Christoffersen, ConditionalCoverageIsSumOfComponents) {
    std::vector<std::uint8_t> ex(250, 0);
    for (int t = 100; t < 108; ++t) ex[t] = 1;
    const KupiecResult uc = kupiec_pof(250, 8, 0.01);
    const ChristoffersenResult ind = christoffersen_independence(ex);
    const ConditionalCoverageResult cc = christoffersen_cc(ex, 0.01);
    EXPECT_NEAR(cc.lr, uc.lr + ind.lr, 1e-12);
    EXPECT_NEAR(cc.lr_uc, uc.lr, 1e-12);
    EXPECT_NEAR(cc.lr_ind, ind.lr, 1e-12);
    EXPECT_NEAR(cc.pvalue, chi2_sf(cc.lr, 2.0), 1e-14);
    EXPECT_LT(cc.pvalue, 0.05);  // wrong rate AND clustered: joint rejection
}

TEST(Basel, ExactZoneBoundariesAt250Obs) {
    // Green: 0-4 exceptions, multiplier 3.0.
    EXPECT_EQ(basel_traffic_light(0).zone, BaselZone::Green);
    EXPECT_EQ(basel_traffic_light(4).zone, BaselZone::Green);
    EXPECT_DOUBLE_EQ(basel_traffic_light(4).multiplier, 3.0);
    // Yellow: 5-9 with the regulatory add-on ladder.
    EXPECT_EQ(basel_traffic_light(5).zone, BaselZone::Yellow);
    EXPECT_DOUBLE_EQ(basel_traffic_light(5).multiplier, 3.40);
    EXPECT_DOUBLE_EQ(basel_traffic_light(6).multiplier, 3.50);
    EXPECT_DOUBLE_EQ(basel_traffic_light(7).multiplier, 3.65);
    EXPECT_DOUBLE_EQ(basel_traffic_light(8).multiplier, 3.75);
    EXPECT_EQ(basel_traffic_light(9).zone, BaselZone::Yellow);
    EXPECT_DOUBLE_EQ(basel_traffic_light(9).multiplier, 3.85);
    // Red: 10 or more, multiplier 4.0.
    EXPECT_EQ(basel_traffic_light(10).zone, BaselZone::Red);
    EXPECT_DOUBLE_EQ(basel_traffic_light(10).multiplier, 4.0);
    EXPECT_EQ(basel_traffic_light(15).zone, BaselZone::Red);
    EXPECT_THROW(basel_traffic_light(-1), std::invalid_argument);
}

TEST(Basel, BinomialZoneProbabilities) {
    // Under a correct 99 % model the green zone covers ~89.2 % of outcomes
    // and 9 exceptions is already the 99.97th percentile (scipy reference).
    EXPECT_NEAR(basel_traffic_light(4).cumulative_prob, 0.89218762690362508, 1e-10);
    EXPECT_NEAR(basel_traffic_light(9).cumulative_prob, 0.99974980993125950, 1e-10);
    EXPECT_EQ(to_string(BaselZone::Green), "green");
    EXPECT_EQ(to_string(BaselZone::Yellow), "yellow");
    EXPECT_EQ(to_string(BaselZone::Red), "red");
}
