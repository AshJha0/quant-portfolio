// Monte Carlo: 3-standard-error agreement with GK, variance-reduction
// effectiveness, seed determinism, and validation.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/monte_carlo.hpp"

namespace {

using namespace fxopt;

constexpr double S = 1.10, K = 1.12, T = 0.5, RD = 0.0425, RF = 0.0290,
                 SIG = 0.0925;

TEST(MonteCarlo, PriceWithinThreeStandardErrors) {
    for (const auto type : {OptionType::Call, OptionType::Put}) {
        const double analytic = gk_price(S, K, T, RD, RF, SIG, type);
        const MCResult r = mc_price(S, K, T, RD, RF, SIG, type, 200'000, 7);
        EXPECT_NEAR(r.price, analytic, 3.0 * r.std_error)
            << "type " << static_cast<int>(type);
        EXPECT_GT(r.std_error, 0.0);
        EXPECT_LT(r.ci_low, r.price);
        EXPECT_GT(r.ci_high, r.price);
        EXPECT_NEAR(r.ci_high - r.ci_low, 2.0 * 1.96 * r.std_error, 1e-15);
    }
}

TEST(MonteCarlo, StdErrorScalesAsInverseSqrtNFitted) {
    // Plain (no variance reduction) MC has statistical error O(1/sqrt(n))
    // by the CLT. Rather than trust the estimator's own reported
    // std_error formula, measure the *empirical* spread of independent
    // replications of the price at each path count and fit the exponent
    // of empirical_std vs n by log-log regression -- an actual
    // measurement of the realized rate, not a single eyeballed ratio.
    const std::vector<std::int64_t> path_counts{2000,  4000,  8000,
                                                16000, 32000, 64000};
    constexpr int kReps = 40;
    std::vector<double> log_n, log_std;
    for (std::int64_t n : path_counts) {
        std::vector<double> reps;
        reps.reserve(kReps);
        for (int rep = 0; rep < kReps; ++rep) {
            const std::uint64_t seed =
                1000003ULL * static_cast<std::uint64_t>(n) +
                static_cast<std::uint64_t>(rep);
            reps.push_back(mc_price(S, K, T, RD, RF, SIG, OptionType::Call,
                                    n, seed, false, false)
                              .price);
        }
        const double mean =
            std::accumulate(reps.begin(), reps.end(), 0.0) / reps.size();
        double var = 0.0;
        for (double v : reps) var += (v - mean) * (v - mean);
        var /= static_cast<double>(reps.size() - 1);
        log_n.push_back(std::log(static_cast<double>(n)));
        log_std.push_back(0.5 * std::log(var));
    }
    const double mean_x =
        std::accumulate(log_n.begin(), log_n.end(), 0.0) / log_n.size();
    const double mean_y =
        std::accumulate(log_std.begin(), log_std.end(), 0.0) / log_std.size();
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < log_n.size(); ++i) {
        num += (log_n[i] - mean_x) * (log_std[i] - mean_y);
        den += (log_n[i] - mean_x) * (log_n[i] - mean_x);
    }
    const double slope = num / den;
    EXPECT_GT(slope, -0.65) << "fitted MC exponent " << slope;
    EXPECT_LT(slope, -0.35) << "fitted MC exponent " << slope;
}

TEST(MonteCarlo, VarianceReductionShrinksStandardError) {
    const MCResult plain = mc_price(S, K, T, RD, RF, SIG, OptionType::Call,
                                    100'000, 11, false, false);
    const MCResult anti = mc_price(S, K, T, RD, RF, SIG, OptionType::Call,
                                   100'000, 11, true, false);
    const MCResult full = mc_price(S, K, T, RD, RF, SIG, OptionType::Call,
                                   100'000, 11, true, true);
    EXPECT_LT(anti.std_error, plain.std_error);
    EXPECT_LT(full.std_error, anti.std_error);
    EXPECT_EQ(plain.method, "plain");
    EXPECT_EQ(full.method, "antithetic+control_variate");
}

TEST(MonteCarlo, SeedDeterminism) {
    const MCResult a = mc_price(S, K, T, RD, RF, SIG, OptionType::Call,
                                50'000, 42);
    const MCResult b = mc_price(S, K, T, RD, RF, SIG, OptionType::Call,
                                50'000, 42);
    const MCResult c = mc_price(S, K, T, RD, RF, SIG, OptionType::Call,
                                50'000, 43);
    EXPECT_EQ(a.price, b.price);  // bit-identical
    EXPECT_EQ(a.std_error, b.std_error);
    EXPECT_NE(a.price, c.price);
}

TEST(MonteCarlo, AntitheticPathCountRoundsUpToEven) {
    const MCResult r = mc_price(S, K, T, RD, RF, SIG, OptionType::Call,
                                10'001, 1, true, false);
    EXPECT_EQ(r.n_paths, 10'002);
}

TEST(MonteCarlo, ThrowsOnBadInputs) {
    EXPECT_THROW(mc_price(S, K, 0.0, RD, RF, SIG, OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(mc_price(S, K, T, RD, RF, SIG, OptionType::Call, 1),
                 std::invalid_argument);
    EXPECT_THROW(mc_price(-1.0, K, T, RD, RF, SIG, OptionType::Call),
                 std::invalid_argument);
}

}  // namespace
