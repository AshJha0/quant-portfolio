// Monte Carlo: statistical agreement with Black-Scholes, variance-reduction
// effectiveness, bit-reproducibility of the single-threaded core, and
// deterministic seed-partitioned multithreading.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "eqopt/black_scholes.hpp"
#include "eqopt/monte_carlo.hpp"

namespace {

using eqopt::bs_price;
using eqopt::mc_price;
using eqopt::MCResult;
using eqopt::OptionType;

constexpr double kS = 100.0, kK = 105.0, kT = 1.0, kR = 0.04, kQ = 0.015,
                 kSigma = 0.25;

TEST(MonteCarlo, Within3StandardErrorsOfBlackScholes) {
    for (OptionType type : {OptionType::Call, OptionType::Put}) {
        const double bs = bs_price(kS, kK, kT, kR, kSigma, kQ, type);
        const MCResult mc = mc_price(kS, kK, kT, kR, kSigma, kQ, type,
                                     200000, true, true, 42);
        EXPECT_GT(mc.std_error, 0.0);
        EXPECT_NEAR(mc.price, bs, 3.0 * mc.std_error)
            << (type == OptionType::Call ? "call" : "put");
    }
}

TEST(MonteCarlo, StdErrorScalesAsInverseSqrtNFitted) {
    // Plain (no variance reduction) MC has statistical error O(1/sqrt(n))
    // by the CLT. Rather than trust the estimator's own reported
    // std_error formula, measure the *empirical* spread of independent
    // replications of the price at each path count and fit the exponent
    // of empirical_std vs n by log-log regression -- an actual
    // measurement of the realized rate, not a single eyeballed ratio.
    const std::vector<std::size_t> path_counts{2000,  4000,  8000,
                                               16000, 32000, 64000};
    constexpr int kReps = 40;
    std::vector<double> log_n, log_std;
    for (std::size_t n : path_counts) {
        std::vector<double> reps;
        reps.reserve(kReps);
        for (int rep = 0; rep < kReps; ++rep) {
            const std::uint64_t seed =
                1000003ULL * static_cast<std::uint64_t>(n) +
                static_cast<std::uint64_t>(rep);
            reps.push_back(mc_price(kS, kK, kT, kR, kSigma, kQ,
                                    OptionType::Call, n, false, false, seed)
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
    EXPECT_GT(slope, -0.65) << "fitted MC exponent " << slope
                            << "; CLT predicts -0.5 (std ~ C/sqrt(n))";
    EXPECT_LT(slope, -0.35) << "fitted MC exponent " << slope
                            << "; CLT predicts -0.5 (std ~ C/sqrt(n))";
}

TEST(MonteCarlo, PlainEstimatorAlsoUnbiased) {
    const double bs = bs_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call);
    const MCResult mc = mc_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call,
                                 200000, false, false, 7);
    EXPECT_NEAR(mc.price, bs, 3.5 * mc.std_error);
}

TEST(MonteCarlo, VarianceReductionIsEffective) {
    const MCResult plain = mc_price(kS, kK, kT, kR, kSigma, kQ,
                                    OptionType::Call, 100000, false, false, 1);
    const MCResult anti = mc_price(kS, kK, kT, kR, kSigma, kQ,
                                   OptionType::Call, 100000, true, false, 1);
    const MCResult cv = mc_price(kS, kK, kT, kR, kSigma, kQ,
                                 OptionType::Call, 100000, false, true, 1);
    const MCResult both = mc_price(kS, kK, kT, kR, kSigma, kQ,
                                   OptionType::Call, 100000, true, true, 1);
    EXPECT_LT(anti.std_error, plain.std_error);
    EXPECT_LT(cv.std_error, plain.std_error);
    EXPECT_LT(both.std_error, plain.std_error);
    // Control variate alone should cut the SE by well over half for a
    // vanilla call (payoff strongly correlated with S_T).
    EXPECT_LT(cv.std_error, 0.5 * plain.std_error);
}

TEST(MonteCarlo, SingleThreadIsBitReproducible) {
    const MCResult a = mc_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call,
                                50000, true, true, 12345, 1);
    const MCResult b = mc_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call,
                                50000, true, true, 12345, 1);
    EXPECT_TRUE(a == b);  // exact equality of every field
    EXPECT_DOUBLE_EQ(a.price, b.price);
    EXPECT_DOUBLE_EQ(a.std_error, b.std_error);
}

TEST(MonteCarlo, DifferentSeedsGiveDifferentDraws) {
    const MCResult a = mc_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call,
                                50000, true, true, 1, 1);
    const MCResult b = mc_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call,
                                50000, true, true, 2, 1);
    EXPECT_NE(a.price, b.price);
    // ...but both are consistent with the analytic price.
    const double bs = bs_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call);
    EXPECT_NEAR(a.price, bs, 4.0 * a.std_error);
    EXPECT_NEAR(b.price, bs, 4.0 * b.std_error);
}

TEST(MonteCarlo, MultithreadedPathIsDeterministicGivenSeedAndThreads) {
    const MCResult a = mc_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call,
                                100000, true, true, 99, 4);
    const MCResult b = mc_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call,
                                100000, true, true, 99, 4);
    EXPECT_TRUE(a == b);
}

TEST(MonteCarlo, MultithreadedStatisticallyConsistentWithSingleThread) {
    const double bs = bs_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call);
    const MCResult mt = mc_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call,
                                 200000, true, true, 42, 4);
    const MCResult st = mc_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call,
                                 200000, true, true, 42, 1);
    // Different RNG streams -> not bit-equal, but both unbiased.
    EXPECT_NEAR(mt.price, bs, 3.5 * mt.std_error);
    EXPECT_NEAR(st.price, bs, 3.5 * st.std_error);
    EXPECT_NEAR(mt.price, st.price,
                3.5 * std::sqrt(mt.std_error * mt.std_error +
                                st.std_error * st.std_error));
}

TEST(MonteCarlo, ConfidenceIntervalAndContains) {
    const MCResult mc = mc_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call,
                                 100000);
    EXPECT_LT(mc.ci_low, mc.price);
    EXPECT_GT(mc.ci_high, mc.price);
    EXPECT_NEAR(mc.ci_high - mc.ci_low, 2 * 1.959963984540054 * mc.std_error,
                1e-12);
    EXPECT_TRUE(mc.contains(mc.price));
    EXPECT_EQ(mc.n_paths, 100000);
}

TEST(MonteCarlo, DeterministicLimitsAreExact) {
    // T == 0 and sigma == 0 short-circuit to the exact BS value, SE = 0.
    const MCResult expired = mc_price(110.0, 100.0, 0.0, 0.05, 0.2, 0.0,
                                      OptionType::Call, 1000);
    EXPECT_DOUBLE_EQ(expired.price, 10.0);
    EXPECT_DOUBLE_EQ(expired.std_error, 0.0);
    const MCResult zero_vol = mc_price(100.0, 95.0, 1.0, 0.05, 0.0, 0.02,
                                       OptionType::Call, 1000);
    EXPECT_DOUBLE_EQ(zero_vol.price,
                     bs_price(100.0, 95.0, 1.0, 0.05, 0.0, 0.02,
                              OptionType::Call));
    EXPECT_DOUBLE_EQ(zero_vol.std_error, 0.0);
}

TEST(MonteCarlo, OddPathCountRoundsUpUnderAntithetic) {
    // 101 paths with antithetic pairing -> 51 pairs -> 102 effective paths,
    // as documented in the header.
    const MCResult mc = mc_price(kS, kK, kT, kR, kSigma, kQ, OptionType::Call,
                                 101, true, false, 3);
    EXPECT_EQ(mc.n_paths, 102u);
    const MCResult plain = mc_price(kS, kK, kT, kR, kSigma, kQ,
                                    OptionType::Call, 101, false, false, 3);
    EXPECT_EQ(plain.n_paths, 101u);
}

TEST(MonteCarlo, InvalidInputsThrow) {
    EXPECT_THROW(mc_price(-1.0, 100.0, 1.0, 0.05, 0.2), std::invalid_argument);
    // Non-finite rate must be rejected up front, not simulated into NaNs.
    EXPECT_THROW(mc_price(100.0, 100.0, 1.0,
                          std::numeric_limits<double>::infinity(), 0.2),
                 std::invalid_argument);
    EXPECT_THROW(mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0,
                          OptionType::Call, 1),
                 std::invalid_argument);
    EXPECT_THROW(mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0,
                          OptionType::Call, 1000, true, true, 42, 0),
                 std::invalid_argument);
}

}  // namespace
