// CRR binomial tree: convergence to Black-Scholes, American/European
// ordering, early-exercise identities, edge cases and input validation.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "eqopt/binomial.hpp"
#include "eqopt/black_scholes.hpp"

namespace {

using eqopt::bs_price;
using eqopt::crr_price;
using eqopt::early_exercise_premium;
using eqopt::ExerciseStyle;
using eqopt::OptionType;

TEST(BinomialConvergence, EuropeanWithin2e3At2000Steps) {
    const double S = 100.0, K = 100.0, T = 1.0, r = 0.05, q = 0.02,
                 sigma = 0.2;
    for (OptionType type : {OptionType::Call, OptionType::Put}) {
        const double bs = bs_price(S, K, T, r, sigma, q, type);
        const double tree = crr_price(S, K, T, r, sigma, q, type,
                                      ExerciseStyle::European, 2000);
        EXPECT_NEAR(tree, bs, 2e-3);
    }
}

TEST(BinomialConvergence, ErrorShrinksWithSteps) {
    const double S = 100.0, K = 110.0, T = 0.75, r = 0.03, q = 0.01,
                 sigma = 0.25;
    const double bs = bs_price(S, K, T, r, sigma, q, OptionType::Call);
    const auto err = [&](int n) {
        return std::abs(crr_price(S, K, T, r, sigma, q, OptionType::Call,
                                  ExerciseStyle::European, n) -
                        bs);
    };
    // Average consecutive odd/even trees to kill the oscillating term, then
    // require monotone decay across decades of n.
    const auto smooth_err = [&](int n) { return 0.5 * (err(n) + err(n + 1)); };
    const double e50 = smooth_err(50);
    const double e200 = smooth_err(200);
    const double e800 = smooth_err(800);
    const double e2000 = smooth_err(2000);
    EXPECT_LT(e200, e50);
    EXPECT_LT(e800, e200);
    EXPECT_LT(e2000, e800);
    EXPECT_LT(e2000, 2e-3);
}

TEST(BinomialAmerican, AmericanAtLeastEuropeanEverywhere) {
    const std::vector<double> strikes{80.0, 100.0, 120.0};
    const std::vector<double> vols{0.1, 0.3};
    const std::vector<double> divs{0.0, 0.04};
    for (double K : strikes)
        for (double sigma : vols)
            for (double q : divs)
                for (OptionType type : {OptionType::Call, OptionType::Put}) {
                    const double amer = crr_price(100.0, K, 1.0, 0.05, sigma,
                                                  q, type,
                                                  ExerciseStyle::American, 400);
                    const double euro = crr_price(100.0, K, 1.0, 0.05, sigma,
                                                  q, type,
                                                  ExerciseStyle::European, 400);
                    EXPECT_GE(amer, euro - 1e-12)
                        << "K=" << K << " sigma=" << sigma << " q=" << q;
                }
}

TEST(BinomialAmerican, NoDividendAmericanCallEqualsEuropean) {
    // With q = 0 early exercise of a call is never optimal: the American and
    // European prices coincide on the same tree (exact node-by-node).
    for (double K : {80.0, 100.0, 120.0}) {
        const double amer = crr_price(100.0, K, 1.0, 0.05, 0.25, 0.0,
                                      OptionType::Call,
                                      ExerciseStyle::American, 500);
        const double euro = crr_price(100.0, K, 1.0, 0.05, 0.25, 0.0,
                                      OptionType::Call,
                                      ExerciseStyle::European, 500);
        EXPECT_NEAR(amer, euro, 1e-12);
        EXPECT_DOUBLE_EQ(
            early_exercise_premium(100.0, K, 1.0, 0.05, 0.25, 0.0,
                                   OptionType::Call, 500),
            0.0);
    }
}

TEST(BinomialAmerican, PutCarriesEarlyExercisePremiumAndDominatesIntrinsic) {
    // Deep ITM American put: value >= intrinsic, and premium > 0 when r > q.
    const double amer = crr_price(80.0, 100.0, 1.0, 0.05, 0.2, 0.0,
                                  OptionType::Put, ExerciseStyle::American,
                                  500);
    EXPECT_GE(amer, 20.0 - 1e-12);
    EXPECT_GT(early_exercise_premium(80.0, 100.0, 1.0, 0.05, 0.2, 0.0,
                                     OptionType::Put, 500),
              0.0);
}

TEST(BinomialEdgeCases, DegenerateLimitsMatchPolicy) {
    // T == 0 -> intrinsic.
    EXPECT_DOUBLE_EQ(crr_price(110.0, 100.0, 0.0, 0.05, 0.2, 0.0,
                               OptionType::Call, ExerciseStyle::American, 100),
                     10.0);
    // sigma == 0, European -> Black-Scholes deterministic limit.
    EXPECT_DOUBLE_EQ(
        crr_price(100.0, 95.0, 1.0, 0.05, 0.0, 0.02, OptionType::Call,
                  ExerciseStyle::European, 100),
        bs_price(100.0, 95.0, 1.0, 0.05, 0.0, 0.02, OptionType::Call));
    // sigma == 0, American put with r > 0: exercising now beats waiting.
    const double amer0 = crr_price(80.0, 100.0, 1.0, 0.05, 0.0, 0.0,
                                   OptionType::Put, ExerciseStyle::American,
                                   100);
    EXPECT_NEAR(amer0, 20.0, 1e-12);
    // S == 0 American put pays K immediately.
    EXPECT_DOUBLE_EQ(crr_price(0.0, 100.0, 1.0, 0.05, 0.2, 0.0,
                               OptionType::Put, ExerciseStyle::American, 100),
                     100.0);
    // K == 0 put is worthless.
    EXPECT_DOUBLE_EQ(crr_price(100.0, 0.0, 1.0, 0.05, 0.2, 0.0,
                               OptionType::Put, ExerciseStyle::European, 100),
                     0.0);
}

TEST(BinomialValidation, InvalidInputsThrow) {
    EXPECT_THROW(crr_price(-1.0, 100.0, 1.0, 0.05, 0.2), std::invalid_argument);
    EXPECT_THROW(crr_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0,
                           OptionType::Call, ExerciseStyle::European, 0),
                 std::invalid_argument);
    // dt too large for r - q vs sigma: risk-neutral probability leaves (0,1).
    EXPECT_THROW(crr_price(100.0, 100.0, 1.0, 5.0, 0.01, 0.0,
                           OptionType::Call, ExerciseStyle::European, 1),
                 std::invalid_argument);
}

}  // namespace
