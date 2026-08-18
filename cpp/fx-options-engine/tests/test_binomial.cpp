// CRR tree: convergence to GK, American vs European ordering, and the
// FX-specific early-exercise economics (foreign carry as 'dividend').

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "fxopt/binomial.hpp"
#include "fxopt/garman_kohlhagen.hpp"

namespace {

using namespace fxopt;

constexpr double S = 1.10, K = 1.08, T = 0.75, RD = 0.0425, RF = 0.0290,
                 SIG = 0.0925;

TEST(Binomial, EuropeanConvergesToGarmanKohlhagen) {
    const double analytic = gk_call(S, K, T, RD, RF, SIG);
    double prev_err = 1e9;
    for (const int steps : {50, 200, 800}) {
        const double tree = binomial_price(S, K, T, RD, RF, SIG,
                                           OptionType::Call, steps);
        const double err = std::abs(tree - analytic);
        EXPECT_LT(err, prev_err * 1.05);  // roughly monotone O(1/n) decay
        prev_err = err;
    }
    EXPECT_NEAR(binomial_price(S, K, T, RD, RF, SIG, OptionType::Call, 2000),
                analytic, 5e-6);
    EXPECT_NEAR(binomial_price(S, K, T, RD, RF, SIG, OptionType::Put, 2000),
                gk_put(S, K, T, RD, RF, SIG), 5e-6);
}

TEST(Binomial, AmericanAtLeastEuropean) {
    for (const auto type : {OptionType::Call, OptionType::Put})
        for (const double k : {0.95, 1.10, 1.25}) {
            const double eur = binomial_price(S, k, T, RD, RF, SIG, type, 400,
                                              Exercise::European);
            const double amer = binomial_price(S, k, T, RD, RF, SIG, type,
                                               400, Exercise::American);
            EXPECT_GE(amer, eur - 1e-12);
        }
}

TEST(Binomial, EarlyExercisePremiumWhenForeignRateHigh) {
    // Economic test: an American call on a high-yielding foreign currency
    // (r_f > r_d) forfeits foreign carry while unexercised -> strictly
    // positive early-exercise premium for an ITM call.  Mirrors the
    // dividend-yield story for equity calls.
    const double rd = 0.005, rf = 0.0525;  // e.g. JPY investor holding USD
    const double s = 147.5, k = 140.0, t = 1.0, sig = 0.105;
    const double eur = binomial_price(s, k, t, rd, rf, sig, OptionType::Call,
                                      800, Exercise::European);
    const double amer = binomial_price(s, k, t, rd, rf, sig, OptionType::Call,
                                       800, Exercise::American);
    EXPECT_GT(amer - eur, 1e-3);
    // Control: with r_d > r_f the ITM *put* carries the premium instead.
    const double eur_put = binomial_price(S, 1.20, T, 0.0525, 0.005, SIG,
                                          OptionType::Put, 800,
                                          Exercise::European);
    const double amer_put = binomial_price(S, 1.20, T, 0.0525, 0.005, SIG,
                                           OptionType::Put, 800,
                                           Exercise::American);
    EXPECT_GT(amer_put - eur_put, 1e-5);
}

TEST(Binomial, DegenerateLimits) {
    // 1.10 - 1.00 (not the literal 0.10): match the intrinsic's own
    // floating-point subtraction.
    EXPECT_DOUBLE_EQ(
        binomial_price(1.10, 1.00, 0.0, RD, RF, SIG, OptionType::Call),
        1.10 - 1.00);
    // sigma = 0 European defers to the GK zero-vol limit.
    EXPECT_NEAR(binomial_price(S, K, T, RD, RF, 0.0, OptionType::Call),
                gk_call(S, K, T, RD, RF, 0.0), 1e-15);
    // sigma = 0 American: deterministic drifting spot, best exercise value.
    const double amer0 = binomial_price(S, 0.9, T, 0.005, 0.0525, 0.0,
                                        OptionType::Call, 500,
                                        Exercise::American);
    EXPECT_GE(amer0, S - 0.9 - 1e-12);  // at least immediate exercise
}

TEST(Binomial, ThrowsOnBadInputs) {
    EXPECT_THROW(binomial_price(S, K, T, RD, RF, SIG, OptionType::Call, 0),
                 std::invalid_argument);
    EXPECT_THROW(binomial_price(-1.0, K, T, RD, RF, SIG, OptionType::Call),
                 std::invalid_argument);
    // dt too coarse for drift/vol -> p outside [0,1].
    EXPECT_THROW(binomial_price(S, K, 10.0, 0.50, 0.0, 0.01,
                                OptionType::Call, 1),
                 std::invalid_argument);
}

}  // namespace
