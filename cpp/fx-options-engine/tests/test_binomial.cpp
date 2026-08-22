// CRR tree: convergence to GK, American vs European ordering, and the
// FX-specific early-exercise economics (foreign carry as 'dividend').

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

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

TEST(Binomial, ConvergenceRateFittedExponentMatchesTheoreticalOrderOne) {
    // CRR is a first-order scheme: error(n) ~ C / n, with an odd/even (and
    // node-alignment) oscillation superposed. A two-point ratio (as in
    // EuropeanConvergesToGarmanKohlhagen above) cannot *prove* the O(1/n)
    // rate -- it can land anywhere in the oscillation. Regressing
    // log|error| on log(n) over a decade of geometrically spaced step
    // counts averages the oscillation out and recovers the leading
    // exponent, which must come out close to the theoretical -1.
    const double analytic = gk_call(S, K, T, RD, RF, SIG);
    const std::vector<int> steps{200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200};
    std::vector<double> log_n, log_err;
    for (int n : steps) {
        const double err = std::abs(
            binomial_price(S, K, T, RD, RF, SIG, OptionType::Call, n) -
            analytic);
        ASSERT_GT(err, 0.0) << "tree exactly matches GK at n=" << n;
        log_n.push_back(std::log(static_cast<double>(n)));
        log_err.push_back(std::log(err));
    }
    const double mean_x =
        std::accumulate(log_n.begin(), log_n.end(), 0.0) / log_n.size();
    const double mean_y =
        std::accumulate(log_err.begin(), log_err.end(), 0.0) / log_err.size();
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < log_n.size(); ++i) {
        num += (log_n[i] - mean_x) * (log_err[i] - mean_y);
        den += (log_n[i] - mean_x) * (log_n[i] - mean_x);
    }
    const double slope = num / den;
    EXPECT_GT(slope, -1.3) << "fitted CRR exponent " << slope;
    EXPECT_LT(slope, -0.7) << "fitted CRR exponent " << slope;
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
