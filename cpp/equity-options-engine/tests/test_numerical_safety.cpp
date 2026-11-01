// Regression tests for wrong numbers at the edges of the input domain, the
// pair-average control variate and the shared implied-vol core.
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "eqopt/binomial.hpp"
#include "eqopt/black_scholes.hpp"
#include "eqopt/implied_vol.hpp"
#include "eqopt/iv_core.hpp"
#include "eqopt/monte_carlo.hpp"

using namespace eqopt;

namespace {
constexpr double kEps = std::numeric_limits<double>::epsilon();
}

TEST(NumericalSafety, SigmaZeroWithOverflowingForwardIsFinite) {
    // (r - q) T = 800 overflows the forward; the old branch gave NaN.
    EXPECT_EQ(bs_price(100.0, 100.0, 1000.0, 0.8, 0.0, 0.0, OptionType::Call), 100.0);
    EXPECT_EQ(bs_price(100.0, 100.0, 1000.0, 0.8, 0.0, 0.0, OptionType::Put), 0.0);
    EXPECT_EQ(crr_price(100.0, 100.0, 1000.0, 0.8, 0.0, 0.0, OptionType::Call,
                        ExerciseStyle::American, 100),
              100.0);
}

TEST(NumericalSafety, OverflowingDiscountFactorThrows) {
    EXPECT_THROW(bs_price(100.0, 100.0, 1000.0, -0.8, 0.2, 0.0, OptionType::Put),
                 std::invalid_argument);
    EXPECT_THROW(implied_vol(5.0, 100.0, 100.0, 1000.0, -0.8, 0.0, OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(mc_price(100.0, 100.0, 1000.0, 0.8, 0.2, 0.0, OptionType::Call, 1000),
                 std::invalid_argument);
}

TEST(NumericalSafety, AmericanPutAtZeroSpotWaitsWhenRatesAreNegative) {
    for (double r : {-0.02, -0.005}) {
        const double v =
            crr_price(0.0, 100.0, 1.0, r, 0.2, 0.0, OptionType::Put, ExerciseStyle::American, 100);
        const double expected = 100.0 * std::exp(-r);
        EXPECT_NEAR(v, expected, 4.0 * kEps * expected);
    }
    EXPECT_EQ(crr_price(0.0, 100.0, 1.0, 0.03, 0.2, 0.0, OptionType::Put,
                        ExerciseStyle::American, 100),
              100.0);
}

TEST(NumericalSafety, McNeedsIndependentSamples) {
    // Two antithetic paths are one pair: no standard error exists. With the
    // control variate, two samples fit the regression exactly.
    auto mc = [](std::size_t n, bool anti, bool cv) {
        return mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, n, anti, cv, 11);
    };
    EXPECT_THROW(mc(2, true, false), std::invalid_argument);
    EXPECT_THROW(mc(4, true, true), std::invalid_argument);
    EXPECT_THROW(mc(2, false, true), std::invalid_argument);
    EXPECT_GT(mc(3, true, false).std_error, 0.0);
    EXPECT_GT(mc(6, true, true).std_error, 0.0);
}

TEST(NumericalSafety, AntitheticPlusCvBeatsCvAloneAcrossSeeds) {
    // Fitting beta on raw draws and pairing afterwards gave SE 0.0247 at
    // 100k paths here, worse than the control variate alone (0.0177).
    for (std::uint64_t seed = 1; seed <= 7; ++seed) {
        const auto both = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, 100000,
                                   true, true, seed);
        const auto cv = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, 100000,
                                 false, true, seed);
        EXPECT_LE(both.std_error, cv.std_error) << "seed " << seed;
        EXPECT_LT(both.std_error, 0.6 * 0.0246) << "seed " << seed;
        const double exact = bs_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call);
        EXPECT_LT(std::fabs(both.price - exact), 4.0 * both.std_error);
    }
}

TEST(ImpliedVolCore, K90OtmPutRecoversTo1e12) {
    const double put = bs_price(100.0, 90.0, 0.02, 0.03, 0.10, 0.0, OptionType::Put);
    EXPECT_NEAR(implied_vol(put, 100.0, 90.0, 0.02, 0.03, 0.0, OptionType::Put), 0.10, 1e-12);
}

TEST(ImpliedVolCore, K90ItmCallIsBackwardStable) {
    // One ulp of the call (1.8e-15) over vega (3.5e-12) is a vol
    // uncertainty of about 5e-4: the quote cannot give 1e-12.
    const double call = bs_price(100.0, 90.0, 0.02, 0.03, 0.10, 0.0, OptionType::Call);
    const double iv = implied_vol(call, 100.0, 90.0, 0.02, 0.03, 0.0, OptionType::Call);
    EXPECT_NEAR(bs_price(100.0, 90.0, 0.02, 0.03, iv, 0.0, OptionType::Call), call,
                16.0 * kEps * call);
}

TEST(ImpliedVolCore, AtMostTenEvaluationsAndConditionBoundAccuracy) {
    const double df = 0.97;
    for (double lk : {-3.0, -1.0, -0.3, -0.05, 0.0, 0.05, 0.3, 1.0, 3.0}) {
        for (double t : {1e-4, 0.02, 0.25, 1.0, 10.0}) {
            for (double sigma : {0.01, 0.1, 0.5, 2.0}) {
                const double k = std::exp(lk);
                const bool is_call = k >= 1.0;
                const auto ot = is_call ? OptionType::Call : OptionType::Put;
                const double r = -std::log(df) / t;
                const double price = bs_price(df, k, t, r, sigma, 0.0, ot);
                if (!(price > 1e-300)) continue;
                int evals = 0;
                const double iv = iv_core::black_implied_vol(price, 1.0, k, t, df, is_call, evals);
                EXPECT_LE(evals, 10) << lk << " " << t << " " << sigma;
                // Out-of-the-money quote: one rounding of the price over vega.
                const double v = sigma * std::sqrt(t);
                const double d1 = (std::log(1.0 / k)) / v + 0.5 * v;
                const double vega = df * std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI) * std::sqrt(t);
                const double tol = 4.0 * std::fmax(kEps * price / vega, 1e-12 * sigma);
                EXPECT_NEAR(iv, sigma, tol) << lk << " " << t << " " << sigma;
            }
        }
    }
}

TEST(ImpliedVolCore, RootAtTheInflectionPointConvergesFast) {
    // Total vol equal to s_c = sqrt(2|x|) sits on the branch boundary;
    // bisecting towards s_c took 12 evaluations, probing it takes 2.
    const double df = 0.97, t = 10.0, k = std::exp(-0.05);
    const double r = -std::log(df) / t;
    const double price = bs_price(df, k, t, r, 0.1, 0.0, OptionType::Put);
    int evals = 0;
    const double iv = iv_core::black_implied_vol(price, 1.0, k, t, df, false, evals);
    EXPECT_LE(evals, 4);
    EXPECT_NEAR(iv, 0.1, 1e-13);
}
