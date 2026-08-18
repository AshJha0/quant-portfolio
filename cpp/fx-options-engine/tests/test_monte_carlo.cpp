// Monte Carlo: 3-standard-error agreement with GK, variance-reduction
// effectiveness, seed determinism, and validation.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

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
