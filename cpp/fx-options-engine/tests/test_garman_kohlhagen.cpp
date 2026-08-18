// Garman-Kohlhagen core: parity, limits, monotonicity and input validation.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "fxopt/garman_kohlhagen.hpp"

namespace {

using namespace fxopt;

constexpr double S = 1.10, K = 1.08, T = 0.5, RD = 0.0425, RF = 0.0290,
                 SIG = 0.0925;

TEST(GarmanKohlhagen, PutCallParityWithTwoRates) {
    // C - P = S e^{-r_f T} - K e^{-r_d T}, exact to 1e-12.
    const struct {
        double s, k, t, rd, rf, sig;
    } grid[] = {
        {1.10, 1.00, 0.25, 0.045, 0.030, 0.095},
        {1.10, 1.15, 1.00, 0.045, 0.030, 0.095},
        {147.5, 150.0, 0.50, 0.0525, 0.0010, 0.1050},
        {0.92, 0.95, 2.00, -0.0075, -0.0050, 0.0650},
    };
    for (const auto& g : grid) {
        const double c = gk_call(g.s, g.k, g.t, g.rd, g.rf, g.sig);
        const double p = gk_put(g.s, g.k, g.t, g.rd, g.rf, g.sig);
        const double rhs = g.s * std::exp(-g.rf * g.t) -
                           g.k * std::exp(-g.rd * g.t);
        EXPECT_NEAR(c - p, rhs, 1e-12);
    }
}

TEST(GarmanKohlhagen, D1D2Definitions) {
    const double d1v = d1(S, K, T, RD, RF, SIG);
    const double d2v = d2(S, K, T, RD, RF, SIG);
    const double expected_d1 =
        (std::log(S / K) + (RD - RF + 0.5 * SIG * SIG) * T) /
        (SIG * std::sqrt(T));
    EXPECT_NEAR(d1v, expected_d1, 1e-15);
    EXPECT_NEAR(d2v, d1v - SIG * std::sqrt(T), 1e-15);
}

TEST(GarmanKohlhagen, ZeroTimeReturnsIntrinsic) {
    // Compare against the same floating-point subtraction the intrinsic
    // performs (1.10 - 1.00 != the literal 0.10 by a few ULP).
    EXPECT_DOUBLE_EQ(gk_call(1.10, 1.00, 0.0, RD, RF, SIG), 1.10 - 1.00);
    EXPECT_DOUBLE_EQ(gk_put(1.10, 1.00, 0.0, RD, RF, SIG), 0.0);
    EXPECT_DOUBLE_EQ(gk_put(1.00, 1.10, 0.0, RD, RF, SIG), 1.10 - 1.00);
}

TEST(GarmanKohlhagen, ZeroVolReturnsDiscountedForwardIntrinsic) {
    const double fwd = S * std::exp((RD - RF) * T);
    const double expected = std::exp(-RD * T) * std::max(fwd - K, 0.0);
    EXPECT_NEAR(gk_call(S, K, T, RD, RF, 0.0), expected, 1e-15);
    EXPECT_DOUBLE_EQ(gk_put(S, K / 2.0, T, RD, RF, 0.0), 0.0);
}

TEST(GarmanKohlhagen, PriceIncreasesWithVol) {
    double prev = gk_call(S, K, T, RD, RF, 0.01);
    for (double sig = 0.05; sig <= 0.60; sig += 0.05) {
        const double cur = gk_call(S, K, T, RD, RF, sig);
        EXPECT_GT(cur, prev);
        prev = cur;
    }
}

TEST(GarmanKohlhagen, ThrowsOnInvalidInputs) {
    EXPECT_THROW(gk_call(-1.0, K, T, RD, RF, SIG), std::invalid_argument);
    EXPECT_THROW(gk_call(0.0, K, T, RD, RF, SIG), std::invalid_argument);
    EXPECT_THROW(gk_call(S, -1.0, T, RD, RF, SIG), std::invalid_argument);
    EXPECT_THROW(gk_call(S, K, -0.5, RD, RF, SIG), std::invalid_argument);
    EXPECT_THROW(gk_call(S, K, T, RD, RF, -0.1), std::invalid_argument);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_THROW(gk_call(nan, K, T, RD, RF, SIG), std::invalid_argument);
    EXPECT_THROW(gk_call(S, K, T, inf, RF, SIG), std::invalid_argument);
    EXPECT_THROW(d1(S, K, 0.0, RD, RF, SIG), std::invalid_argument);
    EXPECT_THROW(d1(S, K, T, RD, RF, 0.0), std::invalid_argument);
}

}  // namespace
