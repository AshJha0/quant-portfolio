// Black-Scholes core: put-call parity, edge cases, input validation, and a
// NaN-free domain scan. Mirrors the Python reference's documented edge-case
// policy exactly.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "eqopt/black_scholes.hpp"

namespace {

using eqopt::bs_price;
using eqopt::OptionType;

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

TEST(PutCallParity, HoldsTo1e12AcrossGrid) {
    const std::vector<double> spots{50.0, 100.0, 150.0};
    const std::vector<double> strikes{80.0, 100.0, 120.0};
    const std::vector<double> expiries{0.1, 1.0, 3.0};
    const std::vector<double> vols{0.05, 0.2, 0.6};
    const std::vector<double> rates{-0.01, 0.0, 0.05};
    const std::vector<double> divs{0.0, 0.03};
    for (double S : spots)
        for (double K : strikes)
            for (double T : expiries)
                for (double sigma : vols)
                    for (double r : rates)
                        for (double q : divs) {
                            const double c =
                                bs_price(S, K, T, r, sigma, q, OptionType::Call);
                            const double p =
                                bs_price(S, K, T, r, sigma, q, OptionType::Put);
                            const double rhs = S * std::exp(-q * T) -
                                               K * std::exp(-r * T);
                            EXPECT_NEAR(c - p, rhs, 1e-12 * std::max(1.0, S))
                                << "S=" << S << " K=" << K << " T=" << T
                                << " sigma=" << sigma << " r=" << r
                                << " q=" << q;
                        }
}

TEST(EdgeCases, ZeroExpiryReturnsIntrinsic) {
    EXPECT_DOUBLE_EQ(bs_price(110.0, 100.0, 0.0, 0.05, 0.2, 0.0,
                              OptionType::Call), 10.0);
    EXPECT_DOUBLE_EQ(bs_price(90.0, 100.0, 0.0, 0.05, 0.2, 0.0,
                              OptionType::Call), 0.0);
    EXPECT_DOUBLE_EQ(bs_price(90.0, 100.0, 0.0, 0.05, 0.2, 0.0,
                              OptionType::Put), 10.0);
}

TEST(EdgeCases, ZeroVolReturnsDiscountedForwardIntrinsic) {
    const double S = 100.0, K = 95.0, T = 1.0, r = 0.05, q = 0.02;
    const double fwd = S * std::exp((r - q) * T);
    const double expect_call = std::exp(-r * T) * std::max(fwd - K, 0.0);
    EXPECT_DOUBLE_EQ(bs_price(S, K, T, r, 0.0, q, OptionType::Call), expect_call);
    EXPECT_DOUBLE_EQ(bs_price(S, K, T, r, 0.0, q, OptionType::Put),
                     std::exp(-r * T) * std::max(K - fwd, 0.0));
}

TEST(EdgeCases, ZeroStrikeAndZeroSpotLimits) {
    // Zero-strike call is a dividend-adjusted forward on the stock.
    EXPECT_DOUBLE_EQ(bs_price(100.0, 0.0, 2.0, 0.05, 0.2, 0.03,
                              OptionType::Call), 100.0 * std::exp(-0.03 * 2.0));
    EXPECT_DOUBLE_EQ(bs_price(100.0, 0.0, 2.0, 0.05, 0.2, 0.03,
                              OptionType::Put), 0.0);
    EXPECT_DOUBLE_EQ(bs_price(0.0, 100.0, 2.0, 0.05, 0.2, 0.0,
                              OptionType::Call), 0.0);
    EXPECT_DOUBLE_EQ(bs_price(0.0, 100.0, 2.0, 0.05, 0.2, 0.0,
                              OptionType::Put), 100.0 * std::exp(-0.05 * 2.0));
}

TEST(EdgeCases, HugeVolApproachesUpperBound) {
    // sigma -> inf: call -> S e^{-qT}, put -> K e^{-rT}.
    const double S = 100.0, K = 100.0, T = 1.0, r = 0.03, q = 0.01;
    const double call = bs_price(S, K, T, r, 50.0, q, OptionType::Call);
    const double put = bs_price(S, K, T, r, 50.0, q, OptionType::Put);
    EXPECT_TRUE(std::isfinite(call));
    EXPECT_TRUE(std::isfinite(put));
    EXPECT_NEAR(call, S * std::exp(-q * T), 1e-6);
    EXPECT_NEAR(put, K * std::exp(-r * T), 1e-6);
    EXPECT_LE(call, S * std::exp(-q * T));
    EXPECT_LE(put, K * std::exp(-r * T));
}

TEST(EdgeCases, DeepWingsAreExactAndFinite) {
    // erfc-based CDF: deep OTM prices underflow gracefully to ~0, never NaN.
    const double deep_otm =
        bs_price(100.0, 10000.0, 0.1, 0.02, 0.1, 0.0, OptionType::Call);
    EXPECT_TRUE(std::isfinite(deep_otm));
    EXPECT_GE(deep_otm, 0.0);
    EXPECT_LT(deep_otm, 1e-100);
    const double deep_itm =
        bs_price(100.0, 1e-8, 1.0, 0.02, 0.2, 0.0, OptionType::Call);
    EXPECT_NEAR(deep_itm, 100.0 - 1e-8 * std::exp(-0.02), 1e-9);
}

TEST(Validation, NegativeInputsThrowInvalidArgument) {
    EXPECT_THROW(bs_price(-1.0, 100.0, 1.0, 0.05, 0.2), std::invalid_argument);
    EXPECT_THROW(bs_price(100.0, -1.0, 1.0, 0.05, 0.2), std::invalid_argument);
    EXPECT_THROW(bs_price(100.0, 100.0, -1.0, 0.05, 0.2), std::invalid_argument);
    EXPECT_THROW(bs_price(100.0, 100.0, 1.0, 0.05, -0.2), std::invalid_argument);
}

TEST(Validation, NanInputsThrowInvalidArgument) {
    EXPECT_THROW(bs_price(kNan, 100.0, 1.0, 0.05, 0.2), std::invalid_argument);
    EXPECT_THROW(bs_price(100.0, kNan, 1.0, 0.05, 0.2), std::invalid_argument);
    EXPECT_THROW(bs_price(100.0, 100.0, kNan, 0.05, 0.2), std::invalid_argument);
    EXPECT_THROW(bs_price(100.0, 100.0, 1.0, 0.05, kNan), std::invalid_argument);
}

TEST(Validation, NegativeRatesAndYieldsAreSupported) {
    EXPECT_NO_THROW(bs_price(100.0, 100.0, 1.0, -0.02, 0.2, -0.01));
    const double c = bs_price(100.0, 100.0, 1.0, -0.02, 0.2, -0.01);
    EXPECT_TRUE(std::isfinite(c));
    EXPECT_GT(c, 0.0);
}

TEST(Validation, D1D2RejectDegenerateInputs) {
    EXPECT_THROW(eqopt::d1_d2(0.0, 100.0, 1.0, 0.05, 0.2),
                 std::invalid_argument);
    EXPECT_THROW(eqopt::d1_d2(100.0, 100.0, 0.0, 0.05, 0.2),
                 std::invalid_argument);
    EXPECT_THROW(eqopt::d1_d2(100.0, 100.0, 1.0, 0.05, 0.0),
                 std::invalid_argument);
}

TEST(DomainScan, NoNansOrNegativePricesAcrossWideDomain) {
    // Wide sweep incl. extremes: prices must be finite, non-negative and
    // within no-arbitrage bounds everywhere in the valid domain.
    const std::vector<double> spots{1e-6, 1.0, 100.0, 1e6};
    const std::vector<double> strikes{1e-6, 1.0, 100.0, 1e6};
    const std::vector<double> expiries{1e-6, 0.5, 10.0};
    const std::vector<double> vols{1e-8, 0.2, 3.0};
    const std::vector<double> rates{-0.05, 0.0, 0.1};
    for (double S : spots)
        for (double K : strikes)
            for (double T : expiries)
                for (double sigma : vols)
                    for (double r : rates)
                        for (OptionType type :
                             {OptionType::Call, OptionType::Put}) {
                            const double v =
                                bs_price(S, K, T, r, sigma, 0.01, type);
                            ASSERT_TRUE(std::isfinite(v))
                                << "S=" << S << " K=" << K << " T=" << T
                                << " sigma=" << sigma << " r=" << r;
                            ASSERT_GE(v, 0.0);
                            const double upper = type == OptionType::Call
                                                     ? S * std::exp(-0.01 * T)
                                                     : K * std::exp(-r * T);
                            ASSERT_LE(v, upper * (1.0 + 1e-12) + 1e-12);
                        }
}

TEST(Monotonicity, CallPriceIncreasesWithVol) {
    double prev = bs_price(100.0, 100.0, 1.0, 0.05, 0.01, 0.0);
    for (double sigma : {0.05, 0.1, 0.2, 0.4, 0.8, 1.6}) {
        const double p = bs_price(100.0, 100.0, 1.0, 0.05, sigma, 0.0);
        EXPECT_GT(p, prev);
        prev = p;
    }
}

}  // namespace
