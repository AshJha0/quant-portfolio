// Implied volatility: round-trip recovery across a moneyness/expiry/vol grid,
// arbitrage-bound rejection (incl. sub-intrinsic prices), and degenerate
// inputs.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "eqopt/black_scholes.hpp"
#include "eqopt/implied_vol.hpp"

namespace {

using eqopt::bs_price;
using eqopt::implied_vol;
using eqopt::OptionType;

TEST(ImpliedVol, RoundTripTo1e8AcrossMoneynessAndExpiry) {
    const double S = 100.0, r = 0.03, q = 0.01;
    const std::vector<double> moneyness{0.5, 0.8, 1.0, 1.2, 2.0};
    const std::vector<double> expiries{0.05, 0.5, 2.0};
    const std::vector<double> vols{0.1, 0.3, 0.8};
    for (double m : moneyness)
        for (double T : expiries)
            for (double sigma_raw : vols)
                for (OptionType type : {OptionType::Call, OptionType::Put}) {
                    const double K = m * S;
                    // Same guard as the Python reference test: floor the vol
                    // so the strike stays within ~3 standard deviations of
                    // the forward. Further out the option's *time value*
                    // underflows double precision and no solver can recover
                    // vol from the price (that regime is covered by the
                    // sub-intrinsic rejection tests instead).
                    const double sigma = std::max(
                        sigma_raw,
                        std::fabs(std::log(K / S)) / (3.0 * std::sqrt(T)));
                    const double price = bs_price(S, K, T, r, sigma, q, type);
                    const double iv = implied_vol(price, S, K, T, r, q, type);
                    ASSERT_NEAR(iv, sigma, 1e-8)
                        << "K/S=" << m << " T=" << T << " sigma=" << sigma
                        << (type == OptionType::Call ? " call" : " put");
                }
}

TEST(ImpliedVol, RecoversVeryHighVol) {
    const double price =
        bs_price(100.0, 100.0, 1.0, 0.05, 4.0, 0.0, OptionType::Call);
    EXPECT_NEAR(implied_vol(price, 100.0, 100.0, 1.0, 0.05, 0.0), 4.0, 1e-7);
}

TEST(ImpliedVol, RejectsSubIntrinsicPrices) {
    const double S = 100.0, K = 80.0, T = 0.5, r = 0.05, q = 0.0;
    // sigma -> 0 lower bound: discounted forward intrinsic.
    const double lower = bs_price(S, K, T, r, 0.0, q, OptionType::Call);
    EXPECT_THROW(implied_vol(lower, S, K, T, r, q, OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(implied_vol(lower - 0.5, S, K, T, r, q, OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(implied_vol(0.0, S, K, T, r, q, OptionType::Call),
                 std::invalid_argument);
}

TEST(ImpliedVol, RejectsPricesAboveUpperBound) {
    // Call upper bound is S e^{-qT}; put upper bound is K e^{-rT}.
    EXPECT_THROW(
        implied_vol(101.0, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call),
        std::invalid_argument);
    EXPECT_THROW(
        implied_vol(100.0, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Put),
        std::invalid_argument);
}

TEST(ImpliedVol, RejectsDegenerateInputs) {
    EXPECT_THROW(implied_vol(5.0, 100.0, 100.0, 0.0, 0.05),  // T == 0
                 std::invalid_argument);
    EXPECT_THROW(implied_vol(5.0, 0.0, 100.0, 1.0, 0.05),  // S == 0
                 std::invalid_argument);
    EXPECT_THROW(implied_vol(5.0, 100.0, 0.0, 1.0, 0.05),  // K == 0
                 std::invalid_argument);
    EXPECT_THROW(
        implied_vol(std::nan(""), 100.0, 100.0, 1.0, 0.05),
        std::invalid_argument);
}

TEST(ImpliedVol, ShortDatedWingsConverge) {
    // Tiny vega region: Newton would stall; the bracket/bisection fallback
    // must still recover the vol.
    const double S = 100.0, K = 140.0, T = 0.05, r = 0.01, q = 0.0;
    const double sigma = 0.6;
    const double price = bs_price(S, K, T, r, sigma, q, OptionType::Call);
    EXPECT_NEAR(implied_vol(price, S, K, T, r, q, OptionType::Call), sigma,
                1e-7);
}

}  // namespace
