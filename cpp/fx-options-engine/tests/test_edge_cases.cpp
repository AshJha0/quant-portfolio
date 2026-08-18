// Edge cases from the documentation contract: negative rates (EUR/CHF
// era), r_d == r_f, T = 0, deep ITM/OTM, and invalid-input behaviour.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "fxopt/deltas.hpp"
#include "fxopt/forwards.hpp"
#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/greeks.hpp"
#include "fxopt/implied_vol.hpp"

namespace {

using namespace fxopt;

TEST(EdgeCases, NegativeRatesEurChfEra) {
    // EURCHF around 2019: both rates negative, r_f < r_d < 0.
    const double S = 1.08, K = 1.10, T = 1.0, rd = -0.0075, rf = -0.0050,
                 sig = 0.055;
    const double c = gk_call(S, K, T, rd, rf, sig);
    const double p = gk_put(S, K, T, rd, rf, sig);
    EXPECT_GT(c, 0.0);
    EXPECT_GT(p, 0.0);
    // Parity still exact.
    EXPECT_NEAR(c - p, S * std::exp(-rf * T) - K * std::exp(-rd * T), 1e-12);
    // Greeks finite and FD-consistent.
    const GreeksResult g = analytic_greeks(S, K, T, rd, rf, sig,
                                           OptionType::Call);
    const FDGreeks fd = finite_difference_greeks(S, K, T, rd, rf, sig,
                                                 OptionType::Call);
    EXPECT_NEAR(g.delta_spot, fd.delta_spot, 1e-6);
    EXPECT_NEAR(g.rho_domestic, fd.rho_domestic, 1e-6);
    EXPECT_NEAR(g.rho_foreign, fd.rho_foreign, 1e-6);
    // Forward is BELOW spot when r_d < r_f.
    EXPECT_LT(cip_forward(S, T, rd, rf), S);
    // Strike-from-delta still round-trips.
    const double k25 = strike_from_delta(0.25, S, T, rd, rf, sig,
                                         OptionType::Call,
                                         DeltaConvention::SpotPa);
    EXPECT_NEAR(delta(S, k25, T, rd, rf, sig, OptionType::Call,
                      DeltaConvention::SpotPa),
                0.25, 1e-8);
}

TEST(EdgeCases, EqualRatesReduceToDrivelessForward) {
    // r_d == r_f: F = S, forward points 0, ATM-forward strike = spot.
    const double S = 1.25, T = 0.5, r = 0.03, sig = 0.10;
    EXPECT_DOUBLE_EQ(cip_forward(S, T, r, r), S);
    EXPECT_DOUBLE_EQ(forward_points(S, T, r, r), 0.0);
    EXPECT_DOUBLE_EQ(atm_forward_strike(S, T, r, r), S);
    // Price equals undiscounted Black with symmetric discounting.
    const double c = gk_call(S, S, T, r, r, sig);
    const double v = sig * std::sqrt(T);
    const double expected =
        S * std::exp(-r * T) * (norm_cdf(0.5 * v) - norm_cdf(-0.5 * v));
    EXPECT_NEAR(c, expected, 1e-14);
}

TEST(EdgeCases, ZeroTimeAndDeepStrikes) {
    EXPECT_DOUBLE_EQ(gk_call(1.10, 0.90, 0.0, 0.04, 0.03, 0.10), 0.20);
    EXPECT_DOUBLE_EQ(gk_put(1.10, 0.90, 0.0, 0.04, 0.03, 0.10), 0.0);
    // Deep ITM call converges to discounted forward minus strike.
    const double S = 1.10, T = 0.5, rd = 0.0425, rf = 0.0290, sig = 0.0925;
    const double k_deep = 0.30;
    const double bound = S * std::exp(-rf * T) -
                         k_deep * std::exp(-rd * T);
    EXPECT_NEAR(gk_call(S, k_deep, T, rd, rf, sig), bound, 1e-12);
    // Deep OTM call is (numerically) zero but non-negative.
    const double otm = gk_call(S, 5.0, T, rd, rf, sig);
    EXPECT_GE(otm, 0.0);
    EXPECT_LT(otm, 1e-12);
}

TEST(EdgeCases, ImpliedVolAtExtremeMoneyness) {
    const double S = 1.10, T = 0.25, rd = 0.0425, rf = 0.0290;
    for (const double K : {0.85, 1.45}) {
        const double sig = 0.14;
        const double price = gk_price(S, K, T, rd, rf, sig, OptionType::Call);
        if (price > 1e-300) {
            EXPECT_NEAR(implied_vol(price, S, K, T, rd, rf, OptionType::Call),
                        sig, 1e-8);
        }
    }
}

TEST(EdgeCases, InvalidInputsThrowEverywhere) {
    EXPECT_THROW(cip_forward(-1.0, 0.5, 0.04, 0.03), std::invalid_argument);
    EXPECT_THROW(atm_dns_strike(1.1, -0.5, 0.04, 0.03, 0.1),
                 std::invalid_argument);
    EXPECT_THROW(delta(1.1, 1.1, 0.5, 0.04, 0.03, 0.0, OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(premium_adjust_spot_delta(0.5, 0.02, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(implied_vol(-0.02, 1.1, 1.1, 0.5, 0.04, 0.03,
                             OptionType::Call),
                 std::invalid_argument);  // negative premium: no-arbitrage
}

}  // namespace
