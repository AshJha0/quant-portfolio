// Edge cases from the documentation contract: negative rates (EUR/CHF
// era), r_d == r_f, T = 0, deep ITM/OTM, and invalid-input behaviour.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "fxopt/binomial.hpp"
#include "fxopt/black76.hpp"
#include "fxopt/deltas.hpp"
#include "fxopt/forwards.hpp"
#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/greeks.hpp"
#include "fxopt/implied_vol.hpp"
#include "fxopt/monte_carlo.hpp"

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

TEST(EdgeCases, ExtremeMoneynessHundredFold) {
    // S/K = 100x and K/S = 100x: prices collapse onto the arbitrage bounds,
    // stay non-negative and finite, and parity survives at both wings.
    const double T = 1.0, rd = 0.04, rf = 0.02, sig = 0.12;
    const double df_d = std::exp(-rd * T), df_f = std::exp(-rf * T);

    const double deep_itm = gk_call(100.0, 1.0, T, rd, rf, sig);
    EXPECT_NEAR(deep_itm, 100.0 * df_f - 1.0 * df_d, 1e-12);
    const double deep_otm = gk_call(1.0, 100.0, T, rd, rf, sig);
    EXPECT_GE(deep_otm, 0.0);
    EXPECT_LT(deep_otm, 1e-100);
    const double deep_itm_put = gk_put(1.0, 100.0, T, rd, rf, sig);
    EXPECT_NEAR(deep_itm_put, 100.0 * df_d - 1.0 * df_f, 1e-12);

    for (const double S : {1.0, 100.0})
        for (const double K : {1.0, 100.0}) {
            const double c = gk_call(S, K, T, rd, rf, sig);
            const double p = gk_put(S, K, T, rd, rf, sig);
            ASSERT_TRUE(std::isfinite(c));
            ASSERT_TRUE(std::isfinite(p));
            EXPECT_GE(c, 0.0);
            EXPECT_GE(p, 0.0);
            EXPECT_LE(c, S * df_f * (1.0 + 1e-12));
            EXPECT_LE(p, K * df_d * (1.0 + 1e-12));
            EXPECT_NEAR(c - p, S * df_f - K * df_d, 1e-12 * std::max(1.0, S));
        }
}

TEST(EdgeCases, VeryShortAndVeryLongExpiry) {
    const double S = 1.10, K = 1.10, rd = 0.03, rf = 0.01, sig = 0.10;
    // T -> 0 (about 5 minutes): price collapses towards intrinsic (0 ATM)
    // but stays strictly positive, and Black-76 on the CIP forward agrees.
    const double t_tiny = 1e-5;
    const double c_tiny = gk_call(S, K, t_tiny, rd, rf, sig);
    EXPECT_GT(c_tiny, 0.0);
    EXPECT_LT(c_tiny, 1e-3);
    EXPECT_NEAR(c_tiny, black76_from_spot(S, K, t_tiny, rd, rf, sig,
                                          OptionType::Call),
                1e-15);
    // 30-year FX option: finite, inside the bounds, parity exact.
    const double t_long = 30.0;
    const double c = gk_call(S, K, t_long, rd, rf, sig);
    const double p = gk_put(S, K, t_long, rd, rf, sig);
    ASSERT_TRUE(std::isfinite(c));
    ASSERT_TRUE(std::isfinite(p));
    EXPECT_LE(c, S * std::exp(-rf * t_long) * (1.0 + 1e-12));
    EXPECT_LE(p, K * std::exp(-rd * t_long) * (1.0 + 1e-12));
    EXPECT_NEAR(c - p,
                S * std::exp(-rf * t_long) - K * std::exp(-rd * t_long),
                1e-12);
    // Round trip through the solver still works at 30y.
    EXPECT_NEAR(implied_vol(c, S, K, t_long, rd, rf, OptionType::Call), sig,
                1e-8);
}

TEST(EdgeCases, ZeroRatesReduceToUndiscountedBlack) {
    // r_d = r_f = 0: forward = spot, no discounting, parity C - P = S - K.
    const double S = 1.10, K = 1.05, T = 0.75, sig = 0.11;
    EXPECT_DOUBLE_EQ(cip_forward(S, T, 0.0, 0.0), S);
    const double c = gk_call(S, K, T, 0.0, 0.0, sig);
    const double p = gk_put(S, K, T, 0.0, 0.0, sig);
    EXPECT_NEAR(c - p, S - K, 1e-14);
    // rho_d and rho_f are still well defined and opposite in sign for a call.
    const GreeksResult g = analytic_greeks(S, K, T, 0.0, 0.0, sig,
                                           OptionType::Call);
    EXPECT_GT(g.rho_domestic, 0.0);
    EXPECT_LT(g.rho_foreign, 0.0);
}

TEST(EdgeCases, VolNearSolverCapSaturatesAtTheArbitrageBound) {
    // sigma = 10 (the top of the Newton bracket) still round-trips well.
    const double S = 1.10, K = 1.10, T = 1.0, rd = 0.03, rf = 0.01;
    const double px10 = gk_price(S, K, T, rd, rf, 10.0, OptionType::Call);
    EXPECT_NEAR(implied_vol(px10, S, K, T, rd, rf, OptionType::Call), 10.0,
                1e-6);
    // Far above it the premium is numerically identical to the sigma -> inf
    // bound S e^{-r_f T}: N(d1)/N(d2) have saturated to 1/0 in double
    // precision, so gk_price(sigma) is bit-identical to that bound for
    // *every* sigma from the true root up to infinity -- a flat plateau,
    // not a resolvable root. Previously the solver picked an arbitrary
    // point in that plateau (whatever `hi` its bracket-expansion doubling
    // happened to reach) and returned it silently, satisfied because it
    // "reprices exactly" -- but every other point in the plateau also
    // reprices exactly, so that check never actually validated the
    // *vol*, only self-consistency of a number that can be off by whole
    // vol points. The solver now recognises the plateau and throws
    // instead of guessing (matches the Python and eq_options references).
    const double px25 = gk_price(S, K, T, rd, rf, 25.0, OptionType::Call);
    EXPECT_DOUBLE_EQ(px25, S * std::exp(-rf * T));
    EXPECT_THROW(implied_vol(px25, S, K, T, rd, rf, OptionType::Call),
                 std::invalid_argument);
}

TEST(EdgeCases, NonFiniteInputsRejectedAcrossTheApi) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const double S = 1.10, K = 1.10, T = 0.5, rd = 0.03, rf = 0.01,
                 sig = 0.10;
    EXPECT_THROW(implied_vol(nan, S, K, T, rd, rf, OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(implied_vol(inf, S, K, T, rd, rf, OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(premium_adjust_spot_delta(0.5, nan, S),
                 std::invalid_argument);
    EXPECT_THROW(premium_adjust_spot_delta(nan, 0.02, S),
                 std::invalid_argument);
    EXPECT_THROW(mc_price(S, K, T, rd, rf, nan, OptionType::Call, 1000),
                 std::invalid_argument);
    EXPECT_THROW(binomial_price(S, K, T, inf, rf, sig, OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(black76_price(nan, K, T, rd, sig, OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(cip_forward(S, T, nan, rf), std::invalid_argument);
    EXPECT_THROW(strike_from_delta(nan, S, T, rd, rf, sig, OptionType::Call),
                 std::invalid_argument);
}

TEST(EdgeCases, FiniteDifferenceGreeksRejectUnbumpableVol) {
    // sigma below its own central bump would require pricing at a negative
    // volatility: rejected with an explicit message rather than surfacing
    // as a confusing "sigma must be non-negative" from the pricer.
    EXPECT_THROW(finite_difference_greeks(1.10, 1.10, 0.5, 0.03, 0.01, 1e-8,
                                          OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(finite_difference_greeks(1.10, 1.10, 0.0, 0.03, 0.01, 0.10,
                                          OptionType::Call),
                 std::invalid_argument);
}

TEST(EdgeCases, MonteCarloSinglePairReportsFiniteStandardError) {
    // n_paths = 2 with antithetic leaves exactly one independent sample:
    // the SE is unestimable and must be reported as 0, not NaN.
    const MCResult r = mc_price(1.10, 1.10, 1.0, 0.03, 0.01, 0.10,
                                OptionType::Call, 2, 7, true, true);
    EXPECT_TRUE(std::isfinite(r.price));
    EXPECT_TRUE(std::isfinite(r.std_error));
    EXPECT_DOUBLE_EQ(r.std_error, 0.0);
    EXPECT_TRUE(std::isfinite(r.ci_low));
    EXPECT_TRUE(std::isfinite(r.ci_high));
    EXPECT_EQ(r.n_paths, 2);
    // Four paths (two pairs) do produce a real error bar.
    const MCResult r4 = mc_price(1.10, 1.10, 1.0, 0.03, 0.01, 0.10,
                                 OptionType::Call, 4, 7, true, true);
    EXPECT_GT(r4.std_error, 0.0);
}

}  // namespace
