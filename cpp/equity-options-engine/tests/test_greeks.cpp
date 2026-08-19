// Analytic Greeks vs central finite differences (template fd_greeks applied
// to the analytic pricer), plus basic sanity identities.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "eqopt/black_scholes.hpp"
#include "eqopt/greeks.hpp"

namespace {

using eqopt::bs_greeks;
using eqopt::bs_price;
using eqopt::fd_greeks;
using eqopt::OptionType;

// rel tolerance with absolute floor: |a-b| <= tol * max(|b|, floor)
void expect_rel(double actual, double expected, double tol, double floor,
                const char* what) {
    EXPECT_NEAR(actual, expected, tol * std::max(std::abs(expected), floor))
        << what;
}

struct Case {
    double S, K, T, r, q, sigma;
    OptionType type;
};

const std::vector<Case> kCases = {
    {100.0, 100.0, 1.0, 0.05, 0.00, 0.20, OptionType::Call},
    {100.0, 100.0, 1.0, 0.05, 0.00, 0.20, OptionType::Put},
    {100.0, 120.0, 0.5, 0.03, 0.02, 0.35, OptionType::Call},
    {100.0, 80.0, 2.0, 0.02, 0.04, 0.15, OptionType::Put},
    {50.0, 55.0, 0.25, -0.01, 0.00, 0.45, OptionType::Call},
};

TEST(FdGreeks, MatchAnalyticTo1e6Rel) {
    for (const auto& c : kCases) {
        const auto ana = bs_greeks(c.S, c.K, c.T, c.r, c.sigma, c.q, c.type);
        // Second-derivative bump 1e-4: balances truncation O(h^2) against
        // round-off O(ulp-noise/h^2) so both are < 1e-6 rel here, keeping the full
        // Greek set including vanna/volga inside tolerance.
        const auto num = fd_greeks(
            [](double S, double K, double T, double r, double sigma, double q,
               OptionType type) {
                return bs_price(S, K, T, r, sigma, q, type);
            },
            c.S, c.K, c.T, c.r, c.sigma, c.q, c.type, 1e-5, 1e-4);
        expect_rel(num.price, ana.price, 1e-12, 1.0, "price");
        expect_rel(num.delta, ana.delta, 1e-6, 1e-2, "delta");
        expect_rel(num.gamma, ana.gamma, 1e-6, 1e-2, "gamma");
        expect_rel(num.vega, ana.vega, 1e-6, 1e-2, "vega");
        expect_rel(num.theta, ana.theta, 1e-6, 1e-2, "theta");
        expect_rel(num.rho, ana.rho, 1e-6, 1e-2, "rho");
        expect_rel(num.vanna, ana.vanna, 1e-6, 1e-2, "vanna");
        expect_rel(num.volga, ana.volga, 1e-6, 1e-1, "volga");
    }
}

TEST(Greeks, CallPutIdentities) {
    const double S = 100.0, K = 105.0, T = 0.75, r = 0.04, q = 0.02,
                 sigma = 0.25;
    const auto call = bs_greeks(S, K, T, r, sigma, q, OptionType::Call);
    const auto put = bs_greeks(S, K, T, r, sigma, q, OptionType::Put);
    // Gamma / vega / vanna / volga are identical for calls and puts.
    EXPECT_DOUBLE_EQ(call.gamma, put.gamma);
    EXPECT_DOUBLE_EQ(call.vega, put.vega);
    EXPECT_DOUBLE_EQ(call.vanna, put.vanna);
    EXPECT_DOUBLE_EQ(call.volga, put.volga);
    // delta_call - delta_put = e^{-qT} (parity in delta).
    EXPECT_NEAR(call.delta - put.delta, std::exp(-q * T), 1e-14);
    // rho_call - rho_put = K T e^{-rT}.
    EXPECT_NEAR(call.rho - put.rho, K * T * std::exp(-r * T), 1e-10);
}

TEST(Greeks, RangesAndSigns) {
    const auto g = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.01,
                             OptionType::Call);
    EXPECT_GT(g.delta, 0.0);
    EXPECT_LT(g.delta, 1.0);
    EXPECT_GT(g.gamma, 0.0);
    EXPECT_GT(g.vega, 0.0);
    EXPECT_LT(g.theta, 0.0);  // long ATM call decays
    EXPECT_GT(g.rho, 0.0);
    const auto p = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.01,
                             OptionType::Put);
    EXPECT_LT(p.delta, 0.0);
    EXPECT_LT(p.rho, 0.0);
}

TEST(Greeks, DegenerateInputsThrow) {
    EXPECT_THROW(bs_greeks(100.0, 100.0, 0.0, 0.05, 0.2),
                 std::invalid_argument);
    EXPECT_THROW(bs_greeks(100.0, 100.0, 1.0, 0.05, 0.0),
                 std::invalid_argument);
    EXPECT_THROW(bs_greeks(-1.0, 100.0, 1.0, 0.05, 0.2),
                 std::invalid_argument);
}

TEST(FdGreeks, TinySigmaThrowsOnVegaBump) {
    // sigma smaller than the central vega/volga bump must be rejected with a
    // clear message, not silently priced at negative volatility.
    EXPECT_THROW(fd_greeks(
                     [](double S, double K, double T, double r, double sigma,
                        double q, OptionType type) {
                         return bs_price(S, K, T, r, sigma, q, type);
                     },
                     100.0, 100.0, 1.0, 0.05, 1e-6, 0.0, OptionType::Call),
                 std::invalid_argument);
}

TEST(FdGreeks, TinyExpiryThrowsOnThetaBump) {
    // T smaller than the central theta bump must be rejected, not silently
    // priced at negative expiry.
    EXPECT_THROW(fd_greeks(
                     [](double S, double K, double T, double r, double sigma,
                        double q, OptionType type) {
                         return bs_price(S, K, T, r, sigma, q, type);
                     },
                     100.0, 100.0, 1e-9, 0.05, 0.2, 0.0, OptionType::Call),
                 std::invalid_argument);
}

}  // namespace
