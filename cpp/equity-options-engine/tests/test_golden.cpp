// Golden-vector cross-language validation: every case produced by the Python
// reference (eq_options) must be reproduced by the C++ engine to 1e-9 abs/rel
// for the price and all five first-order Greeks.
//
// Unit conventions (checked against the Python greeks module):
//   theta is dV/dt per YEAR (not per day); vega is per UNIT of vol (not per
//   vol point); rho is per UNIT of rate. The golden JSON documents the same.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "eqopt/black_scholes.hpp"
#include "eqopt/greeks.hpp"
#include "golden_vectors.hpp"

namespace {

using eqopt::golden::kCases;

// abs/rel tolerance: |a - b| <= tol * max(1, |b|)
void expect_close(double actual, double expected, double tol,
                  const char* what, std::size_t idx) {
    const double bound = tol * std::max(1.0, std::abs(expected));
    EXPECT_NEAR(actual, expected, bound)
        << what << " mismatch vs Python golden vector, case " << idx;
}

constexpr double kTol = 1e-9;

TEST(GoldenVectors, HasAllThirtyTwoCases) {
    EXPECT_EQ(kCases.size(), 32u);
}

TEST(GoldenVectors, PriceMatchesPythonReference) {
    for (std::size_t i = 0; i < kCases.size(); ++i) {
        const auto& c = kCases[i];
        const double price =
            eqopt::bs_price(c.S, c.K, c.T, c.r, c.sigma, c.q, c.type);
        expect_close(price, c.price, kTol, "price", i);
    }
}

TEST(GoldenVectors, AllGreeksMatchPythonReference) {
    for (std::size_t i = 0; i < kCases.size(); ++i) {
        const auto& c = kCases[i];
        const auto g = eqopt::bs_greeks(c.S, c.K, c.T, c.r, c.sigma, c.q, c.type);
        expect_close(g.price, c.price, kTol, "price", i);
        expect_close(g.delta, c.delta, kTol, "delta", i);
        expect_close(g.gamma, c.gamma, kTol, "gamma", i);
        expect_close(g.vega, c.vega, kTol, "vega", i);
        expect_close(g.theta, c.theta, kTol, "theta", i);
        expect_close(g.rho, c.rho, kTol, "rho", i);
    }
}

// The greeks-path price and the direct bs_price must agree exactly (same
// formula, same evaluation order) — a guard against the two drifting apart.
TEST(GoldenVectors, GreeksPriceConsistentWithBsPrice) {
    for (const auto& c : kCases) {
        const auto g = eqopt::bs_greeks(c.S, c.K, c.T, c.r, c.sigma, c.q, c.type);
        const double p = eqopt::bs_price(c.S, c.K, c.T, c.r, c.sigma, c.q, c.type);
        EXPECT_NEAR(g.price, p, 1e-12 * std::max(1.0, std::abs(p)));
    }
}

}  // namespace
