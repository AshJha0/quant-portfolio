// Cross-language golden validation: every case in the Python-generated
// golden vector file must reproduce price and full Greek set to 1e-9.
// The vectors were produced by python/fx/01-options-pricing (scipy-based)
// with a self-consistency tolerance of 1e-10; agreement here proves the
// C++ engine and the Python reference implement identical semantics.

#include <gtest/gtest.h>

#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/greeks.hpp"
#include "golden_vectors.hpp"

namespace {

using namespace fxopt;

constexpr double kTol = 1e-9;

class GoldenCaseTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(GoldenCaseTest, PriceAndGreeksMatchPythonReference) {
    const auto& c = golden::kCases[GetParam()];

    EXPECT_NEAR(gk_price(c.S, c.K, c.T, c.r_d, c.r_f, c.sigma, c.type),
                c.price, kTol);

    const GreeksResult g =
        analytic_greeks(c.S, c.K, c.T, c.r_d, c.r_f, c.sigma, c.type);
    EXPECT_NEAR(g.price, c.price, kTol);
    EXPECT_NEAR(g.delta_spot, c.delta_spot, kTol);
    EXPECT_NEAR(g.delta_forward, c.delta_fwd, kTol);
    EXPECT_NEAR(g.gamma, c.gamma, kTol);
    EXPECT_NEAR(g.vega, c.vega, kTol);
    EXPECT_NEAR(g.theta, c.theta, kTol);
    EXPECT_NEAR(g.rho_domestic, c.rho_d, kTol);
    EXPECT_NEAR(g.rho_foreign, c.rho_f, kTol);
}

INSTANTIATE_TEST_SUITE_P(AllGoldenCases, GoldenCaseTest,
                         ::testing::Range<std::size_t>(0,
                                                       golden::kCases.size()));

TEST(GoldenSuite, HasExpectedCaseCount) {
    EXPECT_EQ(golden::kCases.size(), 30u);
}

}  // namespace
