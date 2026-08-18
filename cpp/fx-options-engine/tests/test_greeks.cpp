// Analytic Greeks vs central finite differences, plus FX-specific sign and
// consistency checks (two rhos, vanna/volga).

#include <gtest/gtest.h>

#include <cmath>

#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/greeks.hpp"

namespace {

using namespace fxopt;

constexpr double S = 1.10, K = 1.12, T = 0.5, RD = 0.0425, RF = 0.0290,
                 SIG = 0.0925;

TEST(Greeks, AnalyticMatchesFiniteDifferences) {
    for (const auto type : {OptionType::Call, OptionType::Put}) {
        const GreeksResult a = analytic_greeks(S, K, T, RD, RF, SIG, type);
        const FDGreeks fd = finite_difference_greeks(S, K, T, RD, RF, SIG,
                                                     type);
        EXPECT_NEAR(a.delta_spot, fd.delta_spot, 1e-6);
        EXPECT_NEAR(a.gamma, fd.gamma, 1e-4);  // second difference, O(1) value
        EXPECT_NEAR(a.vega, fd.vega, 1e-6);
        EXPECT_NEAR(a.theta, fd.theta, 1e-6);
        EXPECT_NEAR(a.rho_domestic, fd.rho_domestic, 1e-6);
        EXPECT_NEAR(a.rho_foreign, fd.rho_foreign, 1e-6);
        EXPECT_NEAR(a.vanna, fd.vanna, 1e-5);
        EXPECT_NEAR(a.volga, fd.volga, 1e-5);
    }
}

TEST(Greeks, TemplateComparatorAcceptsArbitraryPricer) {
    // The FD comparator is templated on the pricing function: feed it the
    // GK pricer as a lambda and check delta against the closed form.
    const auto pricer = [](double s, double k, double t, double rd, double rf,
                           double sig) {
        return gk_price(s, k, t, rd, rf, sig, OptionType::Call);
    };
    const FDGreeks fd = finite_difference_greeks(pricer, S, K, T, RD, RF, SIG);
    const GreeksResult a =
        analytic_greeks(S, K, T, RD, RF, SIG, OptionType::Call);
    EXPECT_NEAR(fd.delta_spot, a.delta_spot, 1e-6);
    EXPECT_NEAR(fd.vega, a.vega, 1e-6);
}

TEST(Greeks, RhoSigns) {
    const GreeksResult call =
        analytic_greeks(S, K, T, RD, RF, SIG, OptionType::Call);
    const GreeksResult put =
        analytic_greeks(S, K, T, RD, RF, SIG, OptionType::Put);
    // Calls: higher r_d lifts the forward (+), higher r_f is a larger
    // 'dividend' on the base currency (-).  Puts are reversed.
    EXPECT_GT(call.rho_domestic, 0.0);
    EXPECT_LT(call.rho_foreign, 0.0);
    EXPECT_LT(put.rho_domestic, 0.0);
    EXPECT_GT(put.rho_foreign, 0.0);
}

TEST(Greeks, CallPutShareSymmetricGreeks) {
    // Gamma, vega, vanna, volga are identical for calls and puts.
    const GreeksResult c =
        analytic_greeks(S, K, T, RD, RF, SIG, OptionType::Call);
    const GreeksResult p =
        analytic_greeks(S, K, T, RD, RF, SIG, OptionType::Put);
    EXPECT_NEAR(c.gamma, p.gamma, 1e-15);
    EXPECT_NEAR(c.vega, p.vega, 1e-15);
    EXPECT_NEAR(c.vanna, p.vanna, 1e-15);
    EXPECT_NEAR(c.volga, p.volga, 1e-15);
    // Standalone functions agree with the bundled result.
    EXPECT_NEAR(gamma(S, K, T, RD, RF, SIG), c.gamma, 1e-15);
    EXPECT_NEAR(vega(S, K, T, RD, RF, SIG), c.vega, 1e-15);
    EXPECT_NEAR(vanna(S, K, T, RD, RF, SIG), c.vanna, 1e-15);
    EXPECT_NEAR(volga(S, K, T, RD, RF, SIG), c.volga, 1e-15);
}

TEST(Greeks, GammaAndVegaArePositive) {
    for (const double k : {0.95, 1.10, 1.30}) {
        EXPECT_GT(gamma(S, k, T, RD, RF, SIG), 0.0);
        EXPECT_GT(vega(S, k, T, RD, RF, SIG), 0.0);
    }
}

TEST(Greeks, RequiresPositiveTimeAndVol) {
    EXPECT_THROW(analytic_greeks(S, K, 0.0, RD, RF, SIG, OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(analytic_greeks(S, K, T, RD, RF, 0.0, OptionType::Call),
                 std::invalid_argument);
}

}  // namespace
