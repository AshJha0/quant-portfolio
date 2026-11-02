// Phase 5 ports: Leisen-Reimer, Bachelier, higher-order Greeks.
// Golden values are the Python reference (eq_options) printed with repr();
// the oracle tests for the models themselves live in
// python/equity/01-options-pricing/tests/test_phase5_models.py.

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <stdexcept>

#include "eqopt/bachelier.hpp"
#include "eqopt/black_scholes.hpp"
#include "eqopt/greeks.hpp"
#include "eqopt/greeks_higher.hpp"
#include "eqopt/lr_tree.hpp"

namespace {

using eqopt::ExerciseStyle;
using eqopt::OptionType;

void expect_rel(double a, double b, double tol) {
    EXPECT_NEAR(a, b, tol * std::fmax(std::fabs(b), 1e-3));
}

TEST(LeisenReimer, GoldenVsPython) {
    expect_rel(eqopt::lr_price(100, 100, 1, 0.05, 0.2, 0, OptionType::Call,
                               ExerciseStyle::European, 101),
               10.450549336567306, 1e-12);
    expect_rel(eqopt::lr_price(100, 110, 0.5, 0.03, 0.3, 0.02, OptionType::Put,
                               ExerciseStyle::European, 101),
               14.21511061827863, 1e-12);
    expect_rel(eqopt::lr_price(100, 110, 0.5, 0.03, 0.3, 0.02, OptionType::Put,
                               ExerciseStyle::American, 101),
               14.33188433420645, 1e-12);
    expect_rel(eqopt::lr_price(50, 60, 0.25, -0.01, 0.4, 0, OptionType::Call,
                               ExerciseStyle::American, 101),
               1.051752922776993, 1e-12);
}

TEST(LeisenReimer, SecondOrderConvergenceToBS) {
    const double bs = eqopt::bs_price(100, 105, 1, 0.04, 0.25, 0.01, OptionType::Call);
    const double e1 = std::fabs(eqopt::lr_price(100, 105, 1, 0.04, 0.25, 0.01, OptionType::Call,
                                                ExerciseStyle::European, 101) - bs);
    const double e2 = std::fabs(eqopt::lr_price(100, 105, 1, 0.04, 0.25, 0.01, OptionType::Call,
                                                ExerciseStyle::European, 201) - bs);
    EXPECT_GT(e1 / e2, 3.0);
    EXPECT_LT(e1 / e2, 5.0);
}

TEST(LeisenReimer, Rejects) {
    EXPECT_THROW(eqopt::peizer_pratt(0.1, 10), std::invalid_argument);
    EXPECT_THROW(eqopt::lr_price(0, 100, 1, 0.05, 0.2), std::invalid_argument);
}

TEST(Bachelier, GoldenVsPython) {
    auto g = eqopt::bachelier_greeks(1.0, 1.3, 2.0, 0.4, 0.03, OptionType::Call);
    expect_rel(g.price, 0.10047506715747369, 1e-13);
    expect_rel(g.delta, 0.2805907804284276, 1e-13);
    expect_rel(g.gamma, 0.5770384415187559, 1e-13);
    expect_rel(g.vega, 0.46163075321500496, 1e-13);
    expect_rel(g.theta, -0.04314882330677628, 1e-13);
    g = eqopt::bachelier_greeks(-0.5, 0.2, 1, 0.8, 0.01, OptionType::Put);
    expect_rel(g.price, 0.7762912746894666, 1e-13);
    expect_rel(g.delta, -0.8011612427960295, 1e-13);
    expect_rel(g.theta, -0.09997628961922823, 1e-13);
}

TEST(Bachelier, WorkedExampleAndImpliedVol) {
    EXPECT_NEAR(eqopt::bachelier_price(100, 100, 1, 20), 20.0 / std::sqrt(2.0 * std::numbers::pi),
                1e-13);
    for (double K : {-1.0, 0.5, 1.0, 1.5, 4.0}) {
        for (auto t : {OptionType::Call, OptionType::Put}) {
            const double p = eqopt::bachelier_price(1.0, K, 1.5, 0.6, 0.02, t);
            EXPECT_NEAR(eqopt::bachelier_implied_vol(p, 1.0, K, 1.5, 0.02, t), 0.6, 6e-11);
        }
    }
    EXPECT_THROW(eqopt::bachelier_implied_vol(0.0, 1, 1, 1), std::invalid_argument);
}

TEST(HigherGreeks, GoldenVsPython) {
    auto g = eqopt::higher_greeks(100, 100, 1, 0.05, 0.2, 0, OptionType::Call);
    expect_rel(g.charm, -0.06566706071046415, 1e-12);
    expect_rel(g.speed, -0.0005159554770107896, 1e-12);
    expect_rel(g.colour, 0.01053018223535657, 1e-12);
    expect_rel(g.zomma, -0.08888505717594967, 1e-12);
    expect_rel(g.veta, -16.46367022098065, 1e-12);
    g = eqopt::higher_greeks(100, 110, 0.5, 0.03, 0.3, 0.02, OptionType::Put);
    expect_rel(g.charm, -0.22965667914717408, 1e-12);
    expect_rel(g.speed, 8.967850765599677e-05, 1e-12);
    expect_rel(g.colour, 0.01477156521455538, 1e-12);
    expect_rel(g.zomma, -0.048947752194907886, 1e-12);
    expect_rel(g.veta, -30.91807129987024, 1e-12);
}

TEST(HigherGreeks, SpeedMatchesFiniteDifferenceOfGamma) {
    const double h = 1e-3;
    const auto gp = eqopt::bs_greeks(100 + h, 95, 0.7, 0.02, 0.25, 0.01, OptionType::Call);
    const auto gm = eqopt::bs_greeks(100 - h, 95, 0.7, 0.02, 0.25, 0.01, OptionType::Call);
    const auto hg = eqopt::higher_greeks(100, 95, 0.7, 0.02, 0.25, 0.01, OptionType::Call);
    EXPECT_NEAR(hg.speed, (gp.gamma - gm.gamma) / (2 * h), 1e-8);
}

}  // namespace
