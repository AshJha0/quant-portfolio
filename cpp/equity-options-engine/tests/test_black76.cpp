// Black-76: exact equivalence with Black-Scholes at F = S e^{(r-q)T},
// forward-space put-call parity, Greeks sanity vs finite differences, and
// edge cases.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "eqopt/black76.hpp"
#include "eqopt/black_scholes.hpp"

namespace {

using eqopt::black76_greeks;
using eqopt::black76_price;
using eqopt::bs_price;
using eqopt::forward_price;
using eqopt::OptionType;

TEST(Black76, EqualsBlackScholesOnForwardTo1e12) {
    const std::vector<double> spots{50.0, 100.0, 250.0};
    const std::vector<double> strikes{80.0, 100.0, 130.0};
    const std::vector<double> expiries{0.1, 1.0, 3.0};
    const std::vector<double> vols{0.1, 0.3};
    const std::vector<double> rates{-0.01, 0.04};
    const std::vector<double> divs{0.0, 0.03};
    for (double S : spots)
        for (double K : strikes)
            for (double T : expiries)
                for (double sigma : vols)
                    for (double r : rates)
                        for (double q : divs)
                            for (OptionType type :
                                 {OptionType::Call, OptionType::Put}) {
                                const double F = forward_price(S, T, r, q);
                                const double b76 =
                                    black76_price(F, K, T, r, sigma, type);
                                const double bs =
                                    bs_price(S, K, T, r, sigma, q, type);
                                ASSERT_NEAR(b76, bs,
                                            1e-12 * std::max(1.0, bs))
                                    << "S=" << S << " K=" << K << " T=" << T
                                    << " sigma=" << sigma << " r=" << r
                                    << " q=" << q;
                            }
}

TEST(Black76, ForwardParity) {
    // C - P = e^{-rT} (F - K) to 1e-12.
    const double F = 105.0, K = 98.0, T = 1.4, r = 0.035, sigma = 0.27;
    const double c = black76_price(F, K, T, r, sigma, OptionType::Call);
    const double p = black76_price(F, K, T, r, sigma, OptionType::Put);
    EXPECT_NEAR(c - p, std::exp(-r * T) * (F - K), 1e-12);
}

TEST(Black76, GreeksMatchCentralDifferences) {
    const double F = 100.0, K = 95.0, T = 0.8, r = 0.04, sigma = 0.3;
    for (OptionType type : {OptionType::Call, OptionType::Put}) {
        const auto g = black76_greeks(F, K, T, r, sigma, type);
        const double h_f = 1e-5 * F;
        const double h_v = 1e-5;
        const double h_t = 1e-5 * T;
        const double h_r = 1e-5;
        const auto price = [&](double f, double t, double rr, double sig) {
            return black76_price(f, K, t, rr, sig, type);
        };
        EXPECT_NEAR(g.delta,
                    (price(F + h_f, T, r, sigma) - price(F - h_f, T, r, sigma)) /
                        (2 * h_f),
                    1e-7);
        const double h_f2 = 2e-4 * F;
        EXPECT_NEAR(g.gamma,
                    (price(F + h_f2, T, r, sigma) - 2 * g.price +
                     price(F - h_f2, T, r, sigma)) /
                        (h_f2 * h_f2),
                    1e-7);
        EXPECT_NEAR(g.vega,
                    (price(F, T, r, sigma + h_v) - price(F, T, r, sigma - h_v)) /
                        (2 * h_v),
                    1e-5);
        EXPECT_NEAR(g.theta,
                    -(price(F, T + h_t, r, sigma) - price(F, T - h_t, r, sigma)) /
                        (2 * h_t),
                    1e-5);
        EXPECT_NEAR(g.rho,
                    (price(F, T, r + h_r, sigma) - price(F, T, r - h_r, sigma)) /
                        (2 * h_r),
                    1e-5);
        // rho is pure discounting: -T * V by construction.
        EXPECT_NEAR(g.rho, -T * g.price, 1e-12);
    }
}

TEST(Black76, EdgeCasesAndValidation) {
    // T == 0 -> intrinsic (undiscounted).
    EXPECT_DOUBLE_EQ(black76_price(105.0, 100.0, 0.0, 0.05, 0.2), 5.0);
    // sigma == 0 -> discounted intrinsic.
    EXPECT_DOUBLE_EQ(black76_price(105.0, 100.0, 1.0, 0.05, 0.0),
                     std::exp(-0.05) * 5.0);
    // Daily-margined futures options: r = 0 (no premium discounting).
    const double undisc = black76_price(105.0, 100.0, 1.0, 0.0, 0.2);
    EXPECT_GT(undisc, black76_price(105.0, 100.0, 1.0, 0.05, 0.2));
    EXPECT_THROW(black76_price(-1.0, 100.0, 1.0, 0.05, 0.2),
                 std::invalid_argument);
    EXPECT_THROW(black76_greeks(100.0, 100.0, 0.0, 0.05, 0.2),
                 std::invalid_argument);
}

}  // namespace
