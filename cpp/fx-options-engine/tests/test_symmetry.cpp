// Foreign-domestic symmetry (notional duality).
//
// A EURUSD call (right to buy EUR paying USD) is, viewed from the EUR side,
// a USDEUR put (right to sell USD receiving EUR):
//
//   C_d(S, K, T, r_d, r_f, sigma) = S * K * P_f(1/S, 1/K, T, r_f, r_d, sigma)
//
// where the flipped option is priced with the rate roles swapped, its
// premium expressed in foreign currency and rescaled by S*K.

#include <gtest/gtest.h>

#include <cmath>

#include "fxopt/deltas.hpp"
#include "fxopt/garman_kohlhagen.hpp"

namespace {

using namespace fxopt;

TEST(ForeignDomesticSymmetry, CallEqualsFlippedPutAcrossGrid) {
    const double spots[] = {0.65, 1.10, 147.5};
    const double mults[] = {0.85, 1.0, 1.20};
    const double tenors[] = {0.1, 0.5, 2.0};
    const double rates[][2] = {{0.0425, 0.0290}, {0.0050, 0.0525},
                               {-0.0075, -0.0050}};
    const double sigma = 0.11;
    for (const double S : spots)
        for (const double m : mults)
            for (const double T : tenors)
                for (const auto& r : rates) {
                    const double K = S * m;
                    const double lhs =
                        gk_price(S, K, T, r[0], r[1], sigma, OptionType::Call);
                    const double rhs =
                        S * K * gk_price(1.0 / S, 1.0 / K, T, r[1], r[0],
                                         sigma, OptionType::Put);
                    EXPECT_NEAR(lhs, rhs, 1e-10 * std::max(1.0, S * K));
                }
}

TEST(ForeignDomesticSymmetry, PutEqualsFlippedCall) {
    const double S = 1.10, K = 1.05, T = 0.75, rd = 0.03, rf = 0.01,
                 sig = 0.09;
    const double lhs = gk_price(S, K, T, rd, rf, sig, OptionType::Put);
    const double rhs = S * K * gk_price(1.0 / S, 1.0 / K, T, rf, rd, sig,
                                        OptionType::Call);
    EXPECT_NEAR(lhs, rhs, 1e-12);
}

TEST(ForeignDomesticSymmetry, PaDeltaIsFlippedForwardDelta) {
    // PA call forward delta = (K/F) N(d2); the flipped put's unadjusted
    // forward delta is -N(d2).  This is WHY PA deltas exist: they are the
    // hedge seen from the other currency's viewpoint.
    const double S = 1.10, K = 1.15, T = 0.5, rd = 0.0425, rf = 0.0290,
                 sig = 0.0825;
    const double F = S * std::exp((rd - rf) * T);
    const double pa = delta(S, K, T, rd, rf, sig, OptionType::Call,
                            DeltaConvention::ForwardPa);
    const double flipped = delta(1.0 / S, 1.0 / K, T, rf, rd, sig,
                                 OptionType::Put, DeltaConvention::Forward);
    EXPECT_NEAR(pa, -(K / F) * flipped, 1e-12);
}

}  // namespace
