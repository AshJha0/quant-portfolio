// CIP forwards, forward points, synthetic forwards, and the GK == Black-76
// equivalence on the CIP forward.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "fxopt/black76.hpp"
#include "fxopt/forwards.hpp"
#include "fxopt/garman_kohlhagen.hpp"

namespace {

using namespace fxopt;

TEST(Forwards, CipForwardMatchesClosedForm) {
    EXPECT_NEAR(cip_forward(1.10, 0.5, 0.0425, 0.0290),
                1.10 * std::exp((0.0425 - 0.0290) * 0.5), 1e-15);
    // Carry direction: r_d > r_f -> forward premium on the base ccy.
    EXPECT_GT(cip_forward(1.10, 1.0, 0.05, 0.01), 1.10);
    EXPECT_LT(cip_forward(1.10, 1.0, 0.01, 0.05), 1.10);
    // T = 0 -> F = S.
    EXPECT_DOUBLE_EQ(cip_forward(1.10, 0.0, 0.05, 0.01), 1.10);
}

TEST(Forwards, ForwardPointsScaleAndSign) {
    const double S = 1.10, T = 0.5, rd = 0.0425, rf = 0.0290;
    const double F = cip_forward(S, T, rd, rf);
    EXPECT_NEAR(forward_points(S, T, rd, rf), (F - S) * 1e4, 1e-12);
    EXPECT_NEAR(forward_points(147.5, T, 0.001, 0.0525, kPipFactorJpy),
                (cip_forward(147.5, T, 0.001, 0.0525) - 147.5) * 1e2, 1e-10);
    EXPECT_LT(forward_points(147.5, T, 0.001, 0.0525, kPipFactorJpy), 0.0);
    EXPECT_THROW(forward_points(S, T, rd, rf, 0.0), std::invalid_argument);
}

TEST(Forwards, SyntheticForwardRecoversCipForward) {
    const double S = 1.10, K = 1.12, T = 0.75, rd = 0.0425, rf = 0.0290,
                 sig = 0.0925;
    const double c = gk_call(S, K, T, rd, rf, sig);
    const double p = gk_put(S, K, T, rd, rf, sig);
    EXPECT_NEAR(synthetic_forward_from_options(c, p, K, T, rd),
                cip_forward(S, T, rd, rf), 1e-12);
}

TEST(Black76, EqualsGarmanKohlhagenOnCipForward) {
    // GK == Black-76 on the CIP forward, to 1e-12, across a grid.
    const double spots[] = {0.65, 1.10, 147.5};
    const double mults[] = {0.85, 1.0, 1.20};
    const double tenors[] = {0.1, 0.5, 2.0};
    const double rates[][2] = {{0.0425, 0.0290}, {0.0050, 0.0525},
                               {-0.0075, -0.0050}};
    for (const double S : spots)
        for (const double m : mults)
            for (const double T : tenors)
                for (const auto& r : rates)
                    for (const auto type :
                         {OptionType::Call, OptionType::Put}) {
                        const double K = S * m;
                        const double gk =
                            gk_price(S, K, T, r[0], r[1], 0.11, type);
                        const double b76 = black76_from_spot(
                            S, K, T, r[0], r[1], 0.11, type);
                        EXPECT_NEAR(gk, b76, 1e-12 * std::max(1.0, S));
                    }
}

TEST(Black76, ZeroVolAndValidation) {
    EXPECT_NEAR(black76_price(1.12, 1.10, 0.5, 0.04, 0.0, OptionType::Call),
                std::exp(-0.02) * 0.02, 1e-15);
    EXPECT_THROW(black76_price(-1.0, 1.1, 0.5, 0.04, 0.1, OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(black76_price(1.1, 1.1, -0.5, 0.04, 0.1, OptionType::Put),
                 std::invalid_argument);
}

}  // namespace
