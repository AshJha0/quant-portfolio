// Implied vol: round trips across moneyness/tenor/vol grids, no-arbitrage
// rejection, and degenerate time-value behaviour.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/implied_vol.hpp"

namespace {

using namespace fxopt;

constexpr double S = 1.10, RD = 0.0425, RF = 0.0290;

TEST(ImpliedVol, RoundTripAcrossGrid) {
    // The Python reference's representative grid (well-conditioned points:
    // vega healthy, time value far above double-precision resolution) plus
    // a high-vol wing case.  Round trip price -> vol to 1e-10.
    const struct {
        double s, k, t, rd, rf, sig;
    } grid[] = {
        {1.10, 1.10, 0.5, 0.0425, 0.0290, 0.0825},
        {1.10, 1.00, 0.25, 0.0425, 0.0290, 0.0825},
        {1.10, 1.25, 1.0, 0.0425, 0.0290, 0.0825},
        {147.5, 147.5, 0.5, 0.0050, 0.0525, 0.1075},
        {147.5, 130.0, 2.0, 0.0050, 0.0525, 0.1075},
        {1.08, 1.08, 1.0, -0.0075, -0.0050, 0.065},
        {18.5, 20.0, 0.25, 0.1125, 0.045, 0.35},
        {18.5, 22.0, 0.25, 0.1125, 0.045, 0.60},
    };
    for (const auto& g : grid)
        for (const auto type : {OptionType::Call, OptionType::Put}) {
            const double price =
                gk_price(g.s, g.k, g.t, g.rd, g.rf, g.sig, type);
            const double iv =
                implied_vol(price, g.s, g.k, g.t, g.rd, g.rf, type);
            EXPECT_NEAR(iv, g.sig, 1e-10)
                << "S=" << g.s << " K=" << g.k << " T=" << g.t
                << " sig=" << g.sig;
        }
}

TEST(ImpliedVol, RoundTripWithNegativeRates) {
    const double price = gk_price(0.93, 0.95, 1.0, -0.0075, -0.0050, 0.065,
                                  OptionType::Put);
    EXPECT_NEAR(implied_vol(price, 0.93, 0.95, 1.0, -0.0075, -0.0050,
                            OptionType::Put),
                0.065, 1e-10);
}

TEST(ImpliedVol, RejectsArbitragePrices) {
    // Below intrinsic and above the discounted-spot bound both throw.
    EXPECT_THROW(implied_vol(-0.01, S, 1.10, 0.5, RD, RF, OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(
        implied_vol(S * std::exp(-RF * 0.5) + 0.01, S, 1.10, 0.5, RD, RF,
                    OptionType::Call),
        std::invalid_argument);
    EXPECT_THROW(implied_vol(0.02, S, 1.10, 0.0, RD, RF, OptionType::Call),
                 std::invalid_argument);
}

TEST(ImpliedVol, ZeroTimeValueReturnsZeroVol) {
    // Deep ITM with price exactly at the sigma->0 limit: vol unrecoverable,
    // returns 0 by documented convention (matches the Python reference).
    // Build `lower` with the same arithmetic implied_vol uses internally
    // (F = S df_f / df_d) so the comparison is exact to the last bit.
    const double K = 0.80, T = 0.25;
    const double df_d = std::exp(-RD * T);
    const double df_f = std::exp(-RF * T);
    const double lower = df_d * (S * df_f / df_d - K);
    EXPECT_DOUBLE_EQ(implied_vol(lower, S, K, T, RD, RF, OptionType::Call),
                     0.0);
}

}  // namespace
