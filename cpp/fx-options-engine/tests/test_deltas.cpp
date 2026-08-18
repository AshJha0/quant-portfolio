// Delta conventions: inter-convention relations, premium adjustment,
// strike-from-delta round trips under all four conventions, PA-call branch
// selection, and ATM strike conventions.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "fxopt/deltas.hpp"
#include "fxopt/garman_kohlhagen.hpp"

namespace {

using namespace fxopt;

constexpr double S = 1.10, T = 0.5, RD = 0.0425, RF = 0.0290, SIG = 0.0925;
constexpr double K_ATM = 1.10, K_OTM_CALL = 1.16, K_OTM_PUT = 1.04;

TEST(Deltas, ForwardDeltaIsSpotDeltaTimesForeignGrowth) {
    for (const double K : {K_OTM_PUT, K_ATM, K_OTM_CALL})
        for (const auto type : {OptionType::Call, OptionType::Put}) {
            const double ds =
                delta(S, K, T, RD, RF, SIG, type, DeltaConvention::Spot);
            const double df =
                delta(S, K, T, RD, RF, SIG, type, DeltaConvention::Forward);
            EXPECT_NEAR(df, ds * std::exp(RF * T), 1e-14);
            EXPECT_NEAR(spot_to_forward_delta(ds, T, RF), df, 1e-14);
            EXPECT_NEAR(forward_to_spot_delta(df, T, RF), ds, 1e-14);
            // Same relation for the PA pair.
            const double ds_pa =
                delta(S, K, T, RD, RF, SIG, type, DeltaConvention::SpotPa);
            const double df_pa = delta(S, K, T, RD, RF, SIG, type,
                                       DeltaConvention::ForwardPa);
            EXPECT_NEAR(df_pa, ds_pa * std::exp(RF * T), 1e-14);
        }
}

TEST(Deltas, PremiumAdjustedBelowUnadjustedForCalls) {
    for (const double K : {K_OTM_PUT, K_ATM, K_OTM_CALL}) {
        const double un = delta(S, K, T, RD, RF, SIG, OptionType::Call,
                                DeltaConvention::Spot);
        const double pa = delta(S, K, T, RD, RF, SIG, OptionType::Call,
                                DeltaConvention::SpotPa);
        EXPECT_LT(pa, un);
    }
}

TEST(Deltas, PremiumAdjustmentEqualsDeltaMinusPremiumOverSpot) {
    // delta_pa = delta_spot - V/S, for both calls and puts.
    for (const auto type : {OptionType::Call, OptionType::Put}) {
        const double K = type == OptionType::Call ? K_OTM_CALL : K_OTM_PUT;
        const double v = gk_price(S, K, T, RD, RF, SIG, type);
        const double un =
            delta(S, K, T, RD, RF, SIG, type, DeltaConvention::Spot);
        const double pa =
            delta(S, K, T, RD, RF, SIG, type, DeltaConvention::SpotPa);
        EXPECT_NEAR(pa, premium_adjust_spot_delta(un, v, S), 1e-14);
    }
}

TEST(Deltas, StrikeFromDeltaRoundTripsAllConventions) {
    // Invert typical desk deltas (25d / 10d, both wings) under every
    // convention and re-evaluate: |delta(K*) - target| and strike
    // consistency to 1e-8.
    const DeltaConvention convs[] = {
        DeltaConvention::Spot, DeltaConvention::Forward,
        DeltaConvention::SpotPa, DeltaConvention::ForwardPa};
    const double call_targets[] = {0.10, 0.25};
    const double put_targets[] = {-0.10, -0.25, -0.40};
    for (const auto conv : convs) {
        for (const double target : call_targets) {
            const double K = strike_from_delta(target, S, T, RD, RF, SIG,
                                               OptionType::Call, conv);
            EXPECT_NEAR(delta(S, K, T, RD, RF, SIG, OptionType::Call, conv),
                        target, 1e-8)
                << "call conv " << static_cast<int>(conv);
        }
        for (const double target : put_targets) {
            const double K = strike_from_delta(target, S, T, RD, RF, SIG,
                                               OptionType::Put, conv);
            EXPECT_NEAR(delta(S, K, T, RD, RF, SIG, OptionType::Put, conv),
                        target, 1e-8)
                << "put conv " << static_cast<int>(conv);
        }
    }
}

TEST(Deltas, PaCallSolverPicksHighStrikeBranch) {
    // The PA call delta is not monotone in K; the market-standard root is
    // the one on the decreasing branch, above the peak of K*N(d2).
    const double target = 0.25;
    const double K = strike_from_delta(target, S, T, RD, RF, SIG,
                                       OptionType::Call,
                                       DeltaConvention::ForwardPa);
    // The returned strike must be OTM-side (above forward here) and the
    // delta must be locally decreasing at K.
    const double eps = 1e-4 * K;
    const double d_up = delta(S, K + eps, T, RD, RF, SIG, OptionType::Call,
                              DeltaConvention::ForwardPa);
    const double d_dn = delta(S, K - eps, T, RD, RF, SIG, OptionType::Call,
                              DeltaConvention::ForwardPa);
    EXPECT_LT(d_up, d_dn);  // decreasing branch
    EXPECT_GT(K, atm_forward_strike(S, T, RD, RF));
}

TEST(Deltas, PaCallDeltaAboveFoldMaximumThrows) {
    EXPECT_THROW(strike_from_delta(0.99, S, T, RD, RF, SIG, OptionType::Call,
                                   DeltaConvention::ForwardPa),
                 std::invalid_argument);
}

TEST(Deltas, StrikeFromDeltaRejectsWrongSignAndOutOfRange) {
    EXPECT_THROW(strike_from_delta(-0.25, S, T, RD, RF, SIG, OptionType::Call,
                                   DeltaConvention::Spot),
                 std::invalid_argument);
    EXPECT_THROW(strike_from_delta(0.25, S, T, RD, RF, SIG, OptionType::Put,
                                   DeltaConvention::Spot),
                 std::invalid_argument);
    EXPECT_THROW(strike_from_delta(1.5, S, T, RD, RF, SIG, OptionType::Call,
                                   DeltaConvention::Forward),
                 std::invalid_argument);
    EXPECT_THROW(strike_from_delta(0.25, S, 0.0, RD, RF, SIG,
                                   OptionType::Call, DeltaConvention::Spot),
                 std::invalid_argument);
}

TEST(Deltas, AtmForwardStrikeIsCipForward) {
    EXPECT_NEAR(atm_forward_strike(S, T, RD, RF),
                S * std::exp((RD - RF) * T), 1e-15);
}

TEST(Deltas, DnsStraddleDeltaIsZero) {
    // At the DNS strike, call delta + put delta == 0 under the chosen
    // convention (this is the definition of the ATM quote for most pairs).
    for (const auto conv :
         {DeltaConvention::Spot, DeltaConvention::Forward,
          DeltaConvention::SpotPa, DeltaConvention::ForwardPa}) {
        const double K = atm_dns_strike(S, T, RD, RF, SIG, conv);
        const double dc = delta(S, K, T, RD, RF, SIG, OptionType::Call, conv);
        const double dp = delta(S, K, T, RD, RF, SIG, OptionType::Put, conv);
        EXPECT_NEAR(dc + dp, 0.0, 1e-12) << "conv " << static_cast<int>(conv);
    }
    // Closed forms: F e^{+v^2/2} unadjusted, F e^{-v^2/2} premium-adjusted.
    const double F = S * std::exp((RD - RF) * T);
    EXPECT_NEAR(atm_dns_strike(S, T, RD, RF, SIG, DeltaConvention::Spot),
                F * std::exp(0.5 * SIG * SIG * T), 1e-15);
    EXPECT_NEAR(atm_dns_strike(S, T, RD, RF, SIG, DeltaConvention::SpotPa),
                F * std::exp(-0.5 * SIG * SIG * T), 1e-15);
}

}  // namespace
