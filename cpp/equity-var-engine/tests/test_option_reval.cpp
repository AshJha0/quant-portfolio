// Option revaluation: golden values from eq_var.revaluation (Python, scipy
// normal CDF), put-call parity, delta-gamma exact for zero moves.
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "eqvar/option_reval.hpp"

using namespace eqvar;

namespace {
const OptionLeg kCall{10.0, OptionKind::Call, 100.0, 105.0, 0.5, 0.03, 0.01, 0.25, 0};
const OptionLeg kPut{-7.0, OptionKind::Put, 50.0, 45.0, 1.25, 0.02, 0.04, 0.35, 1};
const std::vector<double> kX{0.01, -0.02, -0.05, 0.03, 0.0, 0.0};
const std::vector<double> kDv{0.01, 0.0, 0.05, -0.02, 0.0, 0.0};
constexpr double kTol = 1e-12;
}  // namespace

TEST(OptionReval, PriceGolden) {
    EXPECT_NEAR(bsm_price(100, 105, 0.5, 0.03, 0.01, 0.25, OptionKind::Call), 5.349804578881077,
                kTol);
    EXPECT_NEAR(bsm_price(50, 45, 1.25, 0.02, 0.04, 0.35, OptionKind::Put), 5.403708192910557,
                kTol);
}

TEST(OptionReval, PutCallParity) {
    const double c = bsm_price(100, 105, 0.5, 0.03, 0.01, 0.25, OptionKind::Call);
    const double p = bsm_price(100, 105, 0.5, 0.03, 0.01, 0.25, OptionKind::Put);
    EXPECT_NEAR(c - p, 100 * std::exp(-0.01 * 0.5) - 105 * std::exp(-0.03 * 0.5), 1e-12);
}

TEST(OptionReval, GreeksGolden) {
    const BsmGreeks g = bsm_greeks(kCall);
    EXPECT_NEAR(g.delta, 0.4456372919675482, kTol);
    EXPECT_NEAR(g.gamma, 0.022263053570122843, kTol);
    EXPECT_NEAR(g.vega, 27.828816962653555, 1e-11);
    EXPECT_NEAR(g.vanna, 0.484579085972438, kTol);
    EXPECT_NEAR(g.volga, 4.490141631849943, 1e-11);
    EXPECT_NEAR(g.theta, -7.687984687232052, 1e-11);
}

TEST(OptionReval, FullRevaluationGolden) {
    const std::vector<OptionLeg> legs{kCall, kPut};
    const auto pnl = full_revaluation_pnl(legs, kX, kDv, 2, 1.0 / 365.0);
    ASSERT_EQ(pnl.size(), 3u);
    EXPECT_NEAR(pnl[0], 4.92183106078723, 1e-11);
    EXPECT_NEAR(pnl[1], -0.6845739357538516, 1e-11);
    EXPECT_NEAR(pnl[2], -0.1540808210950111, 1e-11);
}

TEST(OptionReval, DeltaGammaGolden) {
    const std::vector<OptionLeg> legs{kCall, kPut};
    const auto pnl = delta_gamma_pnl(legs, kX, 2);
    EXPECT_NEAR(pnl[0], 2.260614231417638, 1e-11);
    EXPECT_NEAR(pnl[1], -15.741558009225477, 1e-11);
    EXPECT_EQ(pnl[2], 0.0);
}

TEST(OptionReval, RejectsBadInput) {
    const std::vector<OptionLeg> legs{kPut};
    const std::vector<double> one_factor{0.0, 0.0};
    EXPECT_THROW((void)full_revaluation_pnl(legs, one_factor, {}, 1), std::invalid_argument);
    const std::vector<double> nan{std::nan(""), 0.0};
    EXPECT_THROW((void)delta_gamma_pnl(legs, nan, 2), std::invalid_argument);
    EXPECT_THROW((void)bsm_price(100, 100, 1, 0, 0, 0.0, OptionKind::Call), std::invalid_argument);
}

// Round 4 vol modes: golden values from {eq,fx}_var.revaluation
// full_revaluation_pnl(..., vol_mode=...) with skews (-0.1, 0.15)
// (tests/test_round4_vol_modes.py in both Python packages).
TEST(OptionReval, VolModesGolden) {
    const std::vector<OptionLeg> legs{kCall, kPut};
    const std::vector<double> skews{-0.1, 0.15};
    const double dt = 1.0 / 365.0;
    const auto ss = full_revaluation_pnl(legs, kX, kDv, 2, dt, VolMode::StickyStrike, skews);
    const auto sd = full_revaluation_pnl(legs, kX, kDv, 2, dt, VolMode::StickyDelta, skews);
    const auto ha = full_revaluation_pnl(legs, kX, kDv, 2, dt, VolMode::HistoricalAtm, skews);
    const double gss[3] = {4.92183106078723, -0.6845739357538516, -0.1540808210950111};
    const double gsd[3] = {4.793674377411101, -1.3383700871131143, -0.1540808210950111};
    const double gha[3] = {2.8099164421405085, -11.777009547269026, -0.1540808210950111};
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(ss[i], gss[i], 1e-12 * (1.0 + std::abs(gss[i])));
        EXPECT_NEAR(sd[i], gsd[i], 1e-12 * (1.0 + std::abs(gsd[i])));
        EXPECT_NEAR(ha[i], gha[i], 1e-12 * (1.0 + std::abs(gha[i])));
    }
    // The old overload is sticky strike with no skews, bit for bit.
    const auto old = full_revaluation_pnl(legs, kX, kDv, 2, dt);
    const auto ss0 = full_revaluation_pnl(legs, kX, kDv, 2, dt, VolMode::StickyStrike, {});
    for (int i = 0; i < 3; ++i) EXPECT_EQ(old[i], ss0[i]);
    // Sticky delta with flat smiles is sticky strike.
    const auto sd0 = full_revaluation_pnl(legs, kX, kDv, 2, dt, VolMode::StickyDelta, {});
    for (int i = 0; i < 3; ++i) EXPECT_EQ(sd0[i], ss0[i]);
}

TEST(OptionReval, VolModesRejectBadSkews) {
    const std::vector<OptionLeg> legs{kCall, kPut};
    const std::vector<double> one{0.1};
    const std::vector<double> nan{0.1, std::nan("")};
    EXPECT_THROW((void)full_revaluation_pnl(legs, kX, kDv, 2, 0.0, VolMode::StickyDelta, one),
                 std::invalid_argument);
    EXPECT_THROW((void)full_revaluation_pnl(legs, kX, kDv, 2, 0.0, VolMode::StickyDelta, nan),
                 std::invalid_argument);
}
