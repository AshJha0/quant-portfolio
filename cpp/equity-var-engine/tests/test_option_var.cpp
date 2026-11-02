// Option VaR: full revaluation wired into historical simulation and Monte
// Carlo. Golden values from tools/gen_option_var_golden.py (Python reference
// eq_var.revaluation.full_revaluation_pnl + eq_var._tail.var_es); the MC
// check is against the closed-form quantile of a monotone one-factor book.
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "eqvar/historical.hpp"
#include "eqvar/monte_carlo.hpp"
#include "eqvar/option_reval.hpp"

using namespace eqvar;

namespace {
constexpr std::size_t kN = 250;
const std::vector<OptionLeg> kLegs{
    {10.0, OptionKind::Call, 100.0, 105.0, 0.5, 0.03, 0.01, 0.25, 0},
    {-5.0, OptionKind::Put, 50.0, 48.0, 0.25, 0.03, 0.0, 0.30, 1},
};
const std::vector<double> kW{1000.0, -500.0};

Matrix panel() {
    Matrix m(kN, 2);
    for (std::size_t i = 0; i < kN; ++i) {
        const double t = static_cast<double>(i);
        m.data()[i * 2] = 0.02 * std::sin(1.3 * t + 0.4);
        m.data()[i * 2 + 1] = 0.015 * std::cos(0.7 * t) + 0.005 * std::sin(2.9 * t);
    }
    return m;
}

void expect_rel(double got, double want, double rel = 1e-11) {
    EXPECT_NEAR(got, want, rel * std::abs(want)) << "want " << want;
}
}  // namespace

TEST(OptionVar, BookPnlMatchesPython) {
    const Matrix x = panel();
    const auto pnl = option_book_pnl(kLegs, kW, x);
    ASSERT_EQ(pnl.size(), kN);
    expect_rel(pnl[0], 5.077728139045078);
    expect_rel(pnl[7], -5.395781243949972, 1e-10);
}

TEST(OptionVar, HistoricalOptionsOnlyMatchesPython) {
    const TailRisk t = historical_option_var_es(kLegs, {}, panel(), 0.01);
    expect_rel(t.var, 9.667169426304056);
    expect_rel(t.es, 9.719324275447953);
}

TEST(OptionVar, HistoricalMixedMatchesPython) {
    const TailRisk t = historical_option_var_es(kLegs, kW, panel(), 0.01);
    expect_rel(t.var, 33.444115781270625);
    expect_rel(t.es, 34.649898183703485);
    const TailRisk a = historical_option_var_es(kLegs, kW, panel(), 0.01, 1.0 / 252.0);
    expect_rel(a.var, 33.63830117790198);
    expect_rel(a.es, 34.84490535004737);
}

TEST(OptionVar, NoLegsIsTheLinearPath) {
    const Matrix x = panel();
    std::vector<double> lin(kN);
    for (std::size_t i = 0; i < kN; ++i)
        lin[i] = kW[0] * x.data()[2 * i] + kW[1] * x.data()[2 * i + 1];
    const TailRisk want = var_es_order_statistic(lin, 0.01);
    const TailRisk got = historical_option_var_es({}, kW, x, 0.01);
    EXPECT_EQ(got.var, want.var);
    EXPECT_EQ(got.es, want.es);
    EXPECT_EQ(got.var, historical_var(lin, 0.01));

    Matrix cov(2, 2);
    cov.data()[0] = 1e-4;
    cov.data()[1] = cov.data()[2] = 2e-5;
    cov.data()[3] = 4e-4;
    const auto a = monte_carlo_option_var({}, kW, cov, 0.01, 5000, Dist::Normal, 6.0, 7, 50);
    const auto b = monte_carlo_var(kW, cov, 0.01, 5000, Dist::Normal, 6.0, 7, 50);
    EXPECT_EQ(a.var, b.var);
    EXPECT_EQ(a.es, b.es);
    EXPECT_EQ(a.var_se, b.var_se);
}

TEST(OptionVar, MonteCarloMatchesClosedFormQuantile) {
    const std::vector<OptionLeg> leg{
        {10.0, OptionKind::Call, 100.0, 100.0, 0.25, 0.02, 0.0, 0.2, 0}};
    Matrix cov(1, 1);
    cov.data()[0] = 1e-4;  // daily sigma 1 %
    const auto r = monte_carlo_option_var(leg, {}, cov, 0.01, 200'000, Dist::Normal, 6.0, 11, 200);
    EXPECT_EQ(r.n_paths, 200'000u);
    EXPECT_GT(r.var_se, 0.0);
    EXPECT_NEAR(r.var, 11.352269313193162, 4.0 * r.var_se);
    EXPECT_NEAR(r.es, 12.787892115047633, 0.02 * 12.787892115047633);
    // Determinism in the seed.
    const auto r2 = monte_carlo_option_var(leg, {}, cov, 0.01, 200'000, Dist::Normal, 6.0, 11, 200);
    EXPECT_EQ(r.var, r2.var);
}

TEST(OptionVar, ConvexityMakesLongOptionVarBelowDelta) {
    // A long call's full-revaluation VaR is below its delta-normal VaR
    // (positive gamma cuts the downside); a short call's is above.
    const std::vector<OptionLeg> lng{
        {10.0, OptionKind::Call, 100.0, 100.0, 0.25, 0.02, 0.0, 0.2, 0}};
    std::vector<OptionLeg> sht = lng;
    sht[0].quantity = -10.0;
    const double delta = bsm_greeks(lng[0]).delta * 10.0 * 100.0;
    Matrix cov(1, 1);
    cov.data()[0] = 4e-4;
    const auto rl = monte_carlo_option_var(lng, {}, cov, 0.01, 50'000, Dist::Normal, 6.0, 3, 50);
    const auto rs = monte_carlo_option_var(sht, {}, cov, 0.01, 50'000, Dist::Normal, 6.0, 3, 50);
    const auto rd =
        monte_carlo_var(std::vector<double>{delta}, cov, 0.01, 50'000, Dist::Normal, 6.0, 3, 50);
    EXPECT_LT(rl.var, rd.var);
    EXPECT_GT(rs.var, rd.var);
}

TEST(OptionVar, RejectsBadInput) {
    const Matrix x = panel();
    Matrix shortx(10, 2);
    EXPECT_THROW((void)historical_option_var_es(kLegs, kW, shortx, 0.01), std::invalid_argument);
    EXPECT_THROW((void)historical_option_var_es(kLegs, std::vector<double>{1.0}, x, 0.01),
                 std::invalid_argument);
    EXPECT_THROW((void)historical_option_var_es({}, {}, x, 0.01), std::invalid_argument);
    EXPECT_THROW((void)historical_option_var_es(kLegs, kW, x, 0.7), std::invalid_argument);
    EXPECT_THROW((void)historical_option_var_es(kLegs, kW, x, 0.01, -1.0), std::invalid_argument);
    std::vector<OptionLeg> bad = kLegs;
    bad[1].factor = 5;
    EXPECT_THROW((void)historical_option_var_es(bad, kW, x, 0.01), std::invalid_argument);
    Matrix cov(2, 2);
    EXPECT_THROW((void)monte_carlo_option_var(kLegs, std::vector<double>{1.0}, cov),
                 std::invalid_argument);
}

// ---- Round 4: vol scenarios on the combined axis (spot block, then vol block).
namespace {
Matrix combined_panel(bool zero_vol = false) {
    const Matrix x = panel();
    Matrix c(kN, 4);
    for (std::size_t i = 0; i < kN; ++i) {
        const double t = static_cast<double>(i);
        c.data()[i * 4] = x.data()[i * 2];
        c.data()[i * 4 + 1] = x.data()[i * 2 + 1];
        c.data()[i * 4 + 2] = zero_vol ? 0.0 : 0.03 * std::sin(0.9 * t + 0.2);
        c.data()[i * 4 + 3] = zero_vol ? 0.0 : 0.04 * std::cos(1.7 * t);
    }
    return c;
}
const std::vector<double> kSkews{-0.1, -0.2};
}  // namespace

TEST(OptionVarVol, HistoricalMatchesPythonAllModes) {
    const Matrix c = combined_panel();
    const struct {
        VolMode mode;
        double var, es;
    } want[] = {{VolMode::StickyStrike, 39.22448503697698, 40.96304558554884},
                {VolMode::StickyDelta, 39.85818324867974, 41.62283824144462},
                {VolMode::HistoricalAtm, 34.80398626392455, 35.97429554496573}};
    for (const auto& w : want) {
        const TailRisk t = historical_option_var_es_vol(kLegs, kW, c, 0.01, 0.0, w.mode, kSkews);
        expect_rel(t.var, w.var);
        expect_rel(t.es, w.es);
    }
    const auto pnl = option_book_pnl_vol(kLegs, kW, c, 0.0, VolMode::StickyDelta, kSkews);
    expect_rel(pnl[0], 5.032911200617955);
    expect_rel(pnl[7], -5.154584861517498, 1e-10);
}

TEST(OptionVarVol, ZeroVolBlockStickyStrikeIsTheOldPath) {
    const Matrix c = combined_panel(true);
    const TailRisk a =
        historical_option_var_es_vol(kLegs, kW, c, 0.01, 1.0 / 252.0, VolMode::StickyStrike);
    const TailRisk b = historical_option_var_es(kLegs, kW, panel(), 0.01, 1.0 / 252.0);
    EXPECT_EQ(a.var, b.var);
    EXPECT_EQ(a.es, b.es);
}

TEST(OptionVarVol, MonteCarloMatchesClosedFormVolQuantile) {
    const std::vector<OptionLeg> leg{
        {10.0, OptionKind::Call, 100.0, 100.0, 0.25, 0.02, 0.0, 0.2, 0}};
    Matrix cov(2, 2);
    cov.data()[0] = 1e-12;   // spot frozen
    cov.data()[3] = 0.0025;  // ATM log vol change, daily sigma 5 %
    const auto r = monte_carlo_option_var_vol(leg, {}, cov, 0.01, 200'000, Dist::Normal, 6.0, 5,
                                              200, Repair::None, 0.0, VolMode::HistoricalAtm);
    EXPECT_NEAR(r.var, 4.35873555333572, 4.0 * r.var_se);
    EXPECT_NEAR(r.es, 4.948184300200496, 0.02 * 4.948184300200496);
}

TEST(OptionVarVol, MonteCarloIsOptionBookPnlVolOnTheSimulatedPaths) {
    Matrix cov(4, 4);
    const double v[4] = {1e-4, 4e-4, 9e-4, 1.6e-3};
    for (std::size_t i = 0; i < 4; ++i) cov.data()[i * 5] = v[i];
    cov.data()[0 * 4 + 2] = cov.data()[2 * 4 + 0] = -0.5 * std::sqrt(v[0] * v[2]);
    const auto r = monte_carlo_option_var_vol(kLegs, kW, cov, 0.01, 20'000, Dist::Normal, 6.0, 9,
                                              50, Repair::None, 0.0, VolMode::StickyDelta, kSkews);
    const Matrix scen = simulate_factor_returns(cov, 20'000, Dist::Normal, 6.0, 9, Repair::None);
    const auto pnl = option_book_pnl_vol(kLegs, kW, scen, 0.0, VolMode::StickyDelta, kSkews);
    const auto t = mc_tail_metrics(pnl, 0.01, 50, 9 ^ 0x9E3779B97F4A7C15ULL);
    EXPECT_EQ(r.var, t.var);
    EXPECT_EQ(r.es, t.es);
}

TEST(OptionVarVol, RejectsBadInput) {
    const Matrix c = combined_panel();
    EXPECT_THROW((void)historical_option_var_es_vol(kLegs, kW, Matrix(kN, 3), 0.01, 0.0,
                                                    VolMode::StickyStrike),
                 std::invalid_argument);
    EXPECT_THROW((void)historical_option_var_es_vol({}, kW, c, 0.01, 0.0, VolMode::StickyStrike),
                 std::invalid_argument);
    EXPECT_THROW((void)historical_option_var_es_vol(kLegs, std::vector<double>{1, 2, 3, 4}, c, 0.01,
                                                    0.0, VolMode::StickyStrike),
                 std::invalid_argument);
    EXPECT_THROW((void)historical_option_var_es_vol(kLegs, kW, c, 0.01, 0.0, VolMode::StickyDelta,
                                                    std::vector<double>{1.0}),
                 std::invalid_argument);
    Matrix cov3(3, 3);
    EXPECT_THROW((void)monte_carlo_option_var_vol(kLegs, kW, cov3, 0.01, 1000, Dist::Normal, 6.0, 1,
                                                  50, Repair::None, 0.0, VolMode::StickyStrike),
                 std::invalid_argument);
}
