// Stress testing: canned scenarios, joint FX+rates shocks, peg-break
// add-on, reverse stress closed form vs independent numerical search.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "fxvar/stress.hpp"

using namespace fxvar;

namespace {

Market test_market() {
  return Market({{"EUR", 1.10},
                 {"GBP", 1.27},
                 {"JPY", 0.0090},
                 {"CHF", 1.12},
                 {"HKD", 0.1282}},
                {{"USD", 0.050},
                 {"EUR", 0.030},
                 {"GBP", 0.045},
                 {"JPY", 0.001},
                 {"CHF", -0.005}});
}

}  // namespace

TEST(Stress, BrexitScenarioHitsLongCable) {
  // Long 10m GBPUSD loses ~ 10m * 1.27 * (-8.1%) under the GBP -8% shock.
  const Market m = test_market();
  Book book({SpotPosition{"GBPUSD", 10e6, {}}});
  const auto lib = historical_scenarios();
  ASSERT_TRUE(lib.count("brexit_2016"));
  const auto rows = run_stress(book, m, lib);
  const auto it = std::find_if(rows.begin(), rows.end(), [](const StressRow& r) {
    return r.key == "brexit_2016";
  });
  ASSERT_NE(it, rows.end());
  const double expect = 10e6 * 1.27 * (std::exp(std::log1p(-0.081)) - 1.0);
  EXPECT_NEAR(it->pnl, expect, 1e-6 * std::abs(expect));
  // Report is sorted worst-first.
  for (std::size_t i = 1; i < rows.size(); ++i)
    EXPECT_LE(rows[i - 1].pnl, rows[i].pnl);
}

TEST(Stress, ChfDepegHitsShortChf) {
  // Short CHF (long USDCHF) loses when CHF revalues +14.9%.
  const Market m = test_market();
  Book book({SpotPosition{"USDCHF", 20e6, {}}});
  const auto lib = historical_scenarios();
  const auto rows = run_stress(book, m, lib);
  const auto it = std::find_if(rows.begin(), rows.end(), [](const StressRow& r) {
    return r.key == "chf_depeg_2015";
  });
  ASSERT_NE(it, rows.end());
  EXPECT_LT(it->pnl, -2e6);  // ~ -20m * 0.149 USD of the CHF leg
}

TEST(Stress, JointFxRateShockMovesForwardThroughBothLegs) {
  // A GBP forward feels both the spot shock and the IR:GBP shift in the
  // brexit scenario; a spot-only book feels only the FX leg.
  const Market m = test_market();
  Book fwd({ForwardPosition{"GBPUSD", 10e6, 1.0, {}}});
  const auto lib = historical_scenarios();
  const Scenario& brexit = lib.at("brexit_2016");
  const CompiledBook cb(fwd, m);
  const double joint = cb.pnl(brexit.shocks);
  auto fx_only = brexit.shocks;
  fx_only.erase(ir_factor("GBP"));
  const double fx_leg = cb.pnl(fx_only);
  // The -25bp GBP rate shift raises the discounted GBP leg: P&L differs.
  EXPECT_GT(std::abs(joint - fx_leg), 1e3);
  EXPECT_LT(joint, 0.0);
}

TEST(Stress, UsdBroadMoveAndPegBreak) {
  const Market m = test_market();
  // USD +10%: every CCYUSD falls 10/110 in simple terms.
  const Scenario usd10 = usd_broad_move({"EUR", "JPY", "USD"}, 0.10);
  EXPECT_EQ(usd10.shocks.size(), 2u);  // USD skipped
  EXPECT_NEAR(usd10.shocks.at("FX:EUR"), std::log1p(-0.10 / 1.10), 1e-15);
  Book book({SpotPosition{"EURUSD", 11e6, {}}});
  const CompiledBook cb(book, m);
  const double pnl = cb.pnl(usd10.shocks);
  EXPECT_NEAR(pnl, 11e6 * 1.10 * (-0.10 / 1.10), 1e-6 * std::abs(pnl));

  // Peg break: -30% devaluation with contagion.
  const Scenario pb = peg_break_scenario("HKD", -0.30, {{"CHF", -0.05}});
  EXPECT_NEAR(pb.shocks.at("FX:HKD"), std::log1p(-0.30), 1e-15);
  EXPECT_NEAR(pb.shocks.at("FX:CHF"), std::log1p(-0.05), 1e-15);
  EXPECT_THROW(peg_break_scenario("HKD", -1.0), std::invalid_argument);
  EXPECT_THROW(usd_broad_move({"EUR"}, -1.0), std::invalid_argument);
}

TEST(ReverseStress, ClosedFormLossAndDirection) {
  const std::vector<double> w{11e6, -4.5e6, -2.4e6};
  const Matrix cov = Matrix::from_rows({{3.6e-5, 1.1e-5, -2.0e-6},
                                        {1.1e-5, 4.9e-5, -1.0e-6},
                                        {-2.0e-6, -1.0e-6, 2.5e-7}});
  const double sp = std::sqrt(quad_form(w, cov));
  const double k = 3.0;
  const ReverseStress rs = reverse_stress_linear(w, cov, k);
  // Loss = k * sigma_p exactly.
  EXPECT_NEAR(rs.loss, k * sp, 1e-12 * rs.loss);
  // The shock reproduces the loss through the linear map -w'dx.
  double loss = 0.0;
  for (std::size_t i = 0; i < w.size(); ++i) loss -= w[i] * rs.shocks[i];
  EXPECT_NEAR(loss, rs.loss, 1e-10 * rs.loss);
  // Solving for a loss target inverts the radius.
  const ReverseStress rs2 = reverse_stress_for_loss(w, cov, 1e6);
  EXPECT_NEAR(rs2.loss, 1e6, 1e-9 * 1e6);
}

TEST(ReverseStress, NumericalSearchConfirmsClosedForm) {
  const std::vector<double> w{11e6, -4.5e6, -2.4e6};
  const Matrix cov = Matrix::from_rows({{3.6e-5, 1.1e-5, -2.0e-6},
                                        {1.1e-5, 4.9e-5, -1.0e-6},
                                        {-2.0e-6, -1.0e-6, 2.5e-7}});
  const double k = 2.5;
  const ReverseStress closed = reverse_stress_linear(w, cov, k);
  const ReverseStress numeric = reverse_stress_numerical(w, cov, k, 3);
  EXPECT_NEAR(numeric.loss, closed.loss, 1e-6 * closed.loss);
  for (std::size_t i = 0; i < w.size(); ++i)
    EXPECT_NEAR(numeric.shocks[i], closed.shocks[i],
                1e-4 * (std::abs(closed.shocks[i]) + 1e-6));
}

TEST(ReverseStress, ZeroRiskBookIsRejected) {
  const Matrix cov = Matrix::from_rows({{1e-4}});
  EXPECT_THROW(reverse_stress_linear({0.0}, cov, 1.0), std::invalid_argument);
  EXPECT_THROW(reverse_stress_linear({1e6}, cov, 0.0), std::invalid_argument);
  const Market m = test_market();
  EXPECT_THROW(run_stress(Book{}, m, historical_scenarios()),
               std::invalid_argument);
}


TEST(Stress, SimpleToLogRejectsImpossibleMoves) {
  // A -100% (or worse) move is an infinite log return; refuse it rather
  // than pushing -inf/NaN through every position in the report.
  EXPECT_THROW(simple_to_log(-1.0), std::invalid_argument);
  EXPECT_THROW(simple_to_log(-1.5), std::invalid_argument);
  EXPECT_THROW(simple_to_log(std::nan("")), std::invalid_argument);
  EXPECT_THROW(usd_broad_move({"EUR"}, -1.0), std::invalid_argument);
  EXPECT_THROW(peg_break_scenario("HKD", -1.0), std::invalid_argument);
  // Legitimate moves still round-trip exactly.
  EXPECT_DOUBLE_EQ(simple_to_log(0.0), 0.0);
  EXPECT_NEAR(std::expm1(simple_to_log(-0.30)), -0.30, 1e-15);
}
