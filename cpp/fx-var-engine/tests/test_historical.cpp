// Historical-simulation VaR: plain/BRW/FHS behaviour, sqrt-time scaling,
// peg-blindness warnings, validation edge cases.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "fxvar/historical.hpp"

using namespace fxvar;

namespace {

Market test_market() {
  return Market({{"EUR", 1.10}, {"JPY", 0.0090}, {"HKD", 0.1282}},
                {{"USD", 0.050}, {"EUR", 0.030}, {"JPY", 0.001}});
}

// Deterministic sinusoidal factor history (no RNG in tests that assert
// exact figures).
ReturnsMatrix synthetic_returns(int n, const std::vector<std::string>& factors,
                                const std::vector<double>& scales) {
  ReturnsMatrix r;
  r.factors = factors;
  r.data = Matrix(n, factors.size());
  for (int i = 0; i < n; ++i)
    for (std::size_t j = 0; j < factors.size(); ++j) {
      const double t = static_cast<double>(i);
      r.data(i, j) = scales[j] * (std::sin(0.1 * t + static_cast<double>(j)) +
                                  0.5 * std::cos(0.05 * t * (j + 1.0)));
    }
  return r;
}

}  // namespace

TEST(HistoricalVar, PlainMatchesDirectQuantileOfRevaluedPnl) {
  const Market m = test_market();
  Book book({SpotPosition{"EURUSD", 10e6, {}}});
  const ReturnsMatrix rets = synthetic_returns(300, {"FX:EUR"}, {0.006});
  HistoricalOptions opts;
  opts.alpha = 0.99;
  const HistoricalResult res = historical_var(book, m, rets, opts);
  ASSERT_EQ(res.pnl.size(), 300u);
  // With 300 scenarios at 99%, VaR is the 3rd worst loss - recompute by
  // hand from the result's own P&L vector.
  std::vector<double> losses;
  for (double p : res.pnl) losses.push_back(-p);
  std::sort(losses.rbegin(), losses.rend());
  EXPECT_NEAR(res.var, losses[2], 1e-9);
  // ES with integer tail count = mean of 3 worst losses.
  EXPECT_NEAR(res.es, (losses[0] + losses[1] + losses[2]) / 3.0, 1e-9);
  EXPECT_GE(res.es, res.var);
}

TEST(HistoricalVar, SqrtTimeScaling) {
  const Market m = test_market();
  Book book({SpotPosition{"EURUSD", 10e6, {}}});
  const ReturnsMatrix rets = synthetic_returns(300, {"FX:EUR"}, {0.006});
  HistoricalOptions d1;
  HistoricalOptions d10;
  d10.horizon_days = 10.0;
  const auto r1 = historical_var(book, m, rets, d1);
  const auto r10 = historical_var(book, m, rets, d10);
  EXPECT_NEAR(r10.var, r1.var * std::sqrt(10.0), 1e-9);
  EXPECT_NEAR(r10.es, r1.es * std::sqrt(10.0), 1e-9);
}

TEST(HistoricalVar, AgeWeightsSumToOneAndTiltRecent) {
  const Market m = test_market();
  Book book({SpotPosition{"EURUSD", 10e6, {}}});
  const ReturnsMatrix rets = synthetic_returns(200, {"FX:EUR"}, {0.006});
  HistoricalOptions opts;
  opts.method = HsMethod::kAge;
  opts.decay = 0.99;
  const auto res = historical_var(book, m, rets, opts);
  double sum = 0.0;
  for (double w : res.weights) sum += w;
  EXPECT_NEAR(sum, 1.0, 1e-12);
  // Most recent scenario (last row) carries the largest weight.
  EXPECT_GT(res.weights.back(), res.weights.front());
  EXPECT_NEAR(res.weights.back() / res.weights[res.weights.size() - 2],
              1.0 / 0.99, 1e-12);
}

TEST(HistoricalVar, FilteredScalesToCurrentVolRegime) {
  // History whose second half is 3x more volatile: FHS rescales old
  // scenarios up to today's EWMA vol, so FHS VaR > plain VaR (plain
  // dilutes the current regime with the quiet first half).
  const Market m = test_market();
  Book book({SpotPosition{"EURUSD", 10e6, {}}});
  ReturnsMatrix rets = synthetic_returns(400, {"FX:EUR"}, {0.004});
  for (int i = 200; i < 400; ++i) rets.data(i, 0) *= 3.0;
  HistoricalOptions plain;
  HistoricalOptions fhs;
  fhs.method = HsMethod::kFiltered;
  const auto rp = historical_var(book, m, rets, plain);
  const auto rf = historical_var(book, m, rets, fhs);
  EXPECT_GT(rf.var, rp.var);
}

TEST(HistoricalVar, PegBlindnessWarningTriggers) {
  // HKD inside its band: daily vol ~2e-4 < threshold 5e-4 -> flagged.
  const Market m = test_market();
  Book book({SpotPosition{"EURUSD", 10e6, {}}, SpotPosition{"USDHKD", 5e6, {}}});
  const ReturnsMatrix rets =
      synthetic_returns(300, {"FX:EUR", "FX:HKD"}, {0.006, 0.0002});
  const auto res = historical_var(book, m, rets, {});
  ASSERT_EQ(res.flagged_peg_factors.size(), 1u);
  EXPECT_EQ(res.flagged_peg_factors[0], "FX:HKD");
  ASSERT_EQ(res.warnings.size(), 1u);
  EXPECT_NE(res.warnings[0].find("peg"), std::string::npos);
  // Free-float-only book: no flags.
  Book clean({SpotPosition{"EURUSD", 10e6, {}}});
  const auto res2 = historical_var(
      clean, m, synthetic_returns(300, {"FX:EUR"}, {0.006}), {});
  EXPECT_TRUE(res2.flagged_peg_factors.empty());
}

TEST(HistoricalVar, ValidationAndEdgeCases) {
  const Market m = test_market();
  Book book({SpotPosition{"EURUSD", 10e6, {}}});
  // Empty book throws.
  EXPECT_THROW(
      historical_var(Book{}, m, synthetic_returns(100, {"FX:EUR"}, {0.01}), {}),
      std::invalid_argument);
  // Too little history throws.
  EXPECT_THROW(
      historical_var(book, m, synthetic_returns(30, {"FX:EUR"}, {0.01}), {}),
      std::invalid_argument);
  // Missing factor column throws.
  EXPECT_THROW(
      historical_var(book, m, synthetic_returns(300, {"FX:JPY"}, {0.01}), {}),
      std::invalid_argument);
  // NaNs are refused.
  ReturnsMatrix bad = synthetic_returns(300, {"FX:EUR"}, {0.01});
  bad.data(7, 0) = std::nan("");
  EXPECT_THROW(historical_var(book, m, bad, {}), std::invalid_argument);
  // Bad alpha / horizon / decay.
  HistoricalOptions a;
  a.alpha = 1.0;
  EXPECT_THROW(
      historical_var(book, m, synthetic_returns(300, {"FX:EUR"}, {0.01}), a),
      std::invalid_argument);
  HistoricalOptions h;
  h.horizon_days = 0.0;
  EXPECT_THROW(
      historical_var(book, m, synthetic_returns(300, {"FX:EUR"}, {0.01}), h),
      std::invalid_argument);
  HistoricalOptions d;
  d.method = HsMethod::kAge;
  d.decay = 1.5;
  EXPECT_THROW(
      historical_var(book, m, synthetic_returns(300, {"FX:EUR"}, {0.01}), d),
      std::invalid_argument);
}

TEST(HistoricalVar, PureBaseCcyCashIsZeroRisk) {
  const Market m = test_market();
  Book book({CashPosition{"USD", 50e6}});  // USD cash in a USD book
  const auto res =
      historical_var(book, m, synthetic_returns(300, {"FX:EUR"}, {0.01}), {});
  EXPECT_DOUBLE_EQ(res.var, 0.0);
  EXPECT_DOUBLE_EQ(res.es, 0.0);
}
