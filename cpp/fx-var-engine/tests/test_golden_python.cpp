// Cross-language golden tests against the Python reference engine.
//
// Provenance
// ----------
// Every constant comes from `golden_vectors.hpp`, a GENERATED header
// emitted by `tools/gen_golden_header.py` from the JSON written by the
// Python reference package python/fx/03-var-es-engine (fx_var):
//
//   cd python/fx/03-var-es-engine
//   PYTHONPATH=src python3 tests/golden/generate_golden.py   # -> golden_vectors.json
//   cd ../../../cpp/fx-var-engine
//   python3 tools/gen_golden_header.py                       # -> tests/golden_vectors.hpp
//   python3 tools/gen_golden_header.py --check               # provenance check
//
// The JSON is itself locked by python/fx/03-var-es-engine/tests/
// test_golden_vectors.py.  Values are numpy/scipy doubles printed via
// repr() (shortest round-trip).  The three cases are fully deterministic
// (no RNG):
//
//   CASE A - book revaluation + plain/BRW historical VaR/ES on a
//            sinusoidal synthetic history reproduced bit-for-bit here;
//   CASE B - parametric closed-form normal / Student-t VaR & ES from
//            fixed exposures and covariance (fx_var.parametric_var.var_covar);
//   CASE C - Kupiec / Christoffersen / Basel statistics
//            (fx_var.backtesting).
//
// Tolerances (declared in the JSON and re-exported as golden::tol_*): the
// only cross-language differences are libm sin/cos/exp rounding (~1 ulp
// per call) and the special-function implementations (scipy Cephes vs
// this library), so P&L-scale figures agree to ~1e-8 relative and
// probability-scale figures to ~1e-9 absolute.

#include <gtest/gtest.h>

#include <cmath>

#include "fxvar/backtest.hpp"
#include "fxvar/historical.hpp"
#include "fxvar/parametric.hpp"
#include "golden_vectors.hpp"

using namespace fxvar;

namespace {

// CASE A fixture: mirrors the Python golden generator exactly.
Market golden_market() {
  return Market({{"EUR", 1.10}, {"JPY", 0.0090}, {"GBP", 1.27}},
                {{"USD", 0.050}, {"EUR", 0.030}, {"JPY", 0.001}});
}

Book golden_book() {
  return Book({
      SpotPosition{"EURUSD", 10'000'000.0, {}},   // long 10m EUR at market
      ForwardPosition{"USDJPY", 5'000'000.0, 0.5, {}},  // ATM CIP forward
      SpotPosition{"EURJPY", -3'000'000.0, {}},   // short 3m EUR cross
  });
}

// Deterministic history: factors sorted as the book enumerates them
// (FX:EUR, FX:JPY, IR:JPY, IR:USD), r[t][j] = s_j (sin(0.1 t + j)
// + 0.5 cos(0.05 t (j+1))) - identical formula in the Python generator.
ReturnsMatrix golden_returns() {
  ReturnsMatrix r;
  r.factors = {"FX:EUR", "FX:JPY", "IR:JPY", "IR:USD"};
  const double scales[4] = {0.006, 0.007, 0.0004, 0.0005};
  const int n = 300;
  r.data = Matrix(n, 4);
  for (int t = 0; t < n; ++t)
    for (int j = 0; j < 4; ++j) {
      const double td = static_cast<double>(t);
      const double jd = static_cast<double>(j);
      r.data(t, j) = scales[j] * (std::sin(0.1 * td + jd) +
                                  0.5 * std::cos(0.05 * td * (jd + 1.0)));
    }
  return r;
}

}  // namespace

TEST(GoldenPython, CaseA_BookPnlSingleScenario) {
  // Python: book.pnl(market, returns.iloc[17]) = 58177.37489810074
  const Market m = golden_market();
  const CompiledBook cb(golden_book(), m);
  const ReturnsMatrix rets = golden_returns();
  const double got = cb.pnl(rets.data.row(17));
  EXPECT_NEAR(got, golden::a_book_pnl_scenario_17, golden::tol_case_a_abs);
}

TEST(GoldenPython, CaseA_PlainHistoricalVarEs) {
  // Python fx_var.historical_var(..., alpha=.99/.975, method="plain"):
  //   var99  = 61919.80890587624   es99  = 62006.12006224847
  //   var975 = 61237.42600889597   es975 = 61777.93608271857
  const Market m = golden_market();
  const ReturnsMatrix rets = golden_returns();
  HistoricalOptions o99;
  o99.alpha = 0.99;
  const auto r99 = historical_var(golden_book(), m, rets, o99);
  EXPECT_NEAR(r99.var, golden::a_plain_var_99, golden::tol_case_a_abs);
  EXPECT_NEAR(r99.es, golden::a_plain_es_99, golden::tol_case_a_abs);
  HistoricalOptions o975;
  o975.alpha = 0.975;
  const auto r975 = historical_var(golden_book(), m, rets, o975);
  EXPECT_NEAR(r975.var, golden::a_plain_var_975, golden::tol_case_a_abs);
  EXPECT_NEAR(r975.es, golden::a_plain_es_975, golden::tol_case_a_abs);
}

TEST(GoldenPython, CaseA_AgeWeightedHistoricalVarEs) {
  // Python fx_var.historical_var(..., method="age", decay=0.995):
  //   var99 = 61874.26268531149   es99 = 61977.52496594109
  const Market m = golden_market();
  HistoricalOptions o;
  o.alpha = 0.99;
  o.method = HsMethod::kAge;
  o.decay = 0.995;
  const auto r = historical_var(golden_book(), m, golden_returns(), o);
  EXPECT_NEAR(r.var, golden::a_age_var_99_decay_0995, golden::tol_case_a_abs);
  EXPECT_NEAR(r.es, golden::a_age_es_99_decay_0995, golden::tol_case_a_abs);
}

TEST(GoldenPython, CaseB_ParametricClosedForm) {
  // Python fx_var.parametric_var.var_covar on fixed exposures/cov:
  //   w   = {FX:EUR: 11e6, FX:JPY: -4.5e6, IR:USD: -2.4e6}
  //   cov = [[3.6e-5, 1.1e-5, -2e-6],
  //          [1.1e-5, 4.9e-5, -1e-6],
  //          [-2e-6, -1e-6, 2.5e-7]]
  //   normal 99%:  var = 153339.50441962917  es = 175675.6297200285
  //   t(5)   99%:  var = 171803.12389091405  es = 227327.5314974144
  //   normal 99% 10d: var = 484902.0892474838 es = 555535.1192996583
  const std::vector<double> w{11e6, -4.5e6, -2.4e6};
  const Matrix cov = Matrix::from_rows({{3.6e-5, 1.1e-5, -2.0e-6},
                                        {1.1e-5, 4.9e-5, -1.0e-6},
                                        {-2.0e-6, -1.0e-6, 2.5e-7}});
  const VarEs n1 = var_covar(w, cov, 0.99, 1.0, TailDist::kNormal);
  EXPECT_NEAR(n1.var, golden::b_normal_var_99_1d, golden::tol_case_b_rel * n1.var);
  EXPECT_NEAR(n1.es, golden::b_normal_es_99_1d, golden::tol_case_b_rel * n1.es);
  const VarEs t5 = var_covar(w, cov, 0.99, 1.0, TailDist::kStudentT, 5.0);
  EXPECT_NEAR(t5.var, golden::b_t5_var_99_1d, golden::tol_case_b_rel * t5.var);
  EXPECT_NEAR(t5.es, golden::b_t5_es_99_1d, golden::tol_case_b_rel * t5.es);
  const VarEs n10 = var_covar(w, cov, 0.99, 10.0, TailDist::kNormal);
  EXPECT_NEAR(n10.var, golden::b_normal_var_99_10d, golden::tol_case_b_rel * n10.var);
  EXPECT_NEAR(n10.es, golden::b_normal_es_99_10d, golden::tol_case_b_rel * n10.es);
}

TEST(GoldenPython, CaseC_BacktestStatistics) {
  // Python fx_var.backtesting:
  //   kupiec_pof(8, 250, 0.99)   -> LR = 7.7335507244945205
  //                                 p  = 0.0054204051941277994
  //   christoffersen on the fixed pattern (t % 37 == 5, plus 100,101;
  //   9 exceptions in 250)       -> LR = 1.0063610339314124
  //                                 p  = 0.3157762037622499
  //   basel cum prob: P(X<=5)  = 0.9588168159301514  (yellow, 3.40)
  //                   P(X<=4)  = 0.8921876269036249  (green,  3.00)
  //                   P(X<=10) = 0.999946101370953   (red,    4.00)
  const LrTest k = kupiec_pof(8, 250, 0.99);
  EXPECT_NEAR(k.lr, golden::c_kupiec_lr_8_250, golden::tol_kupiec_lr_abs);
  EXPECT_NEAR(k.p, golden::c_kupiec_p_8_250, golden::tol_kupiec_p_abs);

  std::vector<int> e(250, 0);
  for (int t = 0; t < 250; ++t)
    if (t % 37 == 5) e[t] = 1;
  e[100] = 1;
  e[101] = 1;
  int count = 0;
  for (int v : e) count += v;
  ASSERT_EQ(count, 9);  // matches the Python generator's pattern count
  const LrTest c = christoffersen_independence(e);
  EXPECT_NEAR(c.lr, golden::c_christoffersen_ind_lr, golden::tol_christoffersen_abs);
  EXPECT_NEAR(c.p, golden::c_christoffersen_ind_p, golden::tol_christoffersen_abs);

  const TrafficLight t5 = basel_traffic_light(5, 250, 0.99);
  EXPECT_NEAR(t5.cumulative_prob, golden::c_basel_cum_prob_5, golden::tol_basel_cum_prob_abs);
  EXPECT_EQ(t5.zone, Zone::kYellow);
  EXPECT_DOUBLE_EQ(t5.multiplier, golden::c_basel_multiplier_5);
  const TrafficLight t4 = basel_traffic_light(4, 250, 0.99);
  EXPECT_NEAR(t4.cumulative_prob, golden::c_basel_cum_prob_4, golden::tol_basel_cum_prob_abs);
  EXPECT_EQ(t4.zone, Zone::kGreen);
  EXPECT_DOUBLE_EQ(t4.multiplier, golden::c_basel_multiplier_4);
  const TrafficLight t10 = basel_traffic_light(10, 250, 0.99);
  EXPECT_NEAR(t10.cumulative_prob, golden::c_basel_cum_prob_10, golden::tol_basel_cum_prob_abs);
  EXPECT_EQ(t10.zone, Zone::kRed);
  EXPECT_DOUBLE_EQ(t10.multiplier, golden::c_basel_multiplier_10);
}
