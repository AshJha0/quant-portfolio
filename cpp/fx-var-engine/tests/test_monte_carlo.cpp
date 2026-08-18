// Monte Carlo VaR: convergence to closed form (3 SE), tail ordering of
// t / jump-mixture vs normal, bitwise seed determinism, peg-break add-on.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "fxvar/historical.hpp"
#include "fxvar/monte_carlo.hpp"
#include "fxvar/parametric.hpp"

using namespace fxvar;

namespace {

Market test_market() {
  return Market({{"EUR", 1.10}, {"JPY", 0.0090}, {"HKD", 0.1282}},
                {{"USD", 0.050}, {"EUR", 0.030}, {"JPY", 0.001}});
}

FactorCov two_factor_cov() {
  FactorCov cov;
  cov.factors = {"FX:EUR", "FX:JPY"};
  cov.cov = Matrix::from_rows({{3.6e-5, 1.1e-5}, {1.1e-5, 4.9e-5}});
  return cov;
}

Book two_factor_book() {
  return Book({SpotPosition{"EURUSD", 10e6, {}},
               SpotPosition{"USDJPY", -4e6, {}}});
}

}  // namespace

TEST(MonteCarlo, SimulatedCovarianceMatchesTarget) {
  const FactorCov cov = two_factor_cov();
  const ReturnsMatrix scen =
      simulate_factor_returns(cov, 200000, McDist::kNormal, 6.0, {}, 42);
  const FactorCov est = sample_cov(scen);
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      EXPECT_NEAR(est.cov(i, j), cov.cov(i, j),
                  4.0 * cov.cov(i, i) / std::sqrt(200000.0) +
                      0.01 * std::abs(cov.cov(i, j)));
  // Student-t is covariance-matched by construction: same tolerance.
  const ReturnsMatrix scen_t =
      simulate_factor_returns(cov, 200000, McDist::kStudentT, 6.0, {}, 42);
  const FactorCov est_t = sample_cov(scen_t);
  EXPECT_NEAR(est_t.cov(0, 0), cov.cov(0, 0), 0.05 * cov.cov(0, 0));
  EXPECT_NEAR(est_t.cov(0, 1), cov.cov(0, 1), 0.10 * cov.cov(0, 1));
}

TEST(MonteCarlo, NormalMcMatchesParametricWithin3SE) {
  const Market m = test_market();
  const Book book = two_factor_book();
  const FactorCov cov = two_factor_cov();
  MonteCarloOptions opts;
  opts.n_scenarios = 100000;
  opts.seed = 7;
  const MonteCarloResult mc = monte_carlo_var(book, m, cov, opts);
  // Closed form on the linearised book (linear here up to tiny exp
  // convexity).
  const CompiledBook cb(book, m);
  const FactorCov sub = cov.select(cb.factors());
  const VarEs closed = var_covar(cb.linear_exposures(), sub.cov, 0.99, 1.0);
  EXPECT_GT(mc.se_var, 0.0);
  EXPECT_NEAR(mc.var, closed.var, 3.0 * mc.se_var);
  EXPECT_GE(mc.es, mc.var);
}

TEST(MonteCarlo, FatTailOrderingAt99) {
  // Same covariance, same seed: t(4) and jump-mixture must both produce
  // larger 99% VaR than normal - tail shape only.
  const Market m = test_market();
  const Book book = two_factor_book();
  const FactorCov cov = two_factor_cov();
  MonteCarloOptions normal;
  normal.n_scenarios = 60000;
  normal.seed = 11;
  MonteCarloOptions t4 = normal;
  t4.dist = McDist::kStudentT;
  t4.df = 4.0;
  MonteCarloOptions jump = normal;
  jump.dist = McDist::kJump;
  jump.jumps.prob = 0.02;
  jump.jumps.mean = {{"FX:EUR", -0.05}};
  jump.jumps.stdev = {{"FX:EUR", 0.01}};
  const auto rn = monte_carlo_var(book, m, cov, normal);
  const auto rt = monte_carlo_var(book, m, cov, t4);
  const auto rj = monte_carlo_var(book, m, cov, jump);
  EXPECT_GT(rt.var, rn.var);
  EXPECT_GT(rt.es, rn.es);
  EXPECT_GT(rj.var, rn.var);
  EXPECT_GT(rj.es, rn.es);
}

TEST(MonteCarlo, BitwiseSeedDeterminism) {
  const Market m = test_market();
  const Book book = two_factor_book();
  const FactorCov cov = two_factor_cov();
  MonteCarloOptions opts;
  opts.n_scenarios = 20000;
  opts.seed = 123;
  opts.dist = McDist::kStudentT;
  opts.df = 5.0;
  const auto a = monte_carlo_var(book, m, cov, opts);
  const auto b = monte_carlo_var(book, m, cov, opts);
  EXPECT_EQ(a.var, b.var);  // bitwise, not approximate
  EXPECT_EQ(a.es, b.es);
  ASSERT_EQ(a.pnl.size(), b.pnl.size());
  for (std::size_t i = 0; i < a.pnl.size(); i += 997)
    EXPECT_EQ(a.pnl[i], b.pnl[i]);
  // A different seed must change the draw.
  MonteCarloOptions other = opts;
  other.seed = 124;
  const auto c = monte_carlo_var(book, m, cov, other);
  EXPECT_NE(a.var, c.var);
}

TEST(MonteCarlo, PegBreakJumpProducesLossHistoricalSimulationMisses) {
  // A pegged HKD short-USD book: the historical window shows ~zero vol,
  // so HS VaR is negligible - but the jump-mixture MC with a peg-break
  // overlay reports a material loss.  This is the engine's peg-blindness
  // story end to end.
  const Market m = test_market();
  Book book({SpotPosition{"USDHKD", 100e6, {}}});  // long USD vs HKD

  ReturnsMatrix rets;
  rets.factors = {"FX:HKD"};
  rets.data = Matrix(250, 1);
  for (int i = 0; i < 250; ++i)
    rets.data(i, 0) = 1e-4 * std::sin(0.5 * i);  // band-bound noise
  const auto hs = historical_var(book, m, rets, {});
  ASSERT_EQ(hs.flagged_peg_factors.size(), 1u);  // engine warns

  FactorCov cov = sample_cov(rets);
  MonteCarloOptions opts;
  opts.n_scenarios = 50000;
  opts.seed = 5;
  opts.dist = McDist::kJump;
  opts.jumps.prob = 0.02;                  // revaluation event
  opts.jumps.mean = {{"FX:HKD", 0.10}};    // HKD +10% (log) vs USD
  const auto mc = monte_carlo_var(book, m, cov, opts);
  // Short 100m USD of HKD: a +10% HKD reval loses ~10m USD.
  EXPECT_LT(hs.var, 0.1e6);   // HS blind: < 0.1% of notional
  EXPECT_GT(mc.var, 5e6);     // jump MC sees the break
  EXPECT_GT(mc.var, 20.0 * hs.var);
}

TEST(MonteCarlo, SingularPeggedCovarianceRunsWithJitter) {
  // Two perfectly correlated pegs: covariance is singular; the engine
  // must factorise with jitter and surface the diagnostic.
  const Market m = Market({{"HKD", 0.1282}, {"AED", 0.2723}}, {});
  Book book({SpotPosition{"USDHKD", 10e6, {}}, SpotPosition{"USDAED", 5e6, {}}});
  // Exactly representable rank-1 matrix: the second Cholesky pivot is
  // exactly zero, so the jitter path is guaranteed to engage (a general
  // rank-1 double matrix can round to a tiny positive pivot instead).
  FactorCov cov;
  cov.factors = {"FX:AED", "FX:HKD"};
  cov.cov = Matrix::from_rows({{6.25e-2, 3.125e-2}, {3.125e-2, 1.5625e-2}});
  MonteCarloOptions opts;
  opts.n_scenarios = 1000;
  const auto res = monte_carlo_var(book, m, cov, opts);
  EXPECT_FALSE(res.cholesky_warning.empty());
  EXPECT_GE(res.var, 0.0);
}

TEST(MonteCarlo, InputValidation) {
  const FactorCov cov = two_factor_cov();
  EXPECT_THROW(simulate_factor_returns(cov, 0), std::invalid_argument);
  EXPECT_THROW(simulate_factor_returns(cov, 10, McDist::kStudentT, 2.0),
               std::invalid_argument);
  JumpSpec bad_prob;
  bad_prob.prob = 1.5;
  EXPECT_THROW(simulate_factor_returns(cov, 10, McDist::kJump, 6.0, bad_prob),
               std::invalid_argument);
  JumpSpec bad_std;
  bad_std.prob = 0.1;
  bad_std.mean = {{"FX:EUR", -0.1}};
  bad_std.stdev = {{"FX:EUR", -0.2}};
  EXPECT_THROW(simulate_factor_returns(cov, 10, McDist::kJump, 6.0, bad_std),
               std::invalid_argument);
  EXPECT_THROW(var_standard_error({1.0, 2.0}, 0.99), std::invalid_argument);
}
