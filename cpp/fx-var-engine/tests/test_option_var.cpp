// Option VaR: Garman-Kohlhagen full revaluation wired into historical_var and
// monte_carlo_var. Golden values from
// cpp/equity-var-engine/tools/gen_option_var_golden.py (Python reference
// fx_var.revaluation.full_revaluation_pnl + the S2 order statistic); the MC
// check is against the closed-form quantile of a monotone one-factor book.
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "fxvar/historical.hpp"
#include "fxvar/monte_carlo.hpp"
#include "fxvar/option_reval.hpp"

using namespace fxvar;

namespace {
constexpr std::size_t kN = 250;
const std::vector<OptionLeg> kLegs{
    {1.0e6, OptionKind::Call, 1.10, 1.12, 0.25, 0.045, 0.03, 0.08, 0},
    {-2.0e6, OptionKind::Put, 0.75, 0.74, 0.5, 0.045, 0.04, 0.10, 1},
};
const std::vector<std::string> kNames{"FX:EUR", "FX:AUD"};

Market market() { return Market({{"EUR", 1.10}, {"AUD", 0.75}}, {{"USD", 0.045}}); }

ReturnsMatrix panel() {
  ReturnsMatrix r;
  r.factors = {"FX:AUD", "FX:EUR"};  // deliberately not in leg order
  r.data = Matrix(kN, 2);
  for (std::size_t i = 0; i < kN; ++i) {
    const double t = static_cast<double>(i);
    r.data(i, 1) = 0.02 * std::sin(1.3 * t + 0.4);
    r.data(i, 0) = 0.015 * std::cos(0.7 * t) + 0.005 * std::sin(2.9 * t);
  }
  return r;
}

void expect_rel(double got, double want, double rel = 1e-11) {
  EXPECT_NEAR(got, want, rel * std::abs(want)) << "want " << want;
}

HistoricalOptions plain() {
  HistoricalOptions o;
  o.alpha = 0.01;
  o.warn_pegs = false;
  return o;
}
}  // namespace

TEST(OptionVar, HistoricalOptionsOnlyMatchesPython) {
  const auto r = historical_var(Book{}, market(), kLegs, kNames, panel(), plain());
  ASSERT_EQ(r.pnl.size(), kN);
  expect_rel(r.pnl[0], 11397.749935234513);
  expect_rel(r.pnl[7], 3718.275663457116, 1e-10);
  expect_rel(r.var, 16760.147839431305);
  expect_rel(r.es, 17352.525540976494);
}

TEST(OptionVar, HistoricalWithLinearBookMatchesPython) {
  const Book book({CashPosition{"EUR", 1.0e6}});
  const auto r = historical_var(book, market(), kLegs, kNames, panel(), plain(), 1.0 / 252.0);
  expect_rel(r.pnl[0], 19969.669477855823);
  expect_rel(r.var, 36908.47316233292);
  expect_rel(r.es, 37902.526440186884);
}

TEST(OptionVar, NoOptionsIsTheLinearPath) {
  const Book book({SpotPosition{"EURUSD", 5e6, {}}});
  const auto a = historical_var(book, market(), {}, {}, panel(), plain());
  const auto b = historical_var(book, market(), panel(), plain());
  EXPECT_EQ(a.var, b.var);
  EXPECT_EQ(a.es, b.es);
  FactorCov cov{{"FX:AUD", "FX:EUR"}, Matrix(2, 2)};
  cov.cov(0, 0) = 4e-5;
  cov.cov(1, 1) = 3.6e-5;
  cov.cov(0, 1) = cov.cov(1, 0) = 1e-5;
  MonteCarloOptions mo;
  mo.n_scenarios = 5000;
  mo.n_boot = 20;
  mo.seed = 9;
  const auto c = monte_carlo_var(book, market(), {}, {}, cov, mo);
  const auto d = monte_carlo_var(book, market(), cov, mo);
  EXPECT_EQ(c.var, d.var);
  EXPECT_EQ(c.es, d.es);
  EXPECT_EQ(c.se_var, d.se_var);
}

TEST(OptionVar, MonteCarloMatchesClosedFormQuantile) {
  const std::vector<OptionLeg> leg{{1.0e6, OptionKind::Call, 1.10, 1.10, 0.25, 0.045, 0.03, 0.08, 0}};
  FactorCov cov{{"FX:EUR"}, Matrix(1, 1)};
  cov.cov(0, 0) = 0.006 * 0.006;
  MonteCarloOptions mo;
  mo.n_scenarios = 200'000;
  mo.seed = 11;
  const auto r = monte_carlo_var(Book{}, market(), leg, {"FX:EUR"}, cov, mo);
  EXPECT_EQ(r.n_scenarios, 200'000u);
  EXPECT_GT(r.se_var, 0.0);
  EXPECT_NEAR(r.var, 7204.544744658981, 4.0 * r.se_var);
  EXPECT_NEAR(r.es, 8054.686967487454, 0.02 * 8054.686967487454);
}

TEST(OptionVar, FilteredAndAgeMethodsRun) {
  const Book book({CashPosition{"EUR", 1.0e6}});
  for (HsMethod m : {HsMethod::kAge, HsMethod::kFiltered}) {
    HistoricalOptions o = plain();
    o.method = m;
    const auto r = historical_var(book, market(), kLegs, kNames, panel(), o);
    EXPECT_GT(r.var, 0.0);
    EXPECT_GE(r.es, r.var);
    EXPECT_EQ(r.weights.size(), kN);
  }
}

TEST(OptionVar, RejectsBadInput) {
  const Book usd({CashPosition{"EUR", 1.0e6}});
  HistoricalOptions h10 = plain();
  h10.horizon_days = 10.0;
  EXPECT_THROW((void)historical_var(usd, market(), kLegs, kNames, panel(), h10),
               std::invalid_argument);
  const Book eur_base({CashPosition{"AUD", 1.0e6}}, "EUR");
  EXPECT_THROW((void)historical_var(eur_base, market(), kLegs, kNames, panel(), plain()),
               std::invalid_argument);
  EXPECT_THROW((void)historical_var(usd, market(), kLegs, {"FX:EUR"}, panel(), plain()),
               std::invalid_argument);  // leg 1 factor out of range
  EXPECT_THROW((void)historical_var(usd, market(), kLegs, {"EUR", "AUD"}, panel(), plain()),
               std::invalid_argument);
  EXPECT_THROW((void)historical_var(usd, market(), kLegs, {"FX:EUR", "FX:CHF"}, panel(), plain()),
               std::invalid_argument);  // CHF missing from the history
}
