// Fuzz-style input walk: every public entry point of the library is called
// with NaN, +inf, -inf, empty, length-1 and negative inputs where those are
// invalid, and must throw std::invalid_argument. Never a number.
//
// When you add a public function, add it here. The list mirrors the headers
// in include/fxvar/.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "fxvar/backtest.hpp"
#include "fxvar/book.hpp"
#include "fxvar/expected_shortfall.hpp"
#include "fxvar/historical.hpp"
#include "fxvar/matrix.hpp"
#include "fxvar/monte_carlo.hpp"
#include "fxvar/parametric.hpp"
#include "fxvar/returns.hpp"
#include "fxvar/stats.hpp"
#include "fxvar/stress.hpp"

using namespace fxvar;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
const double kBad[] = {kNaN, kInf, -kInf};

std::vector<double> good_pnl(std::size_t n = 200) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = 100.0 * std::sin(0.37 * static_cast<double>(i) + 0.1);
  return v;
}

std::vector<double> with_bad(std::vector<double> v, double bad) {
  v[v.size() / 2] = bad;
  return v;
}

Market market() {
  return Market({{"EUR", 1.10}, {"JPY", 0.0090}}, {{"USD", 0.05}, {"EUR", 0.03}, {"JPY", 0.001}});
}

Book book() { return Book({SpotPosition{"EURUSD", 10e6, {}}, SpotPosition{"USDJPY", -4e6, {}}}); }

ReturnsMatrix returns(std::size_t n = 120) {
  ReturnsMatrix r;
  r.factors = {"FX:EUR", "FX:JPY"};
  r.data = Matrix(n, 2);
  for (std::size_t i = 0; i < n; ++i) {
    r.data(i, 0) = 0.006 * std::sin(0.3 * static_cast<double>(i));
    r.data(i, 1) = 0.007 * std::cos(0.41 * static_cast<double>(i));
  }
  return r;
}

FactorCov fcov() {
  FactorCov c;
  c.factors = {"FX:EUR", "FX:JPY"};
  c.cov = Matrix::from_rows({{3.6e-5, 1.1e-5}, {1.1e-5, 4.9e-5}});
  return c;
}

#define EXPECT_INVALID(expr) EXPECT_THROW((void)(expr), std::invalid_argument) << #expr

}  // namespace

TEST(Fuzz, Stats) {
  for (double b : kBad) {
    EXPECT_INVALID(log_gamma(b));
    EXPECT_INVALID(norm_ppf(b));
    EXPECT_INVALID(reg_lower_gamma(b, 1.0));
    EXPECT_INVALID(reg_upper_gamma(1.0, b));
    EXPECT_INVALID(chi2_sf(1.0, b));
    EXPECT_INVALID(reg_inc_beta(b, 1.0, 0.5));
    EXPECT_INVALID(reg_inc_beta(1.0, 1.0, b));
    EXPECT_INVALID(t_pdf(b, 5.0));
    EXPECT_INVALID(t_pdf(0.0, b));
    EXPECT_INVALID(t_cdf(0.1, b));
    EXPECT_INVALID(t_ppf(b, 5.0));
    EXPECT_INVALID(t_ppf(0.1, b));
    EXPECT_INVALID(binom_cdf(1, 10, b));
    EXPECT_INVALID(sample_moments(with_bad(good_pnl(), b)));
    EXPECT_INVALID(sample_std(with_bad(good_pnl(), b)));
    EXPECT_INVALID(validate_alpha(b));
    EXPECT_INVALID(validate_horizon(b));
  }
  EXPECT_INVALID(chi2_sf(kNaN, 1.0));
  EXPECT_INVALID(t_cdf(kNaN, 5.0));
  EXPECT_INVALID(binom_cdf(1, -1, 0.1));
  EXPECT_INVALID(sample_moments({1.0}));
  EXPECT_INVALID(validate_alpha(0.99));
  EXPECT_INVALID(validate_horizon(-1.0));
}

TEST(Fuzz, MatrixAndReturns) {
  for (double b : kBad) {
    Matrix c = fcov().cov;
    c(0, 1) = c(1, 0) = b;
    EXPECT_INVALID(cholesky(c));
    EXPECT_INVALID(robust_cholesky(c));
    EXPECT_INVALID(repair_covariance(c, Repair::kHigham));
    EXPECT_INVALID(symmetric_eigen(c));
    EXPECT_INVALID(matvec(fcov().cov, {b, 1.0}));
    EXPECT_INVALID(quad_form({b, 1.0}, fcov().cov));
    ReturnsMatrix r = returns();
    r.data(5, 1) = b;
    EXPECT_INVALID(validate_returns(r, {"FX:EUR"}, 60));
    EXPECT_INVALID(sample_cov(r));
    EXPECT_INVALID(ewma_cov(r));
    EXPECT_INVALID(ewma_cov(returns(), b));
    EXPECT_INVALID(ewma_volatility(r));
    EXPECT_INVALID(ewma_volatility(returns(), b));
    EXPECT_INVALID(flag_peg_factors(returns(), b));
  }
  EXPECT_INVALID(cholesky(Matrix()));
  EXPECT_INVALID(cholesky(Matrix::from_rows({{-1.0, 0.0}, {0.0, 1.0}})));
  EXPECT_INVALID(matmul(Matrix(2, 3), Matrix(2, 3)));
  EXPECT_INVALID(sample_cov(returns(1)));
  EXPECT_INVALID(ewma_volatility(returns(1)));
  EXPECT_INVALID(validate_returns(returns(10), {"FX:EUR"}, 60));
}

TEST(Fuzz, Book) {
  for (double b : kBad) {
    EXPECT_INVALID(Market({{"EUR", b}}));
    EXPECT_INVALID(Market({{"EUR", 1.1}}, {{"EUR", b}}));
    EXPECT_INVALID(market().forward("EURUSD", b));
    EXPECT_INVALID(CompiledBook(Book({SpotPosition{"EURUSD", b, {}}}), market()));
    EXPECT_INVALID(CompiledBook(Book({CashPosition{"EUR", b}}), market()));
    EXPECT_INVALID(CompiledBook(Book({ForwardPosition{"EURUSD", 1e6, b, {}}}), market()));
    EXPECT_INVALID(CompiledBook(Book({ForwardPosition{"EURUSD", 1e6, 0.5, b}}), market()));
    const CompiledBook cb(book(), market());
    EXPECT_INVALID(cb.pnl(std::map<std::string, double>{{"FX:EUR", b}}));
    ReturnsMatrix r = returns();
    r.data(3, 0) = b;
    EXPECT_INVALID(cb.pnl(r));
    EXPECT_INVALID(cb.linear_exposures(b));
  }
  EXPECT_INVALID(CompiledBook(Book{}, market()));
  EXPECT_INVALID(Market({{"EUR", -1.0}}));
  EXPECT_INVALID(split_pair("EURUS"));
  EXPECT_INVALID(CompiledBook(Book({ForwardPosition{"EURUSD", 1e6, -1.0, {}}}), market()));
}

TEST(Fuzz, ExpectedShortfallAndParametric) {
  const std::vector<double> pnl = good_pnl();
  const std::vector<double> w = {1e6, -2e6};
  for (double b : kBad) {
    EXPECT_INVALID(empirical_var(with_bad(pnl, b), 0.01));
    EXPECT_INVALID(empirical_es(pnl, b));
    EXPECT_INVALID(empirical_var_es(pnl, 0.01, with_bad(std::vector<double>(pnl.size(), 1.0), b)));
    EXPECT_INVALID(normal_var(b, 0.01));
    EXPECT_INVALID(normal_var(1.0, 0.01, b));
    EXPECT_INVALID(normal_es(b, 0.01));
    EXPECT_INVALID(t_var(1.0, 0.01, b));
    EXPECT_INVALID(t_es(b, 0.01));
    EXPECT_INVALID(t_es(1.0, 0.01, 6.0, b));
    EXPECT_INVALID(portfolio_sigma({b, 1.0}, fcov().cov));
    EXPECT_INVALID(var_covar(w, fcov().cov, b));
    EXPECT_INVALID(var_covar(w, fcov().cov, 0.01, b));
    EXPECT_INVALID(var_covar(w, fcov().cov, 0.01, 1.0, TailDist::kStudentT, b));
    EXPECT_INVALID(var_covar(w, fcov().cov, 0.01, 1.0, TailDist::kNormal, 6.0, b));
    ParametricOptions po;
    po.alpha = b;
    EXPECT_INVALID(parametric_var(book(), market(), returns(), po));
    ReturnsMatrix r = returns();
    r.data(4, 0) = b;
    EXPECT_INVALID(parametric_var(book(), market(), r));
    EXPECT_INVALID(cornish_fisher_z(b, 0.0, 0.0));
    EXPECT_INVALID(cornish_fisher_valid(b, 0.0, 0.01));
    EXPECT_INVALID(cornish_fisher_valid(0.0, 0.0, b));
    EXPECT_INVALID(cornish_fisher_domain_ok(0.0, b));
    EXPECT_INVALID(cornish_fisher_var(b, 0.0, 0.0));
    EXPECT_INVALID(cornish_fisher_var(1.0, b, 0.0, 0.01, 0.0, 1.0, false));
    EXPECT_INVALID(cornish_fisher_var(1.0, 0.0, 0.0, 0.01, b));
  }
  EXPECT_INVALID(empirical_var({}, 0.01));
  EXPECT_INVALID(empirical_var(pnl, 0.01, {1.0}));
  EXPECT_INVALID(normal_var(-1.0, 0.01));
  EXPECT_INVALID(t_var(1.0, 0.01, 2.0));
  EXPECT_INVALID(portfolio_sigma({}, Matrix()));
  EXPECT_INVALID(portfolio_sigma({0.0, 0.0}, fcov().cov));
  EXPECT_INVALID(cornish_fisher_var(-1.0, 0.0, 0.0));
}

TEST(Fuzz, HistoricalAndMonteCarlo) {
  const std::vector<double> pnl = good_pnl(500);
  Rng rng(1);
  for (double b : kBad) {
    HistoricalOptions ho;
    ho.alpha = b;
    EXPECT_INVALID(historical_var(book(), market(), returns(), ho));
    ho = {};
    ho.method = HsMethod::kAge;
    ho.decay = b;
    EXPECT_INVALID(historical_var(book(), market(), returns(), ho));
    ho = {};
    ho.horizon_days = b;
    EXPECT_INVALID(historical_var(book(), market(), returns(), ho));
    ReturnsMatrix r = returns();
    r.data(4, 0) = b;
    EXPECT_INVALID(historical_var(book(), market(), r));
    EXPECT_INVALID(rng.gamma(b));
    EXPECT_INVALID(rng.chisq(b));
    EXPECT_INVALID(rng.poisson(b));
    EXPECT_INVALID(simulate_factor_returns(fcov(), 10, McDist::kStudentT, b));
    JumpSpec j;
    j.prob = b;
    EXPECT_INVALID(simulate_factor_returns(fcov(), 10, McDist::kJump, 6.0, j));
    j = {};
    j.prob = 0.1;
    j.mean = {{"FX:EUR", b}};
    EXPECT_INVALID(simulate_factor_returns(fcov(), 10, McDist::kJump, 6.0, j));
    j.mean = {{"FX:EUR", -0.1}};
    j.stdev = {{"FX:EUR", b}};
    EXPECT_INVALID(simulate_factor_returns(fcov(), 10, McDist::kJump, 6.0, j));
    EXPECT_INVALID(simulate_factor_returns(fcov(), 10, McDist::kNormal, 6.0, {}, 0, b));
    EXPECT_INVALID(var_standard_error(with_bad(pnl, b), 0.01));
    EXPECT_INVALID(var_standard_error_bootstrap(with_bad(pnl, b), 0.01));
    MonteCarloOptions mo;
    mo.n_scenarios = 1000;
    mo.alpha = b;
    EXPECT_INVALID(monte_carlo_var(book(), market(), fcov(), mo));
  }
  EXPECT_INVALID(rng.poisson(-1.0));
  EXPECT_INVALID(simulate_factor_returns(fcov(), 0));
  EXPECT_INVALID(var_standard_error({1.0, 2.0}, 0.01));
  EXPECT_INVALID(var_standard_error_bootstrap(pnl, 0.01, 1));
  MonteCarloOptions mo;
  mo.n_scenarios = 1000;
  mo.n_boot = 1;
  EXPECT_INVALID(monte_carlo_var(book(), market(), fcov(), mo));
  EXPECT_INVALID(monte_carlo_var(Book{}, market(), fcov()));
  EXPECT_INVALID(historical_var(Book{}, market(), returns()));
}

TEST(Fuzz, BacktestAndStress) {
  const std::vector<double> pnl = good_pnl(250);
  for (double b : kBad) {
    EXPECT_INVALID(kupiec_pof({.n_obs = 250, .n_exceptions = 3, .alpha = b}));
    EXPECT_INVALID(kupiec_pof(3, 250, b));
    EXPECT_INVALID(basel_traffic_light(BaselInput{.n_obs = 250, .n_exceptions = 3, .alpha = b}));
    EXPECT_INVALID(basel_traffic_light(3, 250, b));
    EXPECT_INVALID(conditional_coverage(std::vector<int>(10, 0), b));
    EXPECT_INVALID(acerbi_szekely_z2(with_bad(pnl, b), {150.0}, {180.0}, 0.025));
    EXPECT_INVALID(acerbi_szekely_z2(pnl, {b}, {180.0}, 0.025));
    EXPECT_INVALID(acerbi_szekely_z2(pnl, {150.0}, {b}, 0.025));
    EXPECT_INVALID(evaluate_var_backtest(with_bad(pnl, b), std::vector<double>(250, 150.0)));
    EXPECT_INVALID(evaluate_var_backtest(pnl, with_bad(std::vector<double>(250, 150.0), b)));
    EXPECT_INVALID(simple_to_log(b));
    EXPECT_INVALID(usd_broad_move({"EUR"}, b));
    EXPECT_INVALID(peg_break_scenario("HKD", b));
    EXPECT_INVALID(reverse_stress_linear({b, 1.0}, fcov().cov, 2.0));
    EXPECT_INVALID(reverse_stress_linear({1e6, 1.0}, fcov().cov, b));
    EXPECT_INVALID(reverse_stress_for_loss({1e6, 1.0}, fcov().cov, b));
    EXPECT_INVALID(reverse_stress_numerical({1e6, 1.0}, fcov().cov, b));
    std::map<std::string, Scenario> sc = {{"x", Scenario{"x", {{"FX:EUR", b}}, ""}}};
    EXPECT_INVALID(run_stress(book(), market(), sc));
  }
  EXPECT_INVALID(kupiec_pof({.n_obs = 0, .n_exceptions = 0, .alpha = 0.01}));
  EXPECT_INVALID(kupiec_pof({.n_obs = 10, .n_exceptions = 11, .alpha = 0.01}));
  EXPECT_INVALID(basel_traffic_light(BaselInput{.n_obs = 10, .n_exceptions = -1}));
  EXPECT_INVALID(christoffersen_independence({1}));
  EXPECT_INVALID(christoffersen_independence({0, 2}));
  EXPECT_INVALID(evaluate_var_backtest({1.0, 2.0}, {-1.0, 1.0}));
  EXPECT_INVALID(acerbi_szekely_z2({}, {1.0}, {1.0}, 0.025));
  EXPECT_INVALID(run_stress(Book{}, market(), historical_scenarios()));
  EXPECT_INVALID(reverse_stress_linear({1e6, 1.0}, fcov().cov, -1.0));
}
