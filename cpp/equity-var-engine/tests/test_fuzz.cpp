// Fuzz-style input walk: every public entry point of the library is called
// with NaN, +inf, -inf, empty, length-1 and negative inputs where those are
// invalid, and must throw std::invalid_argument. Never a number.
//
// When you add a public function, add it here. The list mirrors the headers
// in include/eqvar/.

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "eqvar/backtest.hpp"
#include "eqvar/expected_shortfall.hpp"
#include "eqvar/historical.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/monte_carlo.hpp"
#include "eqvar/parametric.hpp"
#include "eqvar/returns.hpp"
#include "eqvar/stats.hpp"

using namespace eqvar;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
const double kBad[] = {kNaN, kInf, -kInf};

std::vector<double> good_pnl(std::size_t n = 200) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = 100.0 * std::sin(0.37 * static_cast<double>(i) + 0.1);
    return v;
}

std::vector<double> with_bad(std::vector<double> v, double bad) {
    v[v.size() / 2] = bad;
    return v;
}

Matrix good_returns(std::size_t t = 120, std::size_t n = 2) {
    Matrix r(t, n);
    for (std::size_t i = 0; i < t; ++i)
        for (std::size_t j = 0; j < n; ++j)
            r(i, j) = 0.01 * std::sin(0.3 * static_cast<double>(i) + static_cast<double>(j));
    return r;
}

Matrix cov2() { return Matrix(2, 2, {1.0, 0.3, 0.3, 2.0}); }

#define EXPECT_INVALID(expr) EXPECT_THROW((void)(expr), std::invalid_argument) << #expr

}  // namespace

TEST(Fuzz, Stats) {
    for (double b : kBad) {
        EXPECT_INVALID(normal_ppf(b));
        EXPECT_INVALID(log_gamma(b));
        EXPECT_INVALID(betainc_reg(b, 1.0, 0.5));
        EXPECT_INVALID(betainc_reg(1.0, 1.0, b));
        EXPECT_INVALID(regularized_gamma_p(b, 1.0));
        EXPECT_INVALID(regularized_gamma_q(1.0, b));
        EXPECT_INVALID(chi2_sf(1.0, b));
        EXPECT_INVALID(binomial_cdf(1, 10, b));
        EXPECT_INVALID(student_t_pdf(b, 5.0));
        EXPECT_INVALID(student_t_pdf(0.0, b));
        EXPECT_INVALID(student_t_cdf(0.1, b));
        EXPECT_INVALID(student_t_ppf(b, 5.0));
        EXPECT_INVALID(student_t_ppf(0.1, b));
        EXPECT_INVALID(mean(with_bad(good_pnl(), b)));
        EXPECT_INVALID(stdev(with_bad(good_pnl(), b)));
        EXPECT_INVALID(skewness(with_bad(good_pnl(), b)));
        EXPECT_INVALID(excess_kurtosis(with_bad(good_pnl(), b)));
    }
    EXPECT_INVALID(chi2_sf(kNaN, 1.0));
    EXPECT_INVALID(student_t_cdf(kNaN, 5.0));
    EXPECT_INVALID(log_gamma(-1.0));
    EXPECT_INVALID(binomial_cdf(1, 0, 0.1));
    EXPECT_INVALID(mean(std::vector<double>{}));
    EXPECT_INVALID(stdev(std::vector<double>{1.0}));
}

TEST(Fuzz, ReturnsAndMatrix) {
    const std::vector<double> prices = {100.0, 101.0, 99.0};
    for (double b : kBad) {
        EXPECT_INVALID(log_returns(with_bad(prices, b)));
        EXPECT_INVALID(simple_returns(with_bad(prices, b)));
        Matrix r = good_returns();
        r(3, 1) = b;
        EXPECT_INVALID(portfolio_pnl(r, std::vector<double>{1.0, 1.0}));
        EXPECT_INVALID(portfolio_pnl(good_returns(), std::vector<double>{b, 1.0}));
        EXPECT_INVALID(sample_covariance(r));
        EXPECT_INVALID(ewma_covariance(r, 0.94));
        EXPECT_INVALID(ewma_covariance(good_returns(), b));
        Matrix c = cov2();
        c(0, 1) = c(1, 0) = b;
        EXPECT_INVALID(cholesky(c));
        EXPECT_INVALID(repair_covariance(c, Repair::Higham));
        EXPECT_INVALID(symmetric_eigen(c));
        EXPECT_INVALID(matvec(c, std::vector<double>{1.0, 1.0}));
        EXPECT_INVALID(quad_form(std::vector<double>{b, 1.0}, cov2()));
        EXPECT_INVALID(covariance_from_vols(std::vector<double>{b, 1.0}, Matrix::identity(2)));
        EXPECT_INVALID(validate_alpha(b));
        EXPECT_INVALID(validate_pnl(with_bad(good_pnl(), b), 10));
        EXPECT_INVALID(validate_exposures(std::vector<double>{b}, "fuzz"));
    }
    EXPECT_INVALID(log_returns(std::vector<double>{100.0}));
    EXPECT_INVALID(log_returns(std::vector<double>{100.0, -1.0}));
    EXPECT_INVALID(portfolio_pnl(good_returns(), std::vector<double>{}));
    EXPECT_INVALID(sample_covariance(good_returns(1, 2)));
    EXPECT_INVALID(cholesky(Matrix(2, 2, {-1.0, 0.0, 0.0, 1.0})));  // negative variance
    EXPECT_INVALID(cholesky(Matrix()));
    EXPECT_INVALID(covariance_from_vols(std::vector<double>{-1.0, 1.0}, Matrix::identity(2)));
    EXPECT_INVALID(validate_exposures(std::vector<double>{}, "fuzz"));
    EXPECT_INVALID(validate_exposures(std::vector<double>{0.0, 0.0}, "fuzz"));
}

TEST(Fuzz, Historical) {
    const std::vector<double> pnl = good_pnl();
    for (double b : kBad) {
        const std::vector<double> bad = with_bad(pnl, b);
        EXPECT_INVALID(var_es_order_statistic(bad, 0.01));
        EXPECT_INVALID(var_es_order_statistic(pnl, b));
        EXPECT_INVALID(var_es_weighted(bad, std::vector<double>(pnl.size(), 1.0), 0.01));
        EXPECT_INVALID(
            var_es_weighted(pnl, with_bad(std::vector<double>(pnl.size(), 1.0), b), 0.01));
        EXPECT_INVALID(quantile_linear(bad, 0.5));
        EXPECT_INVALID(quantile_linear(pnl, b));
        EXPECT_INVALID(historical_var(bad, 0.01));
        EXPECT_INVALID(historical_var(pnl, b));
        EXPECT_INVALID(brw_weights(10, b));
        EXPECT_INVALID(age_weighted_var(bad, 0.01));
        EXPECT_INVALID(age_weighted_es(pnl, 0.01, b));
        EXPECT_INVALID(ewma_volatility(bad));
        EXPECT_INVALID(ewma_volatility(pnl, b));
        EXPECT_INVALID(ewma_volatility_next(bad));
        Matrix r = good_returns();
        r(7, 0) = b;
        EXPECT_INVALID(filtered_historical_var_es(r, std::vector<double>{1.0, 1.0}));
        EXPECT_INVALID(filtered_historical_var_es(good_returns(), std::vector<double>{b, 1.0}));
        EXPECT_INVALID(filtered_historical_var(bad));
        EXPECT_INVALID(scale_var_sqrt_time(b, 10));
    }
    EXPECT_INVALID(var_es_order_statistic(std::vector<double>{}, 0.01));
    EXPECT_INVALID(var_es_order_statistic(pnl, -0.01));
    EXPECT_INVALID(var_es_order_statistic(pnl, 0.99));  // a confidence level, not a tail
    EXPECT_INVALID(var_es_weighted(pnl, std::vector<double>(pnl.size(), -1.0), 0.01));
    EXPECT_INVALID(historical_var(std::vector<double>{1.0}, 0.01));
    EXPECT_INVALID(brw_weights(0, 0.9));
    EXPECT_INVALID(ewma_volatility(std::vector<double>{1.0}));
    EXPECT_INVALID(ewma_volatility(pnl, 0.94, 0));
    EXPECT_INVALID(filtered_historical_var_es(good_returns(10, 2), std::vector<double>{1.0, 1.0}));
    EXPECT_INVALID(filtered_historical_var_es(good_returns(), std::vector<double>{}));
    EXPECT_INVALID(scale_var_sqrt_time(1.0, 0));
    EXPECT_INVALID(scale_var_sqrt_time(1.0, -5));
}

TEST(Fuzz, ParametricAndEs) {
    const std::vector<double> w = {1.0, 2.0};
    for (double b : kBad) {
        EXPECT_INVALID(expected_shortfall(with_bad(good_pnl(), b), 0.01));
        EXPECT_INVALID(expected_shortfall(good_pnl(), b));
        EXPECT_INVALID(normal_es(b, 0.01));
        EXPECT_INVALID(normal_es(1.0, b));
        EXPECT_INVALID(normal_es(1.0, 0.01, b));
        EXPECT_INVALID(student_t_es(b, 0.01));
        EXPECT_INVALID(student_t_es(1.0, 0.01, b));
        EXPECT_INVALID(student_t_es(1.0, 0.01, 6.0, b));
        EXPECT_INVALID(portfolio_sigma(std::vector<double>{b, 1.0}, cov2()));
        EXPECT_INVALID(parametric_var(w, cov2(), b));
        EXPECT_INVALID(parametric_var(w, cov2(), 0.01, Dist::StudentT, b));
        EXPECT_INVALID(parametric_var(w, cov2(), 0.01, Dist::Normal, 6.0, b));
        EXPECT_INVALID(cornish_fisher_z(b, 0.0, 0.0));
        EXPECT_INVALID(cornish_fisher_valid(b, 0.0, 0.01));
        EXPECT_INVALID(cornish_fisher_valid(0.0, b, 0.01));
        EXPECT_INVALID(cornish_fisher_valid(0.0, 0.0, b));
        EXPECT_INVALID(cornish_fisher_domain_ok(b, 0.0));
        EXPECT_INVALID(cornish_fisher_var(b, 0.01));
        EXPECT_INVALID(cornish_fisher_var(1.0, 0.01, b, 0.0, 0.0, false));
        EXPECT_INVALID(cornish_fisher_var(1.0, 0.01, 0.0, 0.0, b));
    }
    EXPECT_INVALID(normal_es(-1.0, 0.01));
    EXPECT_INVALID(student_t_es(1.0, 0.01, 2.0));
    EXPECT_INVALID(portfolio_sigma(std::vector<double>{}, cov2()));
    EXPECT_INVALID(parametric_var(w, cov2(), 0.01, Dist::Normal, 6.0, 0.0, 0));
    EXPECT_INVALID(cornish_fisher_var(-1.0, 0.01));
    EXPECT_INVALID(expected_shortfall(std::vector<double>{1.0}, 0.01));
}

TEST(Fuzz, MonteCarlo) {
    const std::vector<double> w = {1.0, 2.0};
    const std::vector<double> pnl = good_pnl(500);
    RandomStream rng(1);
    for (double b : kBad) {
        EXPECT_INVALID(rng.gamma(b));
        EXPECT_INVALID(rng.chi_squared(b));
        EXPECT_INVALID(simulate_factor_returns(cov2(), 10, Dist::StudentT, b));
        Matrix c = cov2();
        c(1, 1) = b;
        EXPECT_INVALID(simulate_factor_returns(c, 10));
        EXPECT_INVALID(mc_tail_metrics(with_bad(pnl, b), 0.01));
        EXPECT_INVALID(mc_tail_metrics(pnl, b));
        EXPECT_INVALID(mc_bootstrap_se(with_bad(pnl, b), 0.01));
        EXPECT_INVALID(kde_var_se(with_bad(pnl, b), 0.01));
        EXPECT_INVALID(monte_carlo_var(std::vector<double>{b, 1.0}, cov2(), 0.01, 1000));
        EXPECT_INVALID(monte_carlo_var(w, cov2(), b, 1000));
    }
    EXPECT_INVALID(rng.gamma(-1.0));
    EXPECT_INVALID(simulate_factor_returns(cov2(), 0));
    EXPECT_INVALID(simulate_factor_returns(cov2(), 10, Dist::StudentT, 2.0));
    EXPECT_INVALID(mc_tail_metrics(good_pnl(99), 0.01));
    EXPECT_INVALID(mc_bootstrap_se(pnl, 0.01, 1));
    EXPECT_INVALID(mc_bootstrap_se(pnl, 0.01, 0));
    EXPECT_INVALID(kde_var_se(good_pnl(5), 0.01));
    EXPECT_INVALID(monte_carlo_var(std::vector<double>{}, cov2(), 0.01, 1000));
    EXPECT_INVALID(monte_carlo_var(std::vector<double>{0.0, 0.0}, cov2(), 0.01, 1000));
}

TEST(Fuzz, Backtest) {
    const std::vector<double> pnl = good_pnl(250);
    const std::vector<double> var = {150.0};
    const std::vector<double> es = {180.0};
    for (double b : kBad) {
        EXPECT_INVALID(exceptions_from_pnl(with_bad(pnl, b), var));
        EXPECT_INVALID(exceptions_from_pnl(pnl, std::vector<double>{b}));
        EXPECT_INVALID(kupiec_pof({.n_obs = 250, .n_exceptions = 3, .alpha = b}));
        EXPECT_INVALID(kupiec_pof(250, 3, b));
        EXPECT_INVALID(
            basel_traffic_light(BaselInput{.n_obs = 250, .n_exceptions = 3, .alpha = b}));
        EXPECT_INVALID(christoffersen_cc(std::vector<std::uint8_t>(10, 0), b));
        EXPECT_INVALID(acerbi_szekely_z2(with_bad(pnl, b), var, es, 0.025));
        EXPECT_INVALID(acerbi_szekely_z2(pnl, std::vector<double>{b}, es, 0.025));
        EXPECT_INVALID(acerbi_szekely_z2(pnl, var, std::vector<double>{b}, 0.025));
        EXPECT_INVALID(acerbi_szekely_z2(pnl, var, es, b));
    }
    EXPECT_INVALID(exceptions_from_pnl(std::vector<double>{}, var));
    EXPECT_INVALID(exceptions_from_pnl(pnl, std::vector<double>{-1.0}));
    EXPECT_INVALID(kupiec_pof({.n_obs = 0, .n_exceptions = 0, .alpha = 0.01}));
    EXPECT_INVALID(kupiec_pof({.n_obs = 10, .n_exceptions = -1, .alpha = 0.01}));
    EXPECT_INVALID(kupiec_pof({.n_obs = 10, .n_exceptions = 11, .alpha = 0.01}));
    EXPECT_INVALID(basel_traffic_light(BaselInput{.n_obs = 10, .n_exceptions = 11}));
    EXPECT_INVALID(basel_traffic_light(11, 10));
    EXPECT_INVALID(christoffersen_independence(std::vector<std::uint8_t>{1}));
    EXPECT_INVALID(acerbi_szekely_z2(std::vector<double>{}, var, es, 0.025));
    EXPECT_INVALID(acerbi_szekely_z2(pnl, std::vector<double>{-1.0}, es, 0.025));
    EXPECT_INVALID(acerbi_szekely_z2(pnl, var, std::vector<double>{1.0, 2.0}, 0.025));
}
