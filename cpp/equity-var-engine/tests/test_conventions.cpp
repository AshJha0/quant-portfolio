// Pins the shared VaR-engine conventions: order-statistic VaR / fractional
// ES, Kupiec and Basel on any window, Acerbi-Szekely Z2, alpha-aware
// Cornish-Fisher validity, capped Cholesky with PSD repair, EWMA seed,
// per-factor FHS, bootstrap SE and the empty-book rule. Each expected value
// is computed by hand or by an independent route in the test itself.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "eqvar/backtest.hpp"
#include "eqvar/expected_shortfall.hpp"
#include "eqvar/historical.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/monte_carlo.hpp"
#include "eqvar/parametric.hpp"
#include "eqvar/stats.hpp"

using namespace eqvar;

namespace {
std::vector<double> ten_losses() {
    std::vector<double> pnl;
    for (int i = 1; i <= 10; ++i) pnl.push_back(-static_cast<double>(i));  // losses 1..10
    return pnl;
}
}  // namespace

TEST(OrderStatistic, FractionalAtomByHand) {
    // n = 10, alpha = 0.25: k = 2.5, m = 3, L = 10, 9, 8.
    // VaR = 8, ES = (10 + 9 + 0.5 * 8) / 2.5 = 9.2. The old type-7 VaR gave 7.75.
    const TailRisk r = var_es_order_statistic(ten_losses(), 0.25);
    EXPECT_EQ(r.var, 8.0);
    EXPECT_NEAR(r.es, 9.2, 1e-15);
    // Integer k: alpha = 0.2, k = 2 (m = 2): VaR = 9, ES = (10 + 9) / 2.
    const TailRisk q = var_es_order_statistic(ten_losses(), 0.2);
    EXPECT_EQ(q.var, 9.0);
    EXPECT_NEAR(q.es, 9.5, 1e-15);
}

TEST(OrderStatistic, WeightedMatchesUniformAndHandCase) {
    const std::vector<double> pnl = ten_losses();
    const std::vector<double> flat(10, 3.0);  // normalised to 0.1 each
    const TailRisk u = var_es_weighted(pnl, flat, 0.25);
    EXPECT_EQ(u.var, 8.0);
    EXPECT_NEAR(u.es, 9.2, 1e-14);
    // Weight 0.5 on the worst loss: W_1 = 0.5 >= 0.25 -> VaR = ES = 10.
    std::vector<double> w(10, 0.5 / 9.0);
    w[9] = 0.5;  // pnl[9] = -10
    const TailRisk h = var_es_weighted(pnl, w, 0.25);
    EXPECT_EQ(h.var, 10.0);
    EXPECT_NEAR(h.es, 10.0, 1e-14);
    EXPECT_THROW(var_es_weighted(pnl, std::vector<double>(9, 1.0), 0.25), std::invalid_argument);
}

TEST(Kupiec, StructInputValidatesWithoutClamping) {
    const KupiecResult r = kupiec_pof({.n_obs = 250, .n_exceptions = 5, .alpha = 0.01});
    // LR = -2[245 ln .99 + 5 ln .01 - 245 ln .98 - 5 ln .02].
    const double lr = -2.0 * (245.0 * std::log(0.99) + 5.0 * std::log(0.01) -
                              245.0 * std::log(0.98) - 5.0 * std::log(0.02));
    EXPECT_NEAR(r.lr, lr, 1e-12);
    EXPECT_THROW(kupiec_pof({.n_obs = 250, .n_exceptions = 251, .alpha = 0.01}),
                 std::invalid_argument);
    // alpha is the tail probability: 0.99 is a confidence level and throws.
    EXPECT_THROW(kupiec_pof({.n_obs = 250, .n_exceptions = 5, .alpha = 0.99}),
                 std::invalid_argument);
}

TEST(Basel, ZonesByBinomialCdfOnAnyWindow) {
    // n = 500, x = 8: F = 0.9329 < 0.95 -> green. The old code zoned by the
    // raw count (8 -> yellow) whatever n_obs was.
    const BaselResult g = basel_traffic_light(BaselInput{.n_obs = 500, .n_exceptions = 8});
    EXPECT_EQ(g.zone, BaselZone::Green);
    EXPECT_NEAR(g.cumulative_prob, 0.9328898400862952, 1e-12);
    // n = 1000, x = 15: F = 0.9521 -> yellow, x_eq = 5 (F250(5) = 0.9588) -> +0.40.
    // Old code: 15 > 9 -> red.
    const BaselResult y = basel_traffic_light(BaselInput{.n_obs = 1000, .n_exceptions = 15});
    EXPECT_EQ(y.zone, BaselZone::Yellow);
    EXPECT_DOUBLE_EQ(y.plus_factor, 0.40);
    EXPECT_DOUBLE_EQ(y.multiplier, 3.40);
    // n = 500, x = 9: F = 0.9689 -> x_eq = 6 -> +0.50.
    EXPECT_DOUBLE_EQ(basel_traffic_light(BaselInput{.n_obs = 500, .n_exceptions = 9}).plus_factor,
                     0.50);
    // The 1996 table is reproduced at n = 250.
    const double table[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.40, 0.50, 0.65, 0.75, 0.85, 1.0};
    for (int x = 0; x <= 10; ++x) {
        EXPECT_DOUBLE_EQ(basel_traffic_light(BaselInput{.n_obs = 250, .n_exceptions = x}).plus_factor,
                         table[x])
            << x;
    }
    // x > n used to be clamped; now it throws.
    EXPECT_THROW(basel_traffic_light(BaselInput{.n_obs = 5, .n_exceptions = 6}),
                 std::invalid_argument);
}

TEST(AcerbiSzekely, Z2ByHand) {
    // T = 4, alpha = 0.25, VaR = 1, ES = 2. Exceptions on days 1 and 3.
    // Z2 = (-3 - 1.5) / (4 * 0.25 * 2) + 1 = -1.25.
    const std::vector<double> pnl = {0.5, -3.0, 0.2, -1.5};
    const std::vector<double> var = {1.0};
    const std::vector<double> es = {2.0};
    EXPECT_NEAR(acerbi_szekely_z2(pnl, var, es, 0.25), -1.25, 1e-15);
    // No exceptions -> Z2 = 1.
    EXPECT_EQ(acerbi_szekely_z2(std::vector<double>{1.0, 2.0}, var, es, 0.25), 1.0);
    EXPECT_THROW(acerbi_szekely_z2(pnl, var, std::vector<double>{0.0}, 0.25),
                 std::invalid_argument);
}

TEST(CornishFisher, AlphaAwareValidity) {
    // (0.122, -0.427): g' is concave, its minimum on [-r, r] is at -r, and
    // g'(-4) < 0 but g'(-z_a) > 0 for alpha = 0.01 and 0.001.
    EXPECT_TRUE(cornish_fisher_valid(0.122, -0.427, 0.01));
    EXPECT_TRUE(cornish_fisher_valid(0.122, -0.427, 0.001));
    EXPECT_FALSE(cornish_fisher_domain_ok(0.122, -0.427, 4.0));
    EXPECT_TRUE(cornish_fisher_valid(0.0, 0.0, 0.01));
    // S = 1, K = 1: A = -1/24, B = 1/3, C = 73/72; minimum at -r:
    // r = 2.326 gives +0.013 (valid), r = 3.090 gives -0.415 (invalid).
    EXPECT_TRUE(cornish_fisher_valid(1.0, 1.0, 0.01));
    EXPECT_FALSE(cornish_fisher_valid(1.0, 1.0, 0.001));
    EXPECT_THROW(cornish_fisher_valid(std::nan(""), 0.0, 0.01), std::invalid_argument);
    // Non-finite moments throw even with the domain check off (old code
    // returned NaN here).
    EXPECT_THROW(cornish_fisher_var(1.0, 0.01, std::nan(""), 0.0, 0.0, false),
                 std::invalid_argument);
}

TEST(Cholesky, JacobiEigenByHand) {
    const Matrix a(2, 2, {2.0, 1.0, 1.0, 2.0});
    const EigenResult e = symmetric_eigen(a);
    EXPECT_NEAR(e.values[0], 1.0, 1e-15);
    EXPECT_NEAR(e.values[1], 3.0, 1e-15);
    EXPECT_NEAR(std::abs(e.vectors(0, 0)), std::sqrt(0.5), 1e-15);
}

TEST(Cholesky, IndefiniteThrowsUnlessRepaired) {
    const Matrix a(3, 3, {1.0, 0.9, 0.9, 0.9, 1.0, -0.9, 0.9, -0.9, 1.0});  // lambda_min = -0.8
    EXPECT_THROW(cholesky(a), std::runtime_error);
    // Higham: the nearest correlation matrix has off-diagonals +-0.5
    // (numpy eigh reference, 1e-8 tolerance as in the golden spec).
    const Matrix h = repair_covariance(a, Repair::Higham);
    EXPECT_NEAR(h(0, 1), 0.5, 1e-8);
    EXPECT_NEAR(h(0, 2), 0.5, 1e-8);
    EXPECT_NEAR(h(1, 2), -0.5, 1e-8);
    for (int i = 0; i < 3; ++i) EXPECT_EQ(h(i, i), 1.0);
    // Clip keeps the variances and gives 0.49999999996... (numpy reference).
    const Matrix c = repair_covariance(a, Repair::Clip);
    EXPECT_NEAR(c(0, 1), 0.49999999996052624, 1e-12);
    EXPECT_NEAR(c(1, 2), -0.4999999999605262, 1e-12);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(c(i, i), 1.0, 1e-15);
    const CholeskyResult r = cholesky(a, Repair::Higham);
    EXPECT_TRUE(r.repaired);
    EXPECT_FALSE(r.warning.empty());
}

TEST(Cholesky, JitterStartsAt1e12AndIsCapped) {
    // Perfectly correlated pair: singular PSD, needs a tiny jitter.
    const Matrix s(2, 2, {4.0, 4.0, 4.0, 4.0});
    const CholeskyResult r = cholesky(s);
    EXPECT_GT(r.jitter_added, 0.0);
    EXPECT_LE(r.jitter_added, 1e-6 * 4.0);
    EXPECT_FALSE(r.warning.empty());
    // Symmetry tolerance is relative (1e-10 max|A|).
    const Matrix big(2, 2, {1e12, 1e11, 1e11 + 1.0, 1e12});
    EXPECT_NO_THROW(cholesky(big));
}

TEST(LogGamma, LanczosMatchesClosedForms) {
    EXPECT_NEAR(log_gamma(0.5), 0.5 * std::log(3.14159265358979323846), 1e-14);
    EXPECT_NEAR(log_gamma(10.0), std::log(362880.0), 1e-13);
    EXPECT_NEAR(log_gamma(1.0), 0.0, 1e-15);
    EXPECT_NEAR(log_gamma(2.0), 0.0, 1e-15);
    EXPECT_THROW(log_gamma(0.0), std::invalid_argument);
    EXPECT_THROW(log_gamma(std::nan("")), std::invalid_argument);
}

TEST(Fhs, PerFactorHomogeneousAndSingleFactorMatchesPnlLevel) {
    const std::size_t t_obs = 300;
    Matrix r(t_obs, 2);
    for (std::size_t t = 0; t < t_obs; ++t) {
        const double td = static_cast<double>(t);
        r(t, 0) = 0.01 * std::sin(0.37 * td) * (1.0 + (t > 200 ? 2.0 : 0.0));
        r(t, 1) = 0.02 * std::cos(0.53 * td + 0.2);
    }
    const std::vector<double> w1 = {1.0e6, 0.0};
    const std::vector<double> w2 = {2.0e6, 0.0};
    const TailRisk a = filtered_historical_var_es(r, w1, 0.01);
    const TailRisk b = filtered_historical_var_es(r, w2, 0.01);
    EXPECT_NEAR(b.var, 2.0 * a.var, 1e-9 * a.var);
    // One factor: r*/sigma rescaling of w r equals the P&L-level FHS of w r.
    std::vector<double> pnl(t_obs);
    for (std::size_t t = 0; t < t_obs; ++t) pnl[t] = 1.0e6 * r(t, 0);
    EXPECT_NEAR(a.var, filtered_historical_var(pnl, 0.01), 1e-9 * a.var);
    EXPECT_THROW(filtered_historical_var_es(r, std::vector<double>{0.0, 0.0}, 0.01),
                 std::invalid_argument);
}

TEST(MonteCarloSe, BootstrapNearAsymptoticAndNeedsTwoReplicates) {
    // N(0, 1) P&L, n = 100k: asymptotic SE = sqrt(a(1-a)/n) / phi(z_a).
    const Matrix cov(1, 1, std::vector<double>{1.0});
    const Matrix scen = simulate_factor_returns(cov, 100'000, Dist::Normal, 6.0, 11);
    std::vector<double> pnl(scen.data(), scen.data() + scen.size());
    const double a = 0.01;
    const double se_true = std::sqrt(a * (1 - a) / 1e5) / normal_pdf(normal_ppf(a));
    const MonteCarloResult r = mc_tail_metrics(pnl, a, 400, 3);
    // 400 replicates: relative sampling error of the SE ~ 1/sqrt(2*399) = 3.5 %;
    // 20 % covers that plus the O(n^-1/2) bias of the quantile bootstrap.
    EXPECT_NEAR(r.var_se, se_true, 0.20 * se_true);
    EXPECT_NEAR(r.var_se_kde, se_true, 0.10 * se_true);
    // n_boot = 1 used to divide by zero (NaN); now it throws.
    EXPECT_THROW(mc_bootstrap_se(pnl, a, 1, 0), std::invalid_argument);
}

TEST(EmptyBook, AllZeroExposuresThrow) {
    const Matrix cov(2, 2, {1.0, 0.0, 0.0, 1.0});
    const std::vector<double> zero = {0.0, 0.0};
    EXPECT_THROW(portfolio_sigma(zero, cov), std::invalid_argument);
    EXPECT_THROW(parametric_var(zero, cov, 0.01), std::invalid_argument);
    EXPECT_THROW(monte_carlo_var(zero, cov, 0.01, 1000), std::invalid_argument);
}
