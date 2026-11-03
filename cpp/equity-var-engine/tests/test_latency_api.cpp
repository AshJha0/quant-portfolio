// The noexcept kernel API must reproduce the throwing API bit for bit, and
// the fused Monte Carlo kernel must draw the same scenarios as
// simulate_factor_returns() + portfolio_pnl().

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "eqvar/expected_shortfall.hpp"
#include "eqvar/historical.hpp"
#include "eqvar/kernels.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/monte_carlo.hpp"
#include "eqvar/parametric.hpp"
#include "eqvar/returns.hpp"

using namespace eqvar;

namespace {

static_assert(noexcept(try_validate_alpha(0.01)));
static_assert(noexcept(try_var_es_order_statistic(std::span<const double>{}, 0.01,
                                                  std::span<double>{})));
static_assert(noexcept(try_historical_var(std::span<const double>{}, 0.01, std::span<double>{})));
static_assert(noexcept(try_expected_shortfall(std::span<const double>{}, 0.01,
                                              std::span<double>{})));
static_assert(noexcept(try_matvec(std::declval<const Matrix&>(), std::span<const double>{},
                                  std::span<double>{})));
static_assert(noexcept(try_quad_form(std::span<const double>{}, std::declval<const Matrix&>())));
static_assert(noexcept(try_cholesky_lower(std::declval<const Matrix&>(), std::span<double>{},
                                          std::span<double>{})));
static_assert(noexcept(try_portfolio_sigma(std::span<const double>{},
                                           std::declval<const Matrix&>())));
static_assert(noexcept(try_parametric_var(std::span<const double>{},
                                          std::declval<const Matrix&>())));
static_assert(noexcept(try_mc_tail_metrics(std::span<const double>{}, 0.01, 10, 0,
                                           std::declval<McVarWorkspace&>())));
static_assert(noexcept(try_monte_carlo_var_factored(std::span<const double>{},
                                                    std::span<const double>{}, 0.01, 10,
                                                    Dist::Normal, 6.0, 0, 10,
                                                    std::declval<McVarWorkspace&>())));
static_assert(std::is_trivially_copyable_v<Error>);

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

std::vector<double> sample(std::size_t n, std::uint64_t seed) {
    RandomStream rng(seed);
    std::vector<double> x(n);
    for (auto& v : x) v = 1e4 * rng.gaussian() * (1.0 + rng.uniform());
    return x;
}

Matrix test_cov(std::size_t n) {
    Matrix c(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double vi = 0.01 + 0.001 * static_cast<double>(i);
            const double vj = 0.01 + 0.001 * static_cast<double>(j);
            c(i, j) = vi * vj * (i == j ? 1.0 : 0.4);
        }
    return c;
}

}  // namespace

TEST(NoexceptApi, HistoricalKernelsEqualThrowingApi) {
    for (std::size_t n : {50UL, 51UL, 250UL, 1000UL, 10007UL}) {
        const auto x = sample(n, n);
        std::vector<double> ws(n);
        for (double a : {0.001, 0.01, 0.025, 0.05, 0.25}) {
            const TailRisk t = var_es_order_statistic(x, a);
            const TailRisk u = *try_var_es_order_statistic(x, a, ws);
            EXPECT_TRUE(same_bits(t.var, u.var) && same_bits(t.es, u.es));
            EXPECT_TRUE(same_bits(historical_var(x, a), *try_historical_var(x, a, ws)));
            EXPECT_TRUE(same_bits(expected_shortfall(x, a), *try_expected_shortfall(x, a, ws)));
        }
    }
}

TEST(NoexceptApi, ErrorsCarryTheThrowingMessage) {
    const auto x = sample(30, 1);
    std::vector<double> ws(30);
    const auto r = try_historical_var(x, 0.01, ws);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, Errc::TooFewPnl);
    try {
        (void)historical_var(x, 0.01);
        FAIL() << "expected throw";
    } catch (const std::invalid_argument& e) {
        EXPECT_EQ(message(r.error()), e.what());
    }
    EXPECT_EQ(try_validate_alpha(0.7).error().code, Errc::BadAlpha);
    std::vector<double> small(10);
    EXPECT_EQ(try_var_es_order_statistic(x, 0.01, small).error().code, Errc::WorkspaceTooSmall);
}

TEST(NoexceptApi, MatrixAndParametricEqualThrowingApi) {
    for (std::size_t n : {1UL, 3UL, 4UL, 7UL, 50UL}) {
        const Matrix cov = test_cov(n);
        const auto w = sample(n, 100 + n);
        std::vector<double> y(n);
        ASSERT_TRUE(try_matvec(cov, w, y).has_value());
        const auto y0 = matvec(cov, w);
        for (std::size_t i = 0; i < n; ++i) EXPECT_TRUE(same_bits(y[i], y0[i]));
        EXPECT_TRUE(same_bits(*try_quad_form(w, cov), quad_form(w, cov)));
        EXPECT_TRUE(same_bits(*try_portfolio_sigma(w, cov), portfolio_sigma(w, cov)));
        for (Dist d : {Dist::Normal, Dist::StudentT}) {
            EXPECT_TRUE(same_bits(*try_parametric_var(w, cov, 0.01, d, 5.0, 1e-4, 10),
                                  parametric_var(w, cov, 0.01, d, 5.0, 1e-4, 10)));
        }
        std::vector<double> l(n * n), scratch(n * n);
        const auto jit = try_cholesky_lower(cov, l, scratch);
        ASSERT_TRUE(jit.has_value());
        const auto c = cholesky(cov);
        EXPECT_TRUE(same_bits(*jit, c.jitter_added));
        for (std::size_t k = 0; k < n * n; ++k) EXPECT_TRUE(same_bits(l[k], c.lower.data()[k]));
    }
}

TEST(NoexceptApi, CholeskyJitterLadderMatchesThrowingApi) {
    // Rank-deficient (rho = 1): plain Cholesky fails, the ladder adds jitter.
    Matrix cov(3, 3, 1e-4);
    std::vector<double> l(9), scratch(9);
    const auto jit = try_cholesky_lower(cov, l, scratch);
    ASSERT_TRUE(jit.has_value());
    const auto c = cholesky(cov);
    EXPECT_GT(*jit, 0.0);
    EXPECT_TRUE(same_bits(*jit, c.jitter_added));
    for (std::size_t k = 0; k < 9; ++k) EXPECT_TRUE(same_bits(l[k], c.lower.data()[k]));
    Matrix bad(2, 2);
    bad(0, 0) = 1.0;
    bad(1, 1) = 1.0;
    bad(0, 1) = bad(1, 0) = 2.0;
    EXPECT_EQ(try_cholesky_lower(bad, l, scratch).error().code, Errc::CholIndefinite);
}

TEST(NoexceptApi, FusedMonteCarloEqualsMonteCarloVar) {
    for (Dist d : {Dist::Normal, Dist::StudentT}) {
        const std::size_t n = 6, paths = 4000;
        const Matrix cov = test_cov(n);
        const auto w = sample(n, 7);
        const double df = 5.0;
        Matrix scale = cov;
        if (d == Dist::StudentT) {
            for (std::size_t i = 0; i < n * n; ++i) scale.data()[i] *= (df - 2.0) / df;
        }
        std::vector<double> l(n * n), scratch(n * n);
        ASSERT_TRUE(try_cholesky_lower(scale, l, scratch).has_value());
        McVarWorkspace ws;
        ws.reserve(n, paths);
        const auto a = try_monte_carlo_var_factored(w, l, 0.01, paths, d, df, 42, 50, ws);
        ASSERT_TRUE(a.has_value());
        const MonteCarloResult b = monte_carlo_var(w, cov, 0.01, paths, d, df, 42, 50);
        EXPECT_TRUE(same_bits(a->var, b.var) && same_bits(a->es, b.es) &&
                    same_bits(a->var_se, b.var_se) && same_bits(a->var_se_kde, b.var_se_kde));
        // W4(a): the fused kernel uses v = L'w once per run, so it equals the
        // two-step simulate + portfolio_pnl path (full matvec per path) to
        // rounding only. Per path the bound is 1e-12 of sum_i |w_i r_i|.
        const Matrix scen = simulate_factor_returns(cov, paths, d, df, 42);
        const auto pnl = portfolio_pnl(scen, w);
        for (std::size_t p = 0; p < paths; ++p) {
            double scale = 0.0;
            for (std::size_t i = 0; i < n; ++i) scale += std::fabs(w[i] * scen(p, i));
            ASSERT_LE(std::fabs(ws.pnl[p] - pnl[p]), 1e-12 * scale) << p;
        }
        const MonteCarloResult old = mc_tail_metrics(pnl, 0.01, 50, 42 ^ 0x9E3779B97F4A7C15ULL);
        EXPECT_LE(std::fabs(a->var - old.var), 1e-12 * std::fabs(old.var));
        EXPECT_LE(std::fabs(a->es - old.es), 1e-12 * std::fabs(old.es));
        // Re-running on the same workspace repeats the result exactly.
        const auto c = try_monte_carlo_var_factored(w, l, 0.01, paths, d, df, 42, 50, ws);
        EXPECT_TRUE(same_bits(a->var_se, c->var_se));
    }
}

TEST(BatchApi, ScenarioPnlMatvecEqualsRowByRowSum) {
    // try_matvec is the scenario P&L batch kernel; check the 4-row blocking
    // against a plain per-row left-to-right sum for every remainder.
    for (std::size_t rows : {1UL, 3UL, 4UL, 5UL, 8UL, 1001UL}) {
        Matrix panel(rows, 9);
        const auto v = sample(rows * 9, rows);
        for (std::size_t k = 0; k < rows * 9; ++k) panel.data()[k] = v[k] * 1e-6;
        const auto w = sample(9, 3);
        std::vector<double> y(rows);
        ASSERT_TRUE(try_matvec(panel, w, y).has_value());
        for (std::size_t i = 0; i < rows; ++i) {
            double acc = 0.0;
            for (std::size_t j = 0; j < 9; ++j) acc += panel(i, j) * w[j];
            EXPECT_TRUE(same_bits(y[i], acc)) << rows << " " << i;
        }
    }
}
