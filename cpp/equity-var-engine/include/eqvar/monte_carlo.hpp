// eqvar/monte_carlo.hpp — Monte Carlo VaR for a linear (delta) portfolio.
//
// Simulate factor returns from a multivariate normal or multivariate
// Student-t (common-mixing-variable construction Z / sqrt(W/df), scale matrix
// cov*(df-2)/df so the COVARIANCE matches cov exactly while the tails
// fatten), revalue the linear portfolio P&L = w . r, and read VaR / ES off
// the scenario distribution with an order-statistic standard error.
//
// Determinism: every draw comes from a seeded std::mt19937_64 through our own
// inverse-CDF normal transform (normal_ppf on 53-bit uniforms) and a
// Marsaglia-Tsang gamma sampler built on the same primitives — NO
// implementation-defined std::normal_distribution — so results are BITWISE
// reproducible for a given seed across standard libraries.
//
// Mirrors eq_var.monte_carlo_var semantically (same factor model, same
// quantile); RNG streams differ from NumPy's, so cross-language agreement is
// statistical (within order-statistic error bars), not bitwise.

#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>

#include "eqvar/matrix.hpp"
#include "eqvar/parametric.hpp"

namespace eqvar {

/// Deterministic random stream: mt19937_64 + inverse-CDF transforms only.
class RandomStream {
public:
    explicit RandomStream(std::uint64_t seed) : engine_(seed) {}

    /// 53-bit uniform on the open interval (0, 1).
    [[nodiscard]] double uniform();

    /// Standard normal via Phi^{-1}(uniform()) — one draw per variate,
    /// platform-independent (unlike std::normal_distribution).
    [[nodiscard]] double gaussian();

    /// Gamma(shape, scale = 1) via Marsaglia-Tsang (2000), with the
    /// U^{1/a} boost for shape < 1.
    [[nodiscard]] double gamma(double shape);

    /// Chi-squared with (possibly non-integer) df: 2 * Gamma(df/2).
    [[nodiscard]] double chi_squared(double df);

private:
    std::mt19937_64 engine_;
};

/// Result of a Monte Carlo tail-risk estimation.
struct MonteCarloResult {
    double var = 0.0;       ///< VaR, positive for a loss (linear-interp quantile).
    double es = 0.0;        ///< Expected Shortfall, positive for a loss.
    double var_se = 0.0;    ///< Order-statistic standard error of the VaR estimate.
    std::size_t n_paths = 0;
};

/// Simulate `n_paths` factor-return scenarios ((n_paths x n) panel) with
/// target covariance `cov`.  dist = StudentT requires df > 2.  Bitwise
/// deterministic in `seed`.  Throws std::invalid_argument on bad inputs.
[[nodiscard]] Matrix simulate_factor_returns(const Matrix& cov, std::size_t n_paths,
                                             Dist dist = Dist::Normal, double df = 6.0,
                                             std::uint64_t seed = 0);

/// VaR + ES + order-statistic SE from a scenario P&L sample (>= 100 paths).
///
/// SE uses the asymptotic quantile variance alpha(1-alpha)/(n f(q)^2) with
/// the density f estimated by a symmetric order-statistic finite difference
/// of bandwidth ceil(sqrt(alpha n)) around the quantile rank.
[[nodiscard]] MonteCarloResult mc_tail_metrics(std::span<const double> pnl, double alpha);

/// Full Monte Carlo VaR + ES on a linear portfolio.  Throws
/// std::invalid_argument on empty exposures, shape mismatch, alpha outside
/// (0, 0.5), or df <= 2 for the Student-t model.  Bitwise deterministic in
/// `seed`.
[[nodiscard]] MonteCarloResult monte_carlo_var(std::span<const double> exposures,
                                               const Matrix& cov, double alpha = 0.01,
                                               std::size_t n_paths = 100'000,
                                               Dist dist = Dist::Normal, double df = 6.0,
                                               std::uint64_t seed = 0);

}  // namespace eqvar
