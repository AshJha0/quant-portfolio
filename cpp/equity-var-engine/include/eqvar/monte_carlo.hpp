// eqvar/monte_carlo.hpp: Monte Carlo VaR for a linear (delta) portfolio.
//
// Simulate factor returns from a multivariate normal or multivariate
// Student-t (common mixing variable Z / sqrt(W/df), scale matrix
// cov*(df-2)/df so the covariance matches cov while the tails fatten),
// revalue the linear portfolio P&L = w . r, and read VaR / ES off the
// scenario distribution with the order statistic. The VaR standard error is
// a bootstrap (primary) plus a KDE value (diagnostic).
//
// Determinism: every draw comes from a seeded std::mt19937_64 through an
// inverse-CDF normal transform (normal_ppf on 53-bit uniforms) and a
// Marsaglia-Tsang gamma sampler on the same primitives, never
// std::normal_distribution. Results repeat exactly for a given seed on one
// platform. Across platforms they can differ in the last bits, because
// std::log, std::exp and std::erfc are not correctly rounded and differ
// between glibc, MSVC and Apple libm.
//
// RNG streams differ from NumPy, so agreement with the Python engine is
// statistical (within Monte Carlo error), not exact.

#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>

#include "eqvar/historical.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/parametric.hpp"

namespace eqvar {

/// Deterministic random stream: mt19937_64 + inverse-CDF transforms only.
class RandomStream {
public:
    explicit RandomStream(std::uint64_t seed) : engine_(seed) {}

    /// 53-bit uniform on the open interval (0, 1).
    [[nodiscard]] double uniform();

    /// Standard normal via Phi^{-1}(uniform()): one draw per variate,
    /// platform-independent (unlike std::normal_distribution).
    [[nodiscard]] double gaussian();

    /// Gamma(shape, scale = 1) via Marsaglia-Tsang (2000), with the
    /// U^{1/a} boost for shape < 1.
    [[nodiscard]] double gamma(double shape);

    /// Chi-squared with (possibly non-integer) df: 2 * Gamma(df/2).
    [[nodiscard]] double chi_squared(double df);

    /// Non-throwing gamma for the noexcept kernels: NaN for a shape that is
    /// NaN, infinite or <= 0, without consuming the stream (never loops,
    /// never throws). Valid shapes give the same draw as gamma().
    [[nodiscard]] double gamma_nothrow(double shape) noexcept;

    /// Non-throwing chi-squared: 2 * gamma_nothrow(df / 2); NaN for bad df.
    [[nodiscard]] double chi_squared_nothrow(double df) noexcept;

private:
    std::mt19937_64 engine_;
};

/// Result of a Monte Carlo tail-risk estimation.
struct MonteCarloResult {
    double var = 0.0;     ///< VaR, positive for a loss (order statistic L(m)).
    double es = 0.0;      ///< Expected Shortfall, positive for a loss.
    double var_se = 0.0;  ///< Bootstrap standard error of the VaR (primary).
    double var_se_kde =
        0.0;  ///< Silverman-KDE asymptotic SE at the reported quantile (diagnostic).
    std::size_t n_paths = 0;
};

/// Simulate `n_paths` factor-return scenarios ((n_paths x n) panel) with
/// target covariance `cov`. dist = StudentT requires finite df > 2. The
/// factorisation is cholesky(cov, repair). Deterministic in `seed` for a given
/// platform and libm. Throws std::invalid_argument on bad inputs.
[[nodiscard]] Matrix simulate_factor_returns(const Matrix& cov, std::size_t n_paths,
                                             Dist dist = Dist::Normal, double df = 6.0,
                                             std::uint64_t seed = 0, Repair repair = Repair::None);

/// VaR + ES (order statistic, see historical.hpp) + bootstrap SE (primary,
/// mc_bootstrap_se with n_boot >= 2) + KDE SE (diagnostic) from one scenario
/// P&L sample of at least 100 finite paths.
[[nodiscard]] MonteCarloResult mc_tail_metrics(std::span<const double> pnl, double alpha,
                                               std::size_t n_boot = 200, std::uint64_t seed = 0);

/// Asymptotic quantile SE sqrt(alpha(1-alpha)/n) / f(q) with f a Gaussian KDE
/// (Silverman bandwidth 0.9 min(sd, IQR/1.34) n^{-1/5}) evaluated at the
/// reported order-statistic quantile q. Diagnostic only. Returns 0 for a
/// zero-variance sample.
[[nodiscard]] double kde_var_se(std::span<const double> pnl, double alpha);

/// Bootstrap standard error of the order-statistic VaR: resample `pnl` with
/// replacement n_boot times, recompute L(m) on each resample with
/// std::nth_element, return the ddof = 1 standard deviation. Memory is one
/// resample buffer (never an n_boot x n index matrix). Throws
/// std::invalid_argument for n < 10, n_boot < 2, non-finite P&L or alpha
/// outside (0, 0.5).
[[nodiscard]] double mc_bootstrap_se(std::span<const double> pnl, double alpha,
                                     std::size_t n_boot = 500, std::uint64_t seed = 0);

/// Full Monte Carlo VaR + ES on a linear portfolio. Simulates once; VaR, ES
/// and both SEs come from the same scenarios. Throws std::invalid_argument on
/// an empty or all-zero book, shape mismatch, alpha outside (0, 0.5), or
/// df <= 2 for the Student-t model.
[[nodiscard]] MonteCarloResult monte_carlo_var(std::span<const double> exposures, const Matrix& cov,
                                               double alpha = 0.01, std::size_t n_paths = 100'000,
                                               Dist dist = Dist::Normal, double df = 6.0,
                                               std::uint64_t seed = 0, std::size_t n_boot = 200,
                                               Repair repair = Repair::None);

}  // namespace eqvar
