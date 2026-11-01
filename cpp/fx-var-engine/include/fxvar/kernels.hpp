// noexcept kernel API for the latency-sensitive paths.
//
// Each try_* function computes exactly what its throwing counterpart
// computes (same arithmetic, same order of operations, so the same bits) and
// returns the throwing form's error as a value instead. None of them
// allocates: buffers are caller-provided std::spans, sized once. The
// throwing API is a thin wrapper over these where the two overlap.
//
//   try_empirical_var_es   uniform-weight historical / MC VaR and ES;
//                          O(n) selection (std::nth_element) in `ws`
//   try_var_standard_error KDE standard error of the VaR; the IQR uses two
//                          selections instead of a full sort
//   try_normal_var/es, try_t_var/es   closed-form tails
//   try_matvec, try_quad_form, try_portfolio_sigma, try_var_covar
//   try_cholesky_lower     plain Cholesky into caller storage (no jitter
//                          ladder; use cholesky() when that is needed)
//   try_simulate_factor_returns  normal / Student-t scenarios into caller
//                          storage, same stream as simulate_factor_returns
//   CompiledBook::try_pnl, CompiledBook::try_pnl_batch (book.hpp)

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include "fxvar/error.hpp"
#include "fxvar/matrix.hpp"
#include "fxvar/monte_carlo.hpp"
#include "fxvar/parametric.hpp"

namespace fxvar {

/// noexcept validate_alpha().
[[nodiscard]] inline Result<void> try_validate_alpha(double alpha) noexcept {
  if (!(alpha > 0.0 && alpha < 0.5)) [[unlikely]]
    return std::unexpected(Error{Errc::kAlphaRange, "", alpha});
  return {};
}

/// noexcept validate_horizon().
[[nodiscard]] Result<void> try_validate_horizon(double horizon_days) noexcept;

/// Uniform-weight empirical (VaR, ES) of `pnl`; equals
/// empirical_var_es(pnl, alpha) bit for bit. `ws` needs pnl.size() doubles.
[[nodiscard]] Result<std::pair<double, double>> try_empirical_var_es(
    std::span<const double> pnl, double alpha, std::span<double> ws) noexcept;

/// var_standard_error(pnl, alpha) with caller storage (pnl.size() doubles).
[[nodiscard]] Result<double> try_var_standard_error(std::span<const double> pnl,
                                                    double alpha,
                                                    std::span<double> ws) noexcept;

[[nodiscard]] Result<double> try_normal_var(double sigma, double alpha = 0.01,
                                            double mean = 0.0) noexcept;
[[nodiscard]] Result<double> try_normal_es(double sigma, double alpha = 0.01,
                                           double mean = 0.0) noexcept;
[[nodiscard]] Result<double> try_t_var(double sigma, double alpha = 0.01,
                                       double df = 6.0, double mean = 0.0) noexcept;
[[nodiscard]] Result<double> try_t_es(double sigma, double alpha = 0.01,
                                      double df = 6.0, double mean = 0.0) noexcept;

/// y = A x into caller storage (y.size() == A.rows()).
[[nodiscard]] Result<void> try_matvec(const Matrix& a, std::span<const double> x,
                                      std::span<double> y) noexcept;
[[nodiscard]] Result<double> try_quad_form(std::span<const double> w,
                                           const Matrix& a) noexcept;
[[nodiscard]] Result<double> try_portfolio_sigma(std::span<const double> exposures,
                                                 const Matrix& cov) noexcept;
[[nodiscard]] Result<VarEs> try_var_covar(std::span<const double> exposures,
                                          const Matrix& cov, double alpha = 0.01,
                                          double horizon_days = 1.0,
                                          TailDist dist = TailDist::kNormal,
                                          double df = 6.0, double mean = 0.0) noexcept;

/// Lower Cholesky factor of a square matrix into `lower` (n*n, row-major,
/// upper triangle zeroed). Plain factorisation: an error if a pivot is not
/// positive. Equals cholesky(a).lower whenever cholesky() needed no jitter.
[[nodiscard]] Result<void> try_cholesky_lower(const Matrix& a,
                                              std::span<double> lower) noexcept;

/// Scenarios x = L z (times the Student-t mixing factor for kStudentT) into
/// `out` (n_scenarios * k, row-major), from the same mt19937_64 stream as
/// simulate_factor_returns(cov, n_scenarios, dist, df, {}, seed, h): pass
/// the factor of cov * h. kJump is not supported here (error).
[[nodiscard]] Result<void> try_simulate_factor_returns(
    std::span<const double> lower, std::size_t k, std::size_t n_scenarios,
    McDist dist, double df, std::uint64_t seed, std::span<double> out) noexcept;

}  // namespace fxvar
