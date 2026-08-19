// eqvar/parametric.hpp — variance-covariance VaR.
//
// Portfolio sigma from dollar exposures w and factor-return covariance Sigma:
// sigma_p = sqrt(w' Sigma w).  Quantiles from the normal, the variance-matched
// Student-t, or the Cornish-Fisher expansion with an explicit
// validity-domain check.  Mirrors eq_var.parametric_var.
//
// Conventions: alpha = tail probability; VaR positive for losses; daily
// covariance in factor-return units matching the dollar exposures.

#pragma once

#include <span>

#include "eqvar/matrix.hpp"

namespace eqvar {

/// Tail model for parametric VaR / ES and Monte Carlo simulation.
enum class Dist { Normal, StudentT };

/// Portfolio P&L standard deviation sqrt(w' Sigma w) (currency units).
/// Throws std::invalid_argument on empty exposures, shape mismatch, a
/// non-finite exposure or quadratic form (NaN/Inf covariance), or a
/// materially negative quadratic form (non-PSD covariance).  A tiny
/// negative w'Sw (rounding on a rank-deficient but PSD covariance) is
/// clamped to zero rather than rejected.
[[nodiscard]] double portfolio_sigma(std::span<const double> exposures, const Matrix& cov);

/// Variance-covariance VaR (positive for a loss):
///   VaR = -(mu h + z_alpha sigma_p sqrt(h))
/// where z is the normal quantile or the Student-t quantile rescaled to unit
/// variance (* sqrt((df-2)/df)) so sigma is matched and only the tail shape
/// changes.  `mean` is the expected daily P&L (usually 0 at daily horizon);
/// `horizon_days` applies square-root-of-time scaling to sigma and linear
/// scaling to the mean.  df must be > 2 for dist = StudentT.
[[nodiscard]] double parametric_var(std::span<const double> exposures, const Matrix& cov,
                                    double alpha = 0.01, Dist dist = Dist::Normal,
                                    double df = 6.0, double mean = 0.0, int horizon_days = 1);

/// Cornish-Fisher adjusted quantile:
///   z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36
/// with skewness S and EXCESS kurtosis K.  Reduces to z when S = K = 0.
[[nodiscard]] double cornish_fisher_z(double z, double skew, double excess_kurt);

/// True when the CF quantile map is monotone (dz_cf/dz > 0) on
/// [-z_range, z_range] — outside this region the fourth-order expansion is
/// not a quantile function (the implied density goes negative) and CF "VaR"
/// is nonsense.  |z| <= 3.5 covers alpha >= 0.02 %.
/// Throws std::invalid_argument unless z_range > 0, n_grid >= 2 and the
/// moments are finite.
[[nodiscard]] bool cornish_fisher_domain_ok(double skew, double excess_kurt,
                                            double z_range = 3.5, int n_grid = 2001);

/// Cornish-Fisher VaR: -(mean + z_cf(alpha) sigma).  With
/// `check_domain = true` (default) throws std::invalid_argument when
/// (skew, excess_kurt) lie outside the monotonicity region.
[[nodiscard]] double cornish_fisher_var(double sigma, double alpha = 0.01, double skew = 0.0,
                                        double excess_kurt = 0.0, double mean = 0.0,
                                        bool check_domain = true);

}  // namespace eqvar
