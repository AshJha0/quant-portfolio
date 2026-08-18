// eqvar/expected_shortfall.hpp — Expected Shortfall (ES / CVaR).
//
// ES_alpha = -(1/alpha) * integral_0^alpha Q_u(pnl) du — the average loss in
// the worst alpha tail.  ES >= VaR by construction, ES is coherent
// (subadditive), and FRTB replaced 99 % VaR with 97.5 % ES as the
// market-risk capital measure.  Mirrors eq_var.expected_shortfall.
//
// Conventions: alpha = tail probability; ES positive for losses.

#pragma once

#include <span>

namespace eqvar {

/// Empirical Expected Shortfall — exact tail integral of the step CDF.
///
/// With sorted P&L x_(1) <= ... <= x_(n) and k = floor(alpha n):
///   ES = -(1/(alpha n)) [ sum_{i<=k} x_(i) + (alpha n - k) x_(k+1) ]
/// i.e. the exact integral of the empirical quantile function over
/// (0, alpha] with a fractional weight on the boundary order statistic.
/// Consistent, satisfies ES >= VaR, exact on known arrays (unit tested).
/// Requires at least 10 finite observations.
[[nodiscard]] double expected_shortfall(std::span<const double> pnl, double alpha = 0.01);

/// Closed-form ES for normal P&L: ES = sigma phi(z_alpha)/alpha - mean,
/// z = Phi^{-1}(alpha).  Unit-tested against numerical integration to 1e-10.
[[nodiscard]] double normal_es(double sigma, double alpha = 0.01, double mean = 0.0);

/// Closed-form ES for variance-matched Student-t P&L (df > 2):
///   ES = sigma f_nu(q) (nu + q^2) / ((nu-1) alpha) sqrt((nu-2)/nu) - mean
/// with q = t_nu^{-1}(alpha).
[[nodiscard]] double student_t_es(double sigma, double alpha = 0.01, double df = 6.0,
                                  double mean = 0.0);

}  // namespace eqvar
