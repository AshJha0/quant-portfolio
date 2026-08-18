// eqvar/stats.hpp — special functions and moment statistics.
//
// Self-contained (no external deps) implementations of the distribution
// machinery the VaR engine needs:
//
//   * normal pdf / cdf             — erfc-based, accurate to ~1e-15;
//   * inverse normal cdf           — Acklam rational approximation refined by
//                                    one Halley step (|abs err| ~1e-13,
//                                    unit-tested against known quantiles);
//   * regularized incomplete beta  — continued fraction (modified Lentz);
//                                    powers the Student-t CDF and the exact
//                                    binomial CDF for the Basel zones;
//   * Student-t pdf/cdf/quantile   — quantile by bisection on the CDF;
//   * regularized incomplete gamma — series + continued fraction; powers the
//                                    chi-squared survival function used by
//                                    the backtest LR tests;
//   * mean / std / skew / kurtosis — sample moments (std with ddof = 1,
//                                    skew/kurtosis are the biased moment
//                                    ratios, matching scipy bias=True).
//
// All functions are pure; invalid inputs throw std::invalid_argument.

#pragma once

#include <span>

namespace eqvar {

// --------------------------------------------------------------------------
// Normal distribution
// --------------------------------------------------------------------------

/// Standard normal density phi(x).
[[nodiscard]] double normal_pdf(double x) noexcept;

/// Standard normal CDF Phi(x) = 0.5 * erfc(-x / sqrt(2)).
[[nodiscard]] double normal_cdf(double x) noexcept;

/// Inverse standard normal CDF Phi^{-1}(p), p in (0, 1).
///
/// Acklam's rational approximation (|rel err| < 1.15e-9) followed by one
/// Halley refinement step using the erfc-based CDF, giving ~1e-13 absolute
/// accuracy across the full domain.  Throws std::invalid_argument outside
/// (0, 1).
[[nodiscard]] double normal_ppf(double p);

// --------------------------------------------------------------------------
// Incomplete beta / gamma (regularized) and derived CDFs
// --------------------------------------------------------------------------

/// Regularized incomplete beta I_x(a, b), a, b > 0, x in [0, 1].
/// Continued-fraction (modified Lentz) evaluation with the symmetry
/// I_x(a,b) = 1 - I_{1-x}(b,a) for fast convergence.
[[nodiscard]] double betainc_reg(double a, double b, double x);

/// Regularized lower incomplete gamma P(a, x), a > 0, x >= 0.
/// Series for x < a + 1, continued fraction otherwise.
[[nodiscard]] double regularized_gamma_p(double a, double x);

/// Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x).
[[nodiscard]] double regularized_gamma_q(double a, double x);

/// Chi-squared survival function P(X > x) for df degrees of freedom:
/// Q(df/2, x/2).  Used for the Kupiec / Christoffersen LR p-values.
[[nodiscard]] double chi2_sf(double x, double df);

/// Exact Binomial(n, p) CDF P(X <= k) via the incomplete beta identity
/// P(X <= k) = I_{1-p}(n - k, k + 1).  k < 0 returns 0, k >= n returns 1.
/// Used for the Basel traffic-light zone probabilities.
[[nodiscard]] double binomial_cdf(int k, int n, double p);

// --------------------------------------------------------------------------
// Student-t distribution
// --------------------------------------------------------------------------

/// Student-t density with df degrees of freedom.
[[nodiscard]] double student_t_pdf(double x, double df);

/// Student-t CDF via the regularized incomplete beta:
/// F(x) = 1 - I_{df/(df+x^2)}(df/2, 1/2) / 2 for x >= 0 (symmetric below).
[[nodiscard]] double student_t_cdf(double x, double df);

/// Student-t quantile by bisection on the CDF (machine-precision bracket;
/// matches scipy.stats.t.ppf to better than 1e-9 in the tested range).
[[nodiscard]] double student_t_ppf(double p, double df);

// --------------------------------------------------------------------------
// Sample moments
// --------------------------------------------------------------------------

/// Arithmetic mean.  Throws on empty input.
[[nodiscard]] double mean(std::span<const double> x);

/// Sample standard deviation with ddof = 1.  Throws if size < 2.
[[nodiscard]] double stdev(std::span<const double> x);

/// Sample skewness m3 / m2^{3/2} (biased moment ratio, scipy bias=True).
/// Throws if size < 3; returns 0 for zero-variance input.
[[nodiscard]] double skewness(std::span<const double> x);

/// Sample *excess* kurtosis m4 / m2^2 - 3 (biased, scipy bias=True).
/// Throws if size < 4; returns 0 for zero-variance input.
[[nodiscard]] double excess_kurtosis(std::span<const double> x);

}  // namespace eqvar
