// Statistical special functions and sample moments.
//
// Everything the engine needs from a stats library, implemented from
// scratch (no external deps):
//   * standard normal pdf / cdf / inverse cdf (Acklam rational approx,
//     refined with one Halley step: |error| < 1e-13, validated in tests
//     against high-precision reference values),
//   * Student-t pdf / cdf / quantile (regularised incomplete beta +
//     Newton),
//   * regularised incomplete gamma (series + continued fraction) and the
//     chi-square survival function built on it (Kupiec / Christoffersen
//     p-values),
//   * exact binomial CDF (Basel traffic light),
//   * sample moments (mean, std, skewness, excess kurtosis).
//
// Conventions: probabilities in (0, 1); `alpha` is a confidence level
// (0.99 = 99%); all sample statistics use the ddof=1 unbiased variance.

#pragma once

#include <cstddef>
#include <vector>

namespace fxvar {

/// Standard normal density phi(x).
double norm_pdf(double x);

/// Standard normal CDF Phi(x) (via std::erfc, ~1e-16 accurate).
double norm_cdf(double x);

/// Inverse standard normal CDF.
///
/// Acklam's rational approximation (|rel err| < 1.15e-9) followed by one
/// Halley refinement step, giving absolute error below 1e-13 on
/// p in (1e-300, 1 - 1e-16).  Throws std::invalid_argument for p outside
/// (0, 1).
double norm_ppf(double p);

/// Regularised lower incomplete gamma P(a, x) = gamma(a, x) / Gamma(a).
/// Series for x < a + 1, continued fraction otherwise (Numerical Recipes).
double reg_lower_gamma(double a, double x);

/// Regularised upper incomplete gamma Q(a, x) = 1 - P(a, x).
double reg_upper_gamma(double a, double x);

/// Chi-square survival function P(X > x) for df degrees of freedom:
/// Q(df/2, x/2) via the regularised incomplete gamma.
double chi2_sf(double x, double df);

/// Regularised incomplete beta I_x(a, b) (continued fraction, Lentz).
double reg_inc_beta(double a, double b, double x);

/// Student-t density with `df` degrees of freedom.
double t_pdf(double x, double df);

/// Student-t CDF with `df` degrees of freedom (via incomplete beta).
double t_cdf(double x, double df);

/// Student-t quantile: t_cdf(t_ppf(p, df), df) == p.
/// Bracketed Newton iteration from a normal/Cornish start; accurate to
/// ~1e-12.  Throws std::invalid_argument for p outside (0,1) or df <= 0.
double t_ppf(double p, double df);

/// Exact binomial CDF P(X <= k) for X ~ Binomial(n, p), summed in the log
/// domain for stability (n up to a few thousand).
double binom_cdf(int k, int n, double p);

/// Sample moments of a data vector (ddof=1 variance; skewness and kurtosis
/// are the standard biased moment ratios m3/m2^1.5 and m4/m2^2 - 3).
struct Moments {
  double mean = 0.0;
  double stdev = 0.0;            ///< ddof=1 standard deviation
  double skewness = 0.0;         ///< m3 / m2^(3/2) (population-style)
  double excess_kurtosis = 0.0;  ///< m4 / m2^2 - 3
};

/// Compute Moments; throws std::invalid_argument for fewer than 2 points.
Moments sample_moments(const std::vector<double>& x);

/// ddof=1 standard deviation (throws for fewer than 2 points).
double sample_std(const std::vector<double>& x);

/// Validate a VaR/ES confidence level: throws std::invalid_argument unless
/// alpha is strictly inside (0, 1).  Returns alpha unchanged.
double validate_alpha(double alpha);

/// Validate a VaR horizon in trading days (> 0), returning it unchanged.
double validate_horizon(double horizon_days);

}  // namespace fxvar
