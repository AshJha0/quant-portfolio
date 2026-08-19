// Monte Carlo pricing of FX options under the domestic risk-neutral measure.
//
// Under the domestic money-market numeraire the spot follows
//   dS/S = (r_d - r_f) dt + sigma dW^d,
// so terminal spot is sampled exactly:
//   S_T = S exp((r_d - r_f - sigma^2/2) T + sigma sqrt(T) Z).
//
// Variance reduction:
//   * Antithetic variates -- pairs (Z, -Z), estimator averaged per pair.
//   * Control variate -- the discounted terminal spot e^{-r_d T} S_T has
//     known mean S e^{-r_f T} (the value today of one unit of foreign
//     currency delivered at T); the optimal coefficient is estimated from
//     the sample covariance.
//
// Determinism: every run takes an explicit std::mt19937_64 seed, draws
// normals by inverse-CDF transform of the raw generator output (no
// implementation-defined std::normal_distribution), and is single-threaded
// -- identical seeds give bit-identical results on any platform.

#pragma once

#include <cstdint>
#include <string>

#include "fxopt/common.hpp"

namespace fxopt {

/// Monte Carlo estimate with sampling-error diagnostics.
struct MCResult {
    double price;      ///< point estimate, domestic ccy per foreign notional
    double std_error;  ///< standard error of the estimate
    double ci_low;     ///< 95% CI lower bound (price - 1.96 SE)
    double ci_high;    ///< 95% CI upper bound (price + 1.96 SE)
    std::int64_t n_paths;  ///< underlying draws (antithetic pairs count as 2)
    std::string method;    ///< variance-reduction techniques applied
};

/// Monte Carlo GK price of a European FX vanilla.  Requires T > 0 and
/// n_paths >= 2 (throws otherwise).  n_paths is rounded up to even when
/// antithetic.  The RNG is std::mt19937_64 with the given seed;
/// single-threaded and fully deterministic.
///
/// The standard error is computed from *independent* samples: the pair
/// averages when antithetic (mirrored draws are perfectly dependent), the
/// raw payoffs otherwise.  Fewer than two independent samples (i.e.
/// n_paths <= 2 with antithetic on) leaves the SE unestimable; it is
/// reported as 0.0 and the CI collapses to the point estimate, so use at
/// least 4 antithetic paths if the error bar matters.
MCResult mc_price(double S, double K, double T, double r_d, double r_f,
                  double sigma, OptionType type,
                  std::int64_t n_paths = 100'000, std::uint64_t seed = 0,
                  bool antithetic = true, bool control_variate = true);

}  // namespace fxopt
