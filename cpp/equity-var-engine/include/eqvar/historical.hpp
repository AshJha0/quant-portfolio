// eqvar/historical.hpp — historical-simulation VaR: plain, age-weighted (BRW)
// and filtered (FHS), plus square-root-of-time horizon scaling.
//
// Conventions (identical to the Python reference eq_var.historical_var):
//   * `pnl` arrays are P&L in currency units, loss < 0;
//   * `alpha` is the tail probability: alpha = 0.01 -> 99 % VaR;
//   * VaR is reported as a POSITIVE number for a loss: VaR = -Q_alpha(pnl);
//   * plain historical VaR uses NumPy's default "linear" (Hyndman-Fan
//     type-7) interpolation between order statistics:
//       h = (n-1) q,  Q = x_(floor h) + (h - floor h)(x_(floor h + 1) - x_(floor h))
//     This is the most common desk choice; lower/higher order-statistic
//     alternatives differ by O(1/n) and are documented in docs/METHODOLOGY.md;
//   * weighted quantiles (BRW) use the step-function inversion of the
//     weighted empirical CDF.
//
// Invalid inputs throw std::invalid_argument.

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace eqvar {

/// Fewer observations than this cannot resolve a 1-5 % tail sensibly
/// (mirrors eq_var.historical_var.MIN_OBS).
inline constexpr std::size_t kMinHistObs = 50;

/// Linear-interpolation (type-7, numpy default) quantile of an UNSORTED
/// sample, q in [0, 1].  Exposed separately so it can be hand-tested on tiny
/// arrays without the min-obs guard.
[[nodiscard]] double quantile_linear(std::span<const double> x, double q);

/// Plain historical-simulation VaR (equal weights):
/// -quantile_alpha(pnl) with linear (type-7) interpolation.
/// Requires at least kMinHistObs finite observations.
[[nodiscard]] double historical_var(std::span<const double> pnl, double alpha = 0.01);

/// Boudoukh-Richardson-Whitelaw exponential age weights.  Observation i
/// (0 = oldest, n-1 = most recent) gets weight (1-lam) lam^{n-1-i} / (1-lam^n);
/// weights sum to 1 exactly and increase with recency.
[[nodiscard]] std::vector<double> brw_weights(std::size_t n, double lam = 0.98);

/// Age-weighted (BRW) historical VaR: invert the weighted empirical CDF at
/// alpha — VaR is minus the smallest P&L whose cumulative weight (ascending
/// P&L order, stable sort) reaches alpha.  lam -> 1 recovers plain historical
/// simulation up to the interpolation-scheme difference.
[[nodiscard]] double age_weighted_var(std::span<const double> pnl, double alpha = 0.01,
                                      double lam = 0.98);

/// One-step-ahead EWMA (RiskMetrics) volatility forecasts:
/// sigma2[t] = lam sigma2[t-1] + (1-lam) x[t-1]^2, seeded with the
/// full-sample population (ddof = 0) variance — mirrors
/// eq_var.historical_var.ewma_volatility with init="sample".  sigma[t] is the
/// forecast for day t using information up to t-1 (no look-ahead).
/// Zero-variance series are floored to avoid downstream division by zero.
[[nodiscard]] std::vector<double> ewma_volatility(std::span<const double> x, double lam = 0.94);

/// Filtered historical simulation (FHS) VaR — devolatilise with EWMA vol,
/// rescale the standardised innovations to tomorrow's vol forecast
/// sigma_{T+1}^2 = lam sigma_T^2 + (1-lam) pnl_T^2, then take the empirical
/// alpha quantile (linear interpolation).  Responds to the current vol regime
/// while keeping the empirical tail shape.
[[nodiscard]] double filtered_historical_var(std::span<const double> pnl, double alpha = 0.01,
                                             double lam = 0.94);

/// Square-root-of-time scaling: VaR_h = VaR_1 * sqrt(h).  Valid only for
/// i.i.d. zero-drift returns; understates multi-day risk under volatility
/// clustering (docs/VALIDATION.md).  horizon_days must be >= 1.
[[nodiscard]] double scale_var_sqrt_time(double var_1d, int horizon_days);

}  // namespace eqvar
