// eqvar/historical.hpp: historical-simulation VaR and ES. Plain, age-weighted
// (BRW) and filtered (per-factor FHS), plus square-root-of-time scaling.
//
// Conventions (shared with the Python, Rust and C++ VaR engines):
//   * `pnl` arrays are P&L in currency units, loss < 0;
//   * `alpha` is the tail probability: alpha = 0.01 -> 99 % VaR;
//   * VaR and ES are reported as POSITIVE numbers for a loss;
//   * the empirical estimator is the order statistic of the losses
//     L = -pnl sorted descending, L(1) >= ... >= L(n), k = n alpha,
//     m = ceil(k - 1e-9) clamped to [1, n]:
//       VaR = L(m),  ES = [ sum_{i<m} L(i) + (k - (m-1)) L(m) ] / k
//     (Acerbi-Tasche, exact fractional atom). No interpolation;
//   * weighted samples (BRW) use the same rule on cumulative weights
//     (docs/METHODOLOGY.md).
//
// Invalid inputs throw std::invalid_argument.

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "eqvar/matrix.hpp"

namespace eqvar {

/// Fewer observations than this cannot resolve a 1-5 % tail sensibly.
inline constexpr std::size_t kMinHistObs = 50;

/// Default EWMA seed window: seed variance = mean of r^2 over the first
/// min(30, n) observations.
inline constexpr std::size_t kEwmaSeedWindow = 30;

/// VaR and ES of one sample, both positive for a loss.
struct TailRisk {
    double var = 0.0;
    double es = 0.0;
};

/// Order-statistic VaR and exact fractional ES (see header comment). Uses
/// std::nth_element (O(n)) and a partial sum, no full sort. Requires a
/// non-empty, finite sample; no minimum length.
[[nodiscard]] TailRisk var_es_order_statistic(std::span<const double> pnl, double alpha);

/// Weighted order-statistic VaR / ES. Weights are normalised to sum 1 and
/// sorted with the losses (stable sort, losses descending). W_j is the
/// cumulative weight; m = first j with W_j >= alpha - 1e-12;
/// VaR = L(m), ES = [ sum_{i<m} w_i L(i) + (alpha - W_{m-1}) L(m) ] / alpha.
[[nodiscard]] TailRisk var_es_weighted(std::span<const double> pnl, std::span<const double> weights,
                                       double alpha);

/// Type-7 (numpy "linear") quantile, q in [0, 1]. No VaR path uses it.
[[deprecated(
    "type-7 interpolation is no longer a VaR convention; use "
    "var_es_order_statistic. Removed next release.")]] [[nodiscard]] double
quantile_linear(std::span<const double> x, double q);

/// Plain historical-simulation VaR (order statistic L(m)).
/// Requires at least kMinHistObs finite observations.
[[nodiscard]] double historical_var(std::span<const double> pnl, double alpha = 0.01);

/// Boudoukh-Richardson-Whitelaw exponential age weights. Observation i
/// (0 = oldest, n-1 = most recent) gets weight (1-lam) lam^{n-1-i} / (1-lam^n).
[[nodiscard]] std::vector<double> brw_weights(std::size_t n, double lam = 0.98);

/// Age-weighted (BRW) historical VaR via var_es_weighted.
[[nodiscard]] double age_weighted_var(std::span<const double> pnl, double alpha = 0.01,
                                      double lam = 0.98);

/// Age-weighted (BRW) historical ES via var_es_weighted.
[[nodiscard]] double age_weighted_es(std::span<const double> pnl, double alpha = 0.01,
                                     double lam = 0.98);

/// RiskMetrics zero-mean EWMA volatility:
///   s2_0 = mean(x[0..w)^2), w = min(seed_window, n),
///   s2_t = lam s2_{t-1} + (1-lam) x[t-1]^2.
/// sigma[t] = sqrt(s2_t) is the forecast for day t from data up to t-1. The
/// seed reads only the first w observations, so sigma[t] for t >= w has no
/// look-ahead; inside the seed window it does. Variances are floored at 1e-32.
[[nodiscard]] std::vector<double> ewma_volatility(std::span<const double> x, double lam = 0.94,
                                                  std::size_t seed_window = kEwmaSeedWindow);

/// One-step-ahead EWMA vol: sqrt(lam s2_{n-1} + (1-lam) x[n-1]^2).
[[nodiscard]] double ewma_volatility_next(std::span<const double> x, double lam = 0.94,
                                          std::size_t seed_window = kEwmaSeedWindow);

/// Filtered historical simulation on the linear book. For each factor j:
/// sigma_{t,j} from ewma_volatility, z = r / sigma, r* = z * sigma_{T+1,j};
/// scenario P&L = exposures . r*. Returns VaR and ES by the order statistic.
/// `factor_returns` is T x J (T >= kMinHistObs), `exposures` has J entries.
[[nodiscard]] TailRisk filtered_historical_var_es(const Matrix& factor_returns,
                                                  std::span<const double> exposures,
                                                  double alpha = 0.01, double lam = 0.94);

/// P&L-level FHS (devolatilises the scalar P&L series). It mixes the vol of
/// the book with the vol of its factors; use filtered_historical_var_es.
/// Kept unchanged for one release: full-sample variance seed and type-7
/// quantile, as before.
[[deprecated(
    "P&L-level FHS; use filtered_historical_var_es (per factor). Removed next "
    "release.")]] [[nodiscard]] double
filtered_historical_var(std::span<const double> pnl, double alpha = 0.01, double lam = 0.94);

/// Square-root-of-time scaling: VaR_h = VaR_1 * sqrt(h). Valid only for
/// i.i.d. zero-drift returns. var_1d must be finite, horizon_days >= 1.
[[nodiscard]] double scale_var_sqrt_time(double var_1d, int horizon_days);

}  // namespace eqvar
