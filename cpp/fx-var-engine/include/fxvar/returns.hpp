// Factor-return histories: validation, covariance estimators, peg screen.
//
// Factor conventions (mirrors python/fx/03-var-es-engine fx_var.common):
//   * "FX:CCY" columns are daily *log returns* of CCYUSD (USD price of one
//     unit of CCY).  USD itself has no FX factor.
//   * "IR:CCY" columns are *absolute* daily changes (decimal p.a.) of the
//     continuously compounded ACT/365 zero rate.
//
// NaN policy: the engine refuses NaNs outright (std::invalid_argument)
// rather than silently dropping or filling - missing FX fixings must be
// handled upstream (holiday calendars differ across time zones).

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "fxvar/matrix.hpp"

namespace fxvar {

/// Daily log-return standard deviation below which an FX factor is treated
/// as "peg-like" (~0.8% annualised - an order of magnitude below any free
/// float; HKD inside its band realises roughly this level).
inline constexpr double kPegVolThreshold = 5e-4;

inline constexpr int kTradingDaysPerYear = 252;

/// A labelled factor-return history: rows = days, columns = factors.
struct ReturnsMatrix {
  std::vector<std::string> factors;  ///< column names, e.g. "FX:EUR"
  Matrix data;                       ///< n_obs x n_factors

  std::size_t n_obs() const { return data.rows(); }
  std::size_t n_factors() const { return factors.size(); }

  /// Column index of `factor`, or -1 if absent.
  int column_index(const std::string& factor) const;

  /// Restrict to (and reorder as) `wanted`; throws std::invalid_argument
  /// listing any missing factor columns.
  ReturnsMatrix select(const std::vector<std::string>& wanted) const;
};

/// A labelled covariance matrix over named factors.
struct FactorCov {
  std::vector<std::string> factors;
  Matrix cov;  ///< n_factors x n_factors, daily units

  /// Restrict to (and reorder as) `wanted`; throws on missing factors.
  FactorCov select(const std::vector<std::string>& wanted) const;
};

/// Validate a history for use by any VaR method: consistent shape, at
/// least `min_obs` rows, every `required` factor present, no NaNs.
/// Throws std::invalid_argument with an informative message otherwise.
void validate_returns(const ReturnsMatrix& returns,
                      const std::vector<std::string>& required,
                      std::size_t min_obs = 60);

/// Unbiased (ddof=1) sample covariance of the daily factor returns.
FactorCov sample_cov(const ReturnsMatrix& returns);

/// RiskMetrics EWMA covariance forecast after the last observation:
/// S_t = lam S_{t-1} + (1-lam) r_{t-1} r_{t-1}', seeded with the sample
/// covariance.  Throws unless 0 < lam < 1.
FactorCov ewma_cov(const ReturnsMatrix& returns, double lam = 0.94);

/// RiskMetrics EWMA per-factor volatility.
/// sigma2_t = lam sigma2_{t-1} + (1-lam) r2_{t-1}: sigma_t is the forecast
/// for day t using data through t-1, seeded with the full-sample variance.
struct EwmaVolatility {
  Matrix sigma;                    ///< per-day forecast vols (n_obs rows)
  std::vector<double> sigma_next;  ///< one-step-ahead forecast per factor
};
EwmaVolatility ewma_volatility(const ReturnsMatrix& returns, double lam = 0.94);

/// Screen FX:* factors for near-zero realised vol (pegged/managed ccys).
/// Returns the flagged factor names (empty if none).  Only FX factors are
/// screened - rate factors are legitimately quiet.  Historical and
/// parametric VaR are blind to peg-break risk; the caller must surface the
/// returned list as a warning (see historical.hpp / parametric.hpp) and
/// add the peg-break stress add-on from stress.hpp.
std::vector<std::string> flag_peg_factors(const ReturnsMatrix& returns,
                                          double threshold = kPegVolThreshold);

}  // namespace fxvar
