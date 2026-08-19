// Parametric (variance-covariance) VaR: normal, Student-t, Cornish-Fisher.
//
// The book is linearised into factor exposures w (finite-difference deltas
// from CompiledBook::linear_exposures - forwards enter via their deposit
// legs), and the portfolio P&L variance is w' Sigma w with Sigma a sample
// or EWMA covariance of daily factor returns (mirrors
// fx_var.parametric_var).
//
// Distributional overlays on the same sigma:
//   * normal         - RiskMetrics classic; underestimates FX tails.
//   * Student-t      - standardised (unit-variance) t captures the fat
//                      tails of EM currency returns at equal sigma.
//   * Cornish-Fisher - moment-corrected quantile from the portfolio's
//                      empirical skew/kurtosis; only valid inside the
//                      monotonicity domain of the expansion, which is
//                      checked explicitly (Maillard 2012 - outside the
//                      domain the "quantile" is not a quantile).
//
// Multi-day horizon: sigma scales by sqrt(h) (i.i.d. assumption).

#pragma once

#include <string>
#include <vector>

#include "fxvar/book.hpp"
#include "fxvar/matrix.hpp"
#include "fxvar/returns.hpp"

namespace fxvar {

enum class TailDist { kNormal, kStudentT };
enum class CovMethod { kSample, kEwma };

/// (VaR, ES) pair, positive losses.
struct VarEs {
  double var = 0.0;
  double es = 0.0;
};

/// 1-day portfolio P&L standard deviation sqrt(w' Sigma w).
///
/// Throws std::invalid_argument on non-finite exposures or covariance, or
/// if the quadratic form is materially negative (covariance not PSD).
/// "Materially" is measured RELATIVE to |w|^2 max|Sigma|: a hedged book
/// against a rank-deficient covariance legitimately rounds to a tiny
/// negative w'Sigma w, which is clamped to zero rather than rejected.
double portfolio_sigma(const std::vector<double>& exposures, const Matrix& cov);

/// Closed-form (VaR, ES) for a linear book: pure function for testing.
/// `exposures` are base-ccy P&L per unit factor move; `cov` is the daily
/// factor covariance in the same order; `df` is used for kStudentT;
/// `mean` is the expected 1-day P&L (usually 0 at daily horizon).
VarEs var_covar(const std::vector<double>& exposures, const Matrix& cov,
                double alpha = 0.99, double horizon_days = 1.0,
                TailDist dist = TailDist::kNormal, double df = 6.0,
                double mean = 0.0);

struct ParametricOptions {
  double alpha = 0.99;
  double horizon_days = 1.0;
  TailDist dist = TailDist::kNormal;
  double df = 6.0;
  CovMethod cov_method = CovMethod::kSample;
  double ewma_lambda = 0.94;
  std::size_t min_obs = 60;
  bool warn_pegs = true;
};

/// Variance-covariance VaR result (positive base-ccy losses).
struct ParametricResult {
  double var = 0.0;
  double es = 0.0;
  double alpha = 0.99;
  double horizon_days = 1.0;
  TailDist dist = TailDist::kNormal;
  double sigma = 0.0;  ///< 1-day portfolio P&L std in base ccy
  std::vector<std::string> factors;
  std::vector<double> exposures;  ///< aligned with `factors`
  std::vector<std::string> flagged_peg_factors;
  std::vector<std::string> warnings;
};

/// Variance-covariance VaR/ES of `book` from a factor-return history.
/// An empty book throws; a factorless (pure base-ccy cash) book reports
/// exactly zero.
ParametricResult parametric_var(const Book& book, const Market& market,
                                const ReturnsMatrix& returns,
                                const ParametricOptions& opts = {});

/// Cornish-Fisher adjusted quantile
/// z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36.
double cornish_fisher_z(double z, double skew, double excess_kurtosis);

/// True if the CF expansion is monotone increasing on [-z_range, z_range]
/// (checked on a dense grid) - the validity condition for the expansion to
/// define a quantile function (Maillard 2012).
bool cornish_fisher_domain_ok(double skew, double excess_kurtosis,
                              double z_range = 4.0, int n_grid = 801);

/// Cornish-Fisher VaR (positive loss) with an explicit domain check:
/// throws std::invalid_argument when (S, K) are outside the monotonicity
/// domain and `check_domain` is true - a silently non-monotone CF
/// "quantile" can report 99% VaR below 95% VaR.
double cornish_fisher_var(double sigma, double skew, double excess_kurtosis,
                          double alpha = 0.99, double mean = 0.0,
                          double horizon_days = 1.0, bool check_domain = true);

}  // namespace fxvar
