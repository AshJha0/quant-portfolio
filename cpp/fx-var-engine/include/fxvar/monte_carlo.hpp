// Monte Carlo VaR: multivariate normal / Student-t / jump-mixture factors.
//
// Factor returns are simulated from a daily covariance (Cholesky with
// jitter escalation - two perfectly correlated pegged currencies are a
// legitimate FX input) and the book is fully revalued scenario by scenario
// through CompiledBook (mirrors fx_var.monte_carlo_var).
//
// Distributions
// -------------
//   * kNormal - MVN(0, Sigma h).
//   * kStudentT - multivariate Student-t scaled to match Sigma exactly
//     (X = Z sqrt((df-2)/df) / sqrt(W/df)), so the comparison with normal
//     MC is at equal covariance: any 99% VaR difference is pure tail
//     shape.  EM currency returns are the textbook case (df 4-6).
//   * kJump - normal diffusion plus a Bernoulli(p) common jump event with
//     per-factor jump sizes ~ N(mu_J, sigma_J): the devaluation /
//     peg-break overlay.  The jump *adds* variance on top of Sigma by
//     design - it models exactly the risk the covariance matrix cannot
//     see.
//
// Determinism
// -----------
// All randomness flows through fxvar::Rng - mt19937_64 driving inverse-CDF
// transforms implemented in this library (never std::normal_distribution,
// whose stream is implementation-defined).  A fixed seed therefore gives
// bitwise-identical scenario sets and VaR figures across runs and across
// standard libraries; tests assert exact (==) reproducibility.
//
// Standard error
// --------------
// var_standard_error uses the asymptotic order-statistic formula
// SE = sqrt(a(1-a)/n) / f(q) with a Gaussian-KDE (Silverman bandwidth)
// density estimate at the quantile.  Convergence tests accept MC vs
// closed form within 3 SE.

#pragma once

#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "fxvar/book.hpp"
#include "fxvar/matrix.hpp"
#include "fxvar/returns.hpp"

namespace fxvar {

/// Deterministic random source: mt19937_64 + inverse-CDF transforms.
/// Every stochastic component takes an explicit seed (portfolio
/// convention); the stream is fully specified by this library.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : engine_(seed) {}

  /// Uniform on (0, 1), 53-bit resolution, never exactly 0 or 1.
  double uniform();

  /// Standard normal via norm_ppf(uniform()) - deterministic, exact
  /// stream reproducibility across platforms and standard libraries
  /// (the mt19937_64 raw stream is fully specified by the C++ standard;
  /// the transform is this library's own).
  double normal();

  /// Gamma(shape, 1) via Marsaglia-Tsang (shape >= 1) with the
  /// U^(1/shape) boost for shape < 1.  Throws for shape <= 0.
  double gamma(double shape);

  /// Chi-square with df > 0 degrees of freedom: 2 * Gamma(df/2, 1).
  double chisq(double df);

 private:
  std::mt19937_64 engine_;
};

enum class McDist { kNormal, kStudentT, kJump };

/// Common-jump overlay for McDist::kJump: with per-scenario probability
/// `prob`, every factor listed in `mean` jumps by mean[f] + std[f] * Z
/// (log-return units for FX factors; factors not listed do not jump).
/// E.g. {"FX:TRY": -0.15} is a 15% (log) devaluation of the lira vs USD.
struct JumpSpec {
  double prob = 0.0;
  std::map<std::string, double> mean;
  std::map<std::string, double> stdev;
};

struct MonteCarloOptions {
  double alpha = 0.99;
  double horizon_days = 1.0;
  std::size_t n_scenarios = 50000;
  McDist dist = McDist::kNormal;
  double df = 6.0;             ///< Student-t dof (must be > 2)
  JumpSpec jumps;              ///< used when dist == kJump
  std::uint64_t seed = 0;
};

/// Monte Carlo VaR result (positive base-ccy losses).  `se_var` is the
/// asymptotic standard error of the VaR estimate; `cholesky_warning` is
/// non-empty when the covariance needed diagonal jitter.
struct MonteCarloResult {
  double var = 0.0;
  double es = 0.0;
  double alpha = 0.99;
  double horizon_days = 1.0;
  McDist dist = McDist::kNormal;
  std::size_t n_scenarios = 0;
  double se_var = 0.0;
  std::vector<double> pnl;
  std::string cholesky_warning;
};

/// Simulate factor-return scenarios from a daily covariance.  `cov` is
/// scaled by `horizon_days` (i.i.d. aggregation); `diag`, if non-null,
/// receives the Cholesky diagnostics.  Throws on invalid inputs
/// (n_scenarios < 1, t df <= 2, jump prob outside [0,1], negative jump
/// stdev).
ReturnsMatrix simulate_factor_returns(const FactorCov& cov,
                                      std::size_t n_scenarios,
                                      McDist dist = McDist::kNormal,
                                      double df = 6.0,
                                      const JumpSpec& jumps = {},
                                      std::uint64_t seed = 0,
                                      double horizon_days = 1.0,
                                      CholeskyResult* diag = nullptr);

/// Asymptotic standard error of the empirical VaR estimate:
/// SE = sqrt(alpha (1-alpha) / n) / f_hat(q), f_hat a Gaussian KDE with
/// Silverman bandwidth evaluated at the loss quantile.  Requires at least
/// 10 scenarios.
double var_standard_error(const std::vector<double>& pnl, double alpha);

/// Monte Carlo VaR/ES with full revaluation of the book.  `cov` must
/// cover every factor in book.factors() (extra factors are ignored).  An
/// empty book throws; a factorless (pure base-ccy cash) book reports
/// exactly zero.
MonteCarloResult monte_carlo_var(const Book& book, const Market& market,
                                 const FactorCov& cov,
                                 const MonteCarloOptions& opts = {});

}  // namespace fxvar
