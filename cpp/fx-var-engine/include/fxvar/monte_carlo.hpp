// Monte Carlo VaR: multivariate normal / Student-t / jump-mixture factors.
//
// Factor returns are simulated from a daily covariance (Cholesky with a
// capped jitter ladder and optional PSD repair; two perfectly correlated
// pegged currencies are a legitimate FX input) and the book is revalued
// scenario by scenario through CompiledBook: spot legs as S exp(r) on log
// returns.
//
// Distributions
//   * kNormal: MVN(0, Sigma h).
//   * kStudentT: multivariate Student-t scaled to match Sigma
//     (X = Z sqrt((df-2)/df) / sqrt(W/df)), so the comparison with normal
//     MC is at equal covariance.
//   * kJump: normal diffusion plus N ~ Poisson(lam h) common jumps,
//     lam = -ln(1 - p) for a per-day jump probability p, each jump moving
//     the listed factors by N(mu_J, sigma_J). Uncompensated: the jumps add
//     variance and a drift on top of Sigma.
//
// Determinism
// All randomness flows through fxvar::Rng: mt19937_64 driving inverse-CDF
// transforms implemented in this library (never std::normal_distribution).
// A fixed seed gives the same scenarios on one platform. Across platforms
// results can differ in the last bits, because std::log, std::exp and
// std::erfc are not correctly rounded and differ between libms.
//
// Standard error
// monte_carlo_var simulates once and reports the bootstrap SE of the VaR
// (primary) and the Silverman-KDE asymptotic SE at the reported quantile
// (diagnostic; biased low in deep tails, docs/VALIDATION.md).

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
  /// Poisson(mu) count, finite mu >= 0. Inversion by the pmf recursion for
  /// mu <= 30; larger means add independent Poisson(30) blocks (exact by
  /// additivity), so exp(-mu) never underflows.
  long poisson(double mu);

 private:
  std::mt19937_64 engine_;
};

enum class McDist { kNormal, kStudentT, kJump };

/// Common-jump overlay for McDist::kJump. `prob` is the per-DAY jump
/// probability p in [0, 1). It becomes an intensity lam = -ln(1 - p), and the
/// number of jumps over the horizon h is N ~ Poisson(lam h). Each jump moves
/// every listed factor f by mean[f] + stdev[f] * Z (log-return units;
/// unlisted factors do not jump), so the jump return is the sum of N iid
/// jump sizes. The jumps are not compensated: with mean != 0 they add a
/// drift of lam h mean[f] to factor f.
struct JumpSpec {
  double prob = 0.0;
  std::map<std::string, double> mean;
  std::map<std::string, double> stdev;
};

struct MonteCarloOptions {
  double alpha = 0.01;  ///< tail probability (0.01 = 99 % VaR)
  double horizon_days = 1.0;
  std::size_t n_scenarios = 50000;
  McDist dist = McDist::kNormal;
  double df = 6.0;  ///< Student-t dof (must be > 2)
  JumpSpec jumps;   ///< used when dist == kJump
  std::uint64_t seed = 0;
  std::size_t n_boot = 200;       ///< bootstrap replicates for se_var (>= 2)
  Repair repair = Repair::kNone;  ///< PSD repair for an indefinite cov
};

/// Monte Carlo VaR result (positive base-ccy losses). VaR, ES and both
/// standard errors come from the same simulated scenarios. `se_var` is the
/// bootstrap SE (primary); `se_var_kde` the Silverman KDE value at the
/// reported quantile (diagnostic). `cholesky_warning` is non-empty when the
/// covariance needed jitter or repair.
struct MonteCarloResult {
  double var = 0.0;
  double es = 0.0;
  double alpha = 0.01;  ///< tail probability (0.01 = 99 % VaR)
  double horizon_days = 1.0;
  McDist dist = McDist::kNormal;
  std::size_t n_scenarios = 0;
  double se_var = 0.0;      ///< bootstrap SE of the VaR (primary)
  double se_var_kde = 0.0;  ///< KDE asymptotic SE (diagnostic)
  std::vector<double> pnl;
  std::string cholesky_warning;
};

/// Simulate factor-return scenarios from a daily covariance.  `cov` is
/// scaled by `horizon_days` (i.i.d. aggregation); `diag`, if non-null,
/// receives the Cholesky diagnostics. Throws on invalid inputs
/// (n_scenarios < 1, t df <= 2 or non-finite, jump prob outside [0, 1),
/// non-finite jump mean, negative or non-finite jump stdev).
ReturnsMatrix simulate_factor_returns(const FactorCov& cov, std::size_t n_scenarios,
                                      McDist dist = McDist::kNormal, double df = 6.0,
                                      const JumpSpec& jumps = {}, std::uint64_t seed = 0,
                                      double horizon_days = 1.0, CholeskyResult* diag = nullptr,
                                      Repair repair = Repair::kNone);

/// Diagnostic asymptotic standard error of the empirical VaR:
/// SE = sqrt(alpha (1-alpha) / n) / f_hat(q), f_hat a Gaussian KDE with
/// Silverman bandwidth 0.9 min(sd, IQR/1.34) n^(-1/5) evaluated at the
/// reported order-statistic quantile q. Biased low in deep tails; the
/// bootstrap below is the primary SE. Requires at least 10 scenarios.
double var_standard_error(const std::vector<double>& pnl, double alpha);

/// Bootstrap standard error of the empirical VaR (primary SE): resample
/// `pnl` with replacement n_boot times, apply empirical_var (order statistic,
/// std::nth_element) to each resample, return the ddof = 1 standard
/// deviation. Memory is one resample buffer, never an n_boot x n index
/// matrix. Requires at least 10 finite scenarios and n_boot >= 2.
double var_standard_error_bootstrap(const std::vector<double>& pnl, double alpha,
                                    std::size_t n_boot = 500, std::uint64_t seed = 0);

/// Monte Carlo VaR/ES with full revaluation of the book.  `cov` must
/// cover every factor in book.factors() (extra factors are ignored).  An
/// empty book throws; a factorless (pure base-ccy cash) book reports
/// exactly zero.
MonteCarloResult monte_carlo_var(const Book& book, const Market& market, const FactorCov& cov,
                                 const MonteCarloOptions& opts = {});

}  // namespace fxvar
