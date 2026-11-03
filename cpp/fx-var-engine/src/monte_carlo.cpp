#include "fxvar/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "fxvar/expected_shortfall.hpp"
#include "fxvar/kernels.hpp"
#include "fxvar/stats.hpp"
#include "qp/rng.hpp"

namespace fxvar {

// -------------------------------------------------------------------- Rng
double Rng::uniform() {
  // 53-bit mantissa in (0, 1): (x >> 11) in [0, 2^53), +0.5 keeps the
  // result strictly inside the open interval so norm_ppf never sees 0/1.
  return qp::uniform_open53(engine_());
}

double Rng::normal() { return norm_ppf(uniform()); }

double Rng::gamma(double shape) {
  if (!(shape > 0.0) || !std::isfinite(shape))
    throw std::invalid_argument("gamma: shape must be finite and > 0");
  auto u = [this] { return uniform(); };
  auto z = [this] { return normal(); };
  return qp::gamma_marsaglia_tsang(shape, u, z);
}

double Rng::gamma_nothrow(double shape) noexcept {
  // Shape test first: qp::gamma_marsaglia_tsang recurses for shape <= 0
  // and never terminates for NaN.
  if (!(shape > 0.0) || !std::isfinite(shape)) return std::numeric_limits<double>::quiet_NaN();
  auto u = [this] { return uniform(); };
  auto z = [this] { return normal(); };
  return qp::gamma_marsaglia_tsang(shape, u, z);
}

double Rng::chisq_nothrow(double df) noexcept { return 2.0 * gamma_nothrow(0.5 * df); }

long Rng::poisson(double mu) {
  if (!(mu >= 0.0) || !std::isfinite(mu))
    throw std::invalid_argument("poisson: mean must be finite and >= 0");
  long total = 0;
  while (mu > 30.0) {
    total += poisson(30.0);
    mu -= 30.0;
  }
  if (mu == 0.0) return total;
  const double u = uniform();
  double pmf = std::exp(-mu);
  double cdf = pmf;
  long k = 0;
  while (u > cdf && k < 10000) {
    ++k;
    pmf *= mu / static_cast<double>(k);
    cdf += pmf;
    if (pmf == 0.0) break;  // cdf has converged to 1 in double
  }
  return total + k;
}

double Rng::chisq(double df) {
  if (!(df > 0.0) || !std::isfinite(df))
    throw std::invalid_argument("chisq: df must be finite and > 0");
  return 2.0 * gamma(0.5 * df);
}

// -------------------------------------------------------------- simulate
ReturnsMatrix simulate_factor_returns(const FactorCov& cov, std::size_t n_scenarios, McDist dist,
                                      double df, const JumpSpec& jumps, std::uint64_t seed,
                                      double horizon_days, CholeskyResult* diag, Repair repair) {
  validate_horizon(horizon_days);
  if (n_scenarios < 1) throw std::invalid_argument("n_scenarios must be >= 1");
  const std::size_t k = cov.factors.size();
  if (cov.cov.rows() != k || cov.cov.cols() != k)
    throw std::invalid_argument("simulate_factor_returns: cov labels do not match matrix shape");
  if (dist == McDist::kStudentT && (!(df > 2.0) || !std::isfinite(df)))
    throw std::invalid_argument("Student-t df must be finite and > 2 for finite variance");
  if (dist == McDist::kJump) {
    if (!(jumps.prob >= 0.0 && jumps.prob < 1.0)) {
      std::ostringstream msg;
      msg << "jump prob (per day) must be in [0, 1), got " << jumps.prob;
      throw std::invalid_argument(msg.str());
    }
    for (const auto& [f, m] : jumps.mean)
      if (!std::isfinite(m)) throw std::invalid_argument("jump mean for " + f + " must be finite");
    for (const auto& [f, s] : jumps.stdev)
      if (!(s >= 0.0) || !std::isfinite(s))
        throw std::invalid_argument("jump stdev for " + f + " must be finite and >= 0");
  }

  Matrix a = cov.cov;
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j) a(i, j) *= horizon_days;
  CholeskyResult chol = cholesky(a, repair);
  if (diag) *diag = chol;

  Rng rng(seed);
  ReturnsMatrix out;
  out.factors = cov.factors;
  out.data = Matrix(n_scenarios, k);

  // Pre-resolve jump columns so the per-scenario loop is branch-light.
  std::vector<int> jump_col;
  std::vector<double> jump_mean, jump_std;
  if (dist == McDist::kJump) {
    for (const auto& [f, m] : jumps.mean) {
      int col = -1;
      for (std::size_t j = 0; j < k; ++j)
        if (cov.factors[j] == f) {
          col = static_cast<int>(j);
          break;
        }
      if (col < 0) continue;  // scenario library convention: ignore unknowns
      jump_col.push_back(col);
      jump_mean.push_back(m);
      const auto it = jumps.stdev.find(f);
      jump_std.push_back(it == jumps.stdev.end() ? 0.0 : it->second);
    }
  }

  std::vector<double> z(k);
  // Poisson jump count over the horizon: lam = -ln(1 - p) per day.
  const double jump_mu = (dist == McDist::kJump) ? -std::log1p(-jumps.prob) * horizon_days : 0.0;
  const double t_scale = (dist == McDist::kStudentT) ? std::sqrt((df - 2.0) / df) : 1.0;
  for (std::size_t s = 0; s < n_scenarios; ++s) {
    for (std::size_t j = 0; j < k; ++j) z[j] = rng.normal();
    double* row = out.data.row(s);
    // x = L z (lower-triangular product).
    for (std::size_t i = 0; i < k; ++i) {
      const double* li = chol.lower.row(i);
      double acc = 0.0;
      for (std::size_t j = 0; j <= i; ++j) acc += li[j] * z[j];
      row[i] = acc;
    }
    if (dist == McDist::kStudentT) {
      const double w = rng.chisq(df) / df;
      const double m = t_scale / std::sqrt(w);
      for (std::size_t i = 0; i < k; ++i) row[i] *= m;
    } else if (dist == McDist::kJump) {
      const long n_jumps = rng.poisson(jump_mu);
      for (long q = 0; q < n_jumps; ++q) {
        for (std::size_t j = 0; j < jump_col.size(); ++j) {
          const double zj = (jump_std[j] > 0.0) ? rng.normal() : 0.0;
          row[jump_col[j]] += jump_mean[j] + jump_std[j] * zj;
        }
      }
    }
  }
  return out;
}

// ------------------------------------------------------------ diagnostics
double var_standard_error(const std::vector<double>& pnl, double alpha) {
  std::vector<double> ws(pnl.size());
  return value_or_throw(try_var_standard_error(pnl, alpha, ws));
}

double var_standard_error_bootstrap(const std::vector<double>& pnl, double alpha,
                                    std::size_t n_boot, std::uint64_t seed) {
  validate_alpha(alpha);
  const std::size_t n = pnl.size();
  if (n < 10)
    throw std::invalid_argument("need at least 10 scenarios to bootstrap a VaR standard error");
  if (n_boot < 2) throw std::invalid_argument("n_boot must be >= 2");
  for (double v : pnl)
    if (!std::isfinite(v)) throw std::invalid_argument("pnl contains NaN or inf");

  // Uniform-weight tail rank depends only on (n, alpha), not the data, so
  // it is identical for the original sample and every resample of the
  // same length: calling empirical_var per resample reuses that exact
  // order-statistic rule (see expected_shortfall.cpp's make_tail) rather
  // than re-deriving it here.
  Rng rng(seed);
  std::vector<double> resample(n);
  std::vector<double> ws(n);  // selection buffer, reused by every replicate
  std::vector<double> boot_vars(n_boot);
  for (std::size_t b = 0; b < n_boot; ++b) {
    for (std::size_t i = 0; i < n; ++i) {
      const auto idx = static_cast<std::size_t>(rng.uniform() * static_cast<double>(n));
      resample[i] = pnl[std::min(idx, n - 1)];
    }
    boot_vars[b] = value_or_throw(try_empirical_var_es(resample, alpha, ws)).first;
  }
  return sample_std(boot_vars);
}

// ---------------------------------------------------------------- driver
MonteCarloResult monte_carlo_var(const Book& book, const Market& market, const FactorCov& cov,
                                 const MonteCarloOptions& opts) {
  validate_alpha(opts.alpha);
  validate_horizon(opts.horizon_days);
  if (opts.n_boot < 2) throw std::invalid_argument("n_boot must be >= 2");
  const CompiledBook compiled(book, market);  // throws on empty book

  MonteCarloResult res;
  res.alpha = opts.alpha;
  res.horizon_days = opts.horizon_days;
  res.dist = opts.dist;
  res.n_scenarios = opts.n_scenarios;
  if (compiled.factors().empty()) {
    res.pnl.assign(opts.n_scenarios, 0.0);
    return res;  // pure base-ccy cash: zero risk
  }

  const FactorCov sub = cov.select(compiled.factors());
  CholeskyResult diag;
  const ReturnsMatrix scen =
      simulate_factor_returns(sub, opts.n_scenarios, opts.dist, opts.df, opts.jumps, opts.seed,
                              opts.horizon_days, &diag, opts.repair);
  res.cholesky_warning = diag.warning;
  res.pnl = compiled.pnl(scen);
  const auto [var, es] = empirical_var_es(res.pnl, opts.alpha);
  res.var = var;
  res.es = es;
  // Simulated once: VaR, ES and both SEs use the same scenarios.
  res.se_var = var_standard_error_bootstrap(res.pnl, opts.alpha, opts.n_boot,
                                            opts.seed ^ 0x9E3779B97F4A7C15ULL);
  res.se_var_kde = var_standard_error(res.pnl, opts.alpha);
  return res;
}

MonteCarloResult monte_carlo_var(const Book& book, const Market& market,
                                 std::span<const OptionLeg> options,
                                 const std::vector<std::string>& option_factors,
                                 const FactorCov& cov, const MonteCarloOptions& opts, double dt) {
  if (options.empty()) return monte_carlo_var(book, market, cov, opts);
  validate_alpha(opts.alpha);
  validate_horizon(opts.horizon_days);
  if (opts.n_boot < 2) throw std::invalid_argument("n_boot must be >= 2");
  const OptionBookLayout layout = layout_option_book(book, market, options, option_factors);

  MonteCarloResult res;
  res.alpha = opts.alpha;
  res.horizon_days = opts.horizon_days;
  res.dist = opts.dist;
  res.n_scenarios = opts.n_scenarios;

  const FactorCov sub = cov.select(layout.factors);
  CholeskyResult diag;
  const ReturnsMatrix scen =
      simulate_factor_returns(sub, opts.n_scenarios, opts.dist, opts.df, opts.jumps, opts.seed,
                              opts.horizon_days, &diag, opts.repair);
  res.cholesky_warning = diag.warning;
  res.pnl = option_book_pnl(layout, scen, dt);
  const auto [var, es] = empirical_var_es(res.pnl, opts.alpha);
  res.var = var;
  res.es = es;
  res.se_var = var_standard_error_bootstrap(res.pnl, opts.alpha, opts.n_boot,
                                            opts.seed ^ 0x9E3779B97F4A7C15ULL);
  res.se_var_kde = var_standard_error(res.pnl, opts.alpha);
  return res;
}

MonteCarloResult monte_carlo_var(const Book& book, const Market& market,
                                 std::span<const OptionLeg> options,
                                 const std::vector<std::string>& option_factors,
                                 const FactorCov& cov, const MonteCarloOptions& opts, double dt,
                                 const VolScenarios& vol) {
  if (options.empty()) throw std::invalid_argument("vol scenarios need option legs");
  validate_alpha(opts.alpha);
  validate_horizon(opts.horizon_days);
  if (opts.n_boot < 2) throw std::invalid_argument("n_boot must be >= 2");
  const OptionBookLayout layout = layout_option_book(book, market, options, option_factors);

  MonteCarloResult res;
  res.alpha = opts.alpha;
  res.horizon_days = opts.horizon_days;
  res.dist = opts.dist;
  res.n_scenarios = opts.n_scenarios;

  const FactorCov sub = cov.select(option_combined_factors(layout));
  CholeskyResult diag;
  const ReturnsMatrix scen =
      simulate_factor_returns(sub, opts.n_scenarios, opts.dist, opts.df, opts.jumps, opts.seed,
                              opts.horizon_days, &diag, opts.repair);
  res.cholesky_warning = diag.warning;
  res.pnl = option_book_pnl(layout, scen, dt, vol);
  const auto [var, es] = empirical_var_es(res.pnl, opts.alpha);
  res.var = var;
  res.es = es;
  res.se_var = var_standard_error_bootstrap(res.pnl, opts.alpha, opts.n_boot,
                                            opts.seed ^ 0x9E3779B97F4A7C15ULL);
  res.se_var_kde = var_standard_error(res.pnl, opts.alpha);
  return res;
}

}  // namespace fxvar
