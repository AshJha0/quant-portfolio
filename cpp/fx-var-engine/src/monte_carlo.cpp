#include "fxvar/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "fxvar/expected_shortfall.hpp"
#include "fxvar/stats.hpp"

namespace fxvar {

// -------------------------------------------------------------------- Rng
double Rng::uniform() {
  // 53-bit mantissa in (0, 1): (x >> 11) in [0, 2^53), +0.5 keeps the
  // result strictly inside the open interval so norm_ppf never sees 0/1.
  const std::uint64_t x = engine_();
  return (static_cast<double>(x >> 11) + 0.5) * 0x1.0p-53;
}

double Rng::normal() { return norm_ppf(uniform()); }

double Rng::gamma(double shape) {
  if (!(shape > 0.0)) throw std::invalid_argument("gamma: shape must be > 0");
  if (shape < 1.0) {
    // Boost: Gamma(a) = Gamma(a + 1) * U^(1/a).
    const double u = uniform();
    return gamma(shape + 1.0) * std::pow(u, 1.0 / shape);
  }
  // Marsaglia-Tsang squeeze method.
  const double d = shape - 1.0 / 3.0;
  const double c = 1.0 / std::sqrt(9.0 * d);
  for (;;) {
    double x, v;
    do {
      x = normal();
      v = 1.0 + c * x;
    } while (v <= 0.0);
    v = v * v * v;
    const double u = uniform();
    const double x2 = x * x;
    if (u < 1.0 - 0.0331 * x2 * x2) return d * v;
    if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) return d * v;
  }
}

double Rng::chisq(double df) {
  if (!(df > 0.0)) throw std::invalid_argument("chisq: df must be > 0");
  return 2.0 * gamma(0.5 * df);
}

// -------------------------------------------------------------- simulate
ReturnsMatrix simulate_factor_returns(const FactorCov& cov,
                                      std::size_t n_scenarios, McDist dist,
                                      double df, const JumpSpec& jumps,
                                      std::uint64_t seed, double horizon_days,
                                      CholeskyResult* diag) {
  validate_horizon(horizon_days);
  if (n_scenarios < 1)
    throw std::invalid_argument("n_scenarios must be >= 1");
  const std::size_t k = cov.factors.size();
  if (cov.cov.rows() != k || cov.cov.cols() != k)
    throw std::invalid_argument(
        "simulate_factor_returns: cov labels do not match matrix shape");
  if (dist == McDist::kStudentT && df <= 2.0)
    throw std::invalid_argument("Student-t df must be > 2 for finite variance");
  if (dist == McDist::kJump) {
    if (!(jumps.prob >= 0.0 && jumps.prob <= 1.0)) {
      std::ostringstream msg;
      msg << "jump prob must be in [0, 1], got " << jumps.prob;
      throw std::invalid_argument(msg.str());
    }
    for (const auto& [f, s] : jumps.stdev)
      if (s < 0.0)
        throw std::invalid_argument("jump stdev for " + f + " must be >= 0");
  }

  Matrix a = cov.cov;
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j) a(i, j) *= horizon_days;
  CholeskyResult chol = robust_cholesky(a);
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
        if (cov.factors[j] == f) { col = static_cast<int>(j); break; }
      if (col < 0) continue;  // scenario library convention: ignore unknowns
      jump_col.push_back(col);
      jump_mean.push_back(m);
      const auto it = jumps.stdev.find(f);
      jump_std.push_back(it == jumps.stdev.end() ? 0.0 : it->second);
    }
  }

  std::vector<double> z(k);
  const double t_scale = (dist == McDist::kStudentT)
                             ? std::sqrt((df - 2.0) / df)
                             : 1.0;
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
      const bool hit = rng.uniform() < jumps.prob;
      // Draw the jump normals unconditionally so the scenario count, not
      // the hit pattern, fixes the random stream (variance-reduction-
      // friendly and easier to reason about for reproducibility).
      for (std::size_t j = 0; j < jump_col.size(); ++j) {
        const double zj = (jump_std[j] > 0.0) ? rng.normal() : 0.0;
        if (hit) row[jump_col[j]] += jump_mean[j] + jump_std[j] * zj;
      }
    }
  }
  return out;
}

// ------------------------------------------------------------ diagnostics
double var_standard_error(const std::vector<double>& pnl, double alpha) {
  validate_alpha(alpha);
  const std::size_t n = pnl.size();
  if (n < 10)
    throw std::invalid_argument(
        "need at least 10 scenarios for a VaR standard error");
  // Loss quantile q with the same tail convention as empirical_var.
  const double q = -empirical_var(pnl, alpha);

  // Gaussian KDE with Silverman bandwidth h = 0.9 min(sd, IQR/1.34) n^-1/5.
  const double sd = sample_std(pnl);
  std::vector<double> sorted(pnl);
  std::sort(sorted.begin(), sorted.end());
  const auto quantile_sorted = [&sorted](double p) {
    const double pos = p * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(pos);
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
  };
  const double iqr = quantile_sorted(0.75) - quantile_sorted(0.25);
  double spread = sd;
  if (iqr > 0.0) spread = std::min(sd, iqr / 1.34);
  if (!(spread > 0.0)) spread = 1e-300;
  const double h =
      0.9 * spread * std::pow(static_cast<double>(n), -0.2);

  double dens = 0.0;
  for (double x : pnl) dens += norm_pdf((q - x) / h);
  dens /= static_cast<double>(n) * h;
  dens = std::max(dens, 1e-300);
  return std::sqrt(alpha * (1.0 - alpha) / static_cast<double>(n)) / dens;
}

// ---------------------------------------------------------------- driver
MonteCarloResult monte_carlo_var(const Book& book, const Market& market,
                                 const FactorCov& cov,
                                 const MonteCarloOptions& opts) {
  validate_alpha(opts.alpha);
  validate_horizon(opts.horizon_days);
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
      simulate_factor_returns(sub, opts.n_scenarios, opts.dist, opts.df,
                              opts.jumps, opts.seed, opts.horizon_days, &diag);
  res.cholesky_warning = diag.warning;
  res.pnl = compiled.pnl(scen);
  const auto [var, es] = empirical_var_es(res.pnl, opts.alpha);
  res.var = var;
  res.es = es;
  res.se_var = var_standard_error(res.pnl, opts.alpha);
  return res;
}

}  // namespace fxvar
