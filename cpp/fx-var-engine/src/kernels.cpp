#include "fxvar/kernels.hpp"

#include "qp/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>

#include "fxvar/stats.hpp"

namespace fxvar {

// ------------------------------------------------------------------ errors
std::string message(const Error& e) {
  std::ostringstream msg;
  switch (e.code) {
    case Errc::kFixed:
      msg << e.text;
      break;
    case Errc::kAlphaRange:
      msg << "alpha is the tail probability (e.g. 0.01 for 99 %); must be in "
             "(0, 0.5), got "
          << e.a;
      break;
    case Errc::kHorizonRange:
      msg << "horizon_days must be > 0, got " << e.a;
      break;
    case Errc::kShockNonFinite:
      msg << "shock for " << e.text << " is not finite";
      break;
    case Errc::kWorkspaceTooSmall:
      msg << e.text << ": workspace holds " << static_cast<long long>(e.a)
          << " doubles, needs " << static_cast<long long>(e.b);
      break;
    case Errc::kSizeMismatch:
      msg << e.text << ": span lengths differ (" << static_cast<long long>(e.a)
          << " vs " << static_cast<long long>(e.b) << ")";
      break;
  }
  return msg.str();
}

void throw_error(const Error& e) { throw std::invalid_argument(message(e)); }

Result<void> try_validate_horizon(double horizon_days) noexcept {
  if (!(horizon_days > 0.0) || !std::isfinite(horizon_days)) [[unlikely]]
    return std::unexpected(Error{Errc::kHorizonRange, "", horizon_days});
  return {};
}

namespace {

Result<void> check_scale(double sigma, double mean) noexcept {
  if (!(sigma >= 0.0) || !std::isfinite(sigma)) [[unlikely]]
    return fail("sigma must be finite and >= 0");
  if (!std::isfinite(mean)) [[unlikely]] return fail("mean must be finite");
  return {};
}

Result<double> t_scale(double df) noexcept {
  if (!(df > 2.0) || !std::isfinite(df)) [[unlikely]]
    return fail("Student-t df must be finite and > 2 for finite variance");
  return std::sqrt((df - 2.0) / df);
}

std::unexpected<Error> too_small(const char* who, std::size_t have, std::size_t need) noexcept {
  return std::unexpected(Error{Errc::kWorkspaceTooSmall, who, static_cast<double>(have),
                               static_cast<double>(need)});
}

/// Value of order statistic `lo` and `lo + 1` (clamped) of `ws`, by
/// selection: the same two numbers a full ascending sort puts there.
std::pair<double, double> order_pair(std::span<double> ws, std::size_t lo) noexcept {
  const std::size_t n = ws.size();
  std::nth_element(ws.begin(), ws.begin() + static_cast<std::ptrdiff_t>(lo), ws.end());
  const double a = ws[lo];
  if (lo + 1 >= n) return {a, a};
  const double b = *std::min_element(ws.begin() + static_cast<std::ptrdiff_t>(lo) + 1, ws.end());
  return {a, b};
}

}  // namespace

// ------------------------------------------------------------ empirical
Result<std::pair<double, double>> try_empirical_var_es(std::span<const double> pnl,
                                                       double alpha,
                                                       std::span<double> ws) noexcept {
  if (auto e = try_validate_alpha(alpha); !e) [[unlikely]] return std::unexpected(e.error());
  const std::size_t n = pnl.size();
  if (n == 0) [[unlikely]] return fail("pnl sample is empty");
  // Finiteness as a branch-free count (vectorises), then the copy.
  std::size_t bad = 0;
  for (std::size_t i = 0; i < n; ++i) bad += std::fabs(pnl[i]) <= 1.7976931348623157e308 ? 0U : 1U;
  if (bad != 0) [[unlikely]] return fail("pnl sample contains NaN or inf (NaN policy: refuse)");
  if (ws.size() < n) [[unlikely]] return too_small("try_empirical_var_es", ws.size(), n);
  double* loss = ws.data();
  for (std::size_t i = 0; i < n; ++i) loss[i] = -pnl[i];
  const double k = static_cast<double>(n) * alpha;
  const double mc = std::ceil(k - 1e-9);
  const std::size_t m = mc < 1.0 ? 1 : std::min(n, static_cast<std::size_t>(mc));
  std::nth_element(loss, loss + static_cast<std::ptrdiff_t>(m - 1), loss + n,
                   std::greater<double>());
  const double lm = loss[m - 1];
  double tail = 0.0;
  for (std::size_t i = 0; i + 1 < m; ++i) tail += loss[i];
  return std::pair<double, double>{lm, (tail + (k - static_cast<double>(m - 1)) * lm) / k};
}

Result<double> try_var_standard_error(std::span<const double> pnl, double alpha,
                                      std::span<double> ws) noexcept {
  if (auto e = try_validate_alpha(alpha); !e) [[unlikely]] return std::unexpected(e.error());
  const std::size_t n = pnl.size();
  if (n < 10) [[unlikely]] return fail("need at least 10 scenarios for a VaR standard error");
  const auto ve = try_empirical_var_es(pnl, alpha, ws);
  if (!ve) [[unlikely]] return std::unexpected(ve.error());
  const double q = -ve->first;

  // sample_std: same two-pass mean / sum of squares as sample_moments.
  const double dn = static_cast<double>(n);
  double mean = 0.0;
  for (double v : pnl) mean += v;
  mean /= dn;
  double m2 = 0.0;
  for (double v : pnl) {
    const double d = v - mean;
    m2 += d * d;
  }
  const double sd = std::sqrt(m2 / (dn - 1.0));

  std::copy(pnl.begin(), pnl.end(), ws.begin());
  const std::span<double> sorted = ws.first(n);
  const auto quantile = [&](double p) {
    const double pos = p * static_cast<double>(n - 1);
    const std::size_t lo = static_cast<std::size_t>(pos);
    const double frac = pos - static_cast<double>(lo);
    const auto [a, b] = order_pair(sorted, lo);
    return a * (1.0 - frac) + b * frac;
  };
  const double q75 = quantile(0.75);
  const double q25 = quantile(0.25);
  const double iqr = q75 - q25;
  double spread = sd;
  if (iqr > 0.0) spread = std::min(sd, iqr / 1.34);
  if (!(spread > 0.0)) return 0.0;
  const double h = 0.9 * spread * std::pow(dn, -0.2);
  double dens = 0.0;
  for (double x : pnl) dens += norm_pdf((q - x) / h);
  dens /= dn * h;
  dens = std::max(dens, 1e-300);
  return std::sqrt(alpha * (1.0 - alpha) / dn) / dens;
}

// ------------------------------------------------------------ closed form
Result<double> try_normal_var(double sigma, double alpha, double mean) noexcept {
  if (auto e = try_validate_alpha(alpha); !e) [[unlikely]] return std::unexpected(e.error());
  if (auto e = check_scale(sigma, mean); !e) [[unlikely]] return std::unexpected(e.error());
  return -mean + sigma * norm_ppf(1.0 - alpha);
}

Result<double> try_normal_es(double sigma, double alpha, double mean) noexcept {
  if (auto e = try_validate_alpha(alpha); !e) [[unlikely]] return std::unexpected(e.error());
  if (auto e = check_scale(sigma, mean); !e) [[unlikely]] return std::unexpected(e.error());
  const double z = norm_ppf(1.0 - alpha);
  return -mean + sigma * norm_pdf(z) / alpha;
}

Result<double> try_t_var(double sigma, double alpha, double df, double mean) noexcept {
  if (auto e = try_validate_alpha(alpha); !e) [[unlikely]] return std::unexpected(e.error());
  if (auto e = check_scale(sigma, mean); !e) [[unlikely]] return std::unexpected(e.error());
  const auto s = t_scale(df);
  if (!s) [[unlikely]] return s;
  return -mean + sigma * *s * t_ppf(1.0 - alpha, df);
}

Result<double> try_t_es(double sigma, double alpha, double df, double mean) noexcept {
  if (auto e = try_validate_alpha(alpha); !e) [[unlikely]] return std::unexpected(e.error());
  if (auto e = check_scale(sigma, mean); !e) [[unlikely]] return std::unexpected(e.error());
  const auto s = t_scale(df);
  if (!s) [[unlikely]] return s;
  const double q = t_ppf(1.0 - alpha, df);
  const double es_std = t_pdf(q, df) * (df + q * q) / (alpha * (df - 1.0));
  return -mean + sigma * *s * es_std;
}

// ------------------------------------------------------------ linear algebra
Result<void> try_matvec(const Matrix& a, std::span<const double> x,
                        std::span<double> y) noexcept {
  if (a.cols() != x.size()) [[unlikely]] return fail("matvec: dimension mismatch");
  for (double v : x)
    if (!std::isfinite(v)) [[unlikely]] return fail("matvec: x contains NaN or inf");
  if (y.size() != a.rows()) [[unlikely]]
    return std::unexpected(Error{Errc::kSizeMismatch, "try_matvec",
                                 static_cast<double>(y.size()),
                                 static_cast<double>(a.rows())});
  // Row-major A: each row is one contiguous dot product (in index order).
  for (std::size_t i = 0; i < a.rows(); ++i) {
    const double* arow = a.row(i);
    double s = 0.0;
    for (std::size_t j = 0; j < a.cols(); ++j) s += arow[j] * x[j];
    y[i] = s;
  }
  return {};
}

Result<double> try_quad_form(std::span<const double> w, const Matrix& a) noexcept {
  if (a.rows() != a.cols() || a.rows() != w.size()) [[unlikely]]
    return fail("quad_form: dimension mismatch");
  for (double v : w)
    if (!std::isfinite(v)) [[unlikely]] return fail("quad_form: w contains NaN or inf");
  double total = 0.0;
  for (std::size_t i = 0; i < a.rows(); ++i) {
    const double* arow = a.row(i);
    double s = 0.0;
    for (std::size_t j = 0; j < a.cols(); ++j) s += arow[j] * w[j];
    total += w[i] * s;
  }
  return total;
}

Result<double> try_portfolio_sigma(std::span<const double> exposures,
                                   const Matrix& cov) noexcept {
  if (exposures.empty()) [[unlikely]] return fail("portfolio_sigma: book has no positions");
  bool any = false;
  for (double w : exposures) any = any || w != 0.0;
  for (double w : exposures)
    if (!std::isfinite(w)) [[unlikely]]
      return fail("portfolio_sigma: exposures must be finite");
  if (!any) [[unlikely]] return fail("portfolio_sigma: book has no positions");
  const auto qf = try_quad_form(exposures, cov);
  if (!qf) [[unlikely]] return qf;
  const double var = *qf;
  if (!std::isfinite(var)) [[unlikely]]
    return fail(
        "portfolio_sigma: w'Sw is not finite (covariance contains NaN/Inf or "
        "the exposures overflow)");
  double wmax = 0.0;
  for (double w : exposures) wmax = std::max(wmax, std::abs(w));
  double cmax = 0.0;
  for (std::size_t i = 0; i < cov.rows(); ++i)
    for (std::size_t j = 0; j < cov.cols(); ++j) cmax = std::max(cmax, std::abs(cov(i, j)));
  const double tol = 1e-10 * std::max(1.0, wmax * wmax * cmax);
  if (var < -tol) [[unlikely]]
    return fail("portfolio_sigma: covariance matrix is not positive semi-definite");
  return std::sqrt(std::max(var, 0.0));
}

Result<VarEs> try_var_covar(std::span<const double> exposures, const Matrix& cov,
                            double alpha, double horizon_days, TailDist dist, double df,
                            double mean) noexcept {
  if (auto e = try_validate_alpha(alpha); !e) [[unlikely]] return std::unexpected(e.error());
  if (auto e = try_validate_horizon(horizon_days); !e) [[unlikely]]
    return std::unexpected(e.error());
  if (!std::isfinite(mean)) [[unlikely]] return fail("mean must be finite");
  const auto sig1 = try_portfolio_sigma(exposures, cov);
  if (!sig1) [[unlikely]] return std::unexpected(sig1.error());
  const double scale = std::sqrt(horizon_days);
  const double sig = *sig1 * scale;
  const double mu = mean * horizon_days;
  const auto v = dist == TailDist::kNormal ? try_normal_var(sig, alpha, mu)
                                           : try_t_var(sig, alpha, df, mu);
  if (!v) [[unlikely]] return std::unexpected(v.error());
  const auto e = dist == TailDist::kNormal ? try_normal_es(sig, alpha, mu)
                                           : try_t_es(sig, alpha, df, mu);
  if (!e) [[unlikely]] return std::unexpected(e.error());
  return VarEs{*v, *e};
}

Result<void> try_cholesky_lower(const Matrix& a, std::span<double> lower) noexcept {
  const std::size_t n = a.rows();
  if (!(n == a.cols() && n > 0)) [[unlikely]]
    return fail("try_cholesky_lower: matrix must be square and non-empty");
  if (lower.size() < n * n) [[unlikely]] return too_small("try_cholesky_lower", lower.size(), n * n);
  for (double v : a.data())
    if (!std::isfinite(v)) [[unlikely]]
      return fail("try_cholesky_lower: matrix contains NaN or infinite entries");
  // Same loop order and arithmetic as cholesky()'s plain factorisation.
  if (!qp::cholesky_raw(a.data().data(), n, lower.data())) [[unlikely]]
    return fail("try_cholesky_lower: matrix is not positive definite");
  return {};
}

Result<void> try_simulate_factor_returns(std::span<const double> lower, std::size_t k,
                                         std::size_t n_scenarios, McDist dist, double df,
                                         std::uint64_t seed, std::span<double> out) noexcept {
  if (n_scenarios < 1) [[unlikely]] return fail("n_scenarios must be >= 1");
  if (dist == McDist::kJump) [[unlikely]]
    return fail("try_simulate_factor_returns: kJump is not supported; use simulate_factor_returns");
  if (dist == McDist::kStudentT && (!(df > 2.0) || !std::isfinite(df))) [[unlikely]]
    return fail("Student-t df must be finite and > 2 for finite variance");
  if (lower.size() < k * k) [[unlikely]]
    return too_small("try_simulate_factor_returns (lower)", lower.size(), k * k);
  if (out.size() < n_scenarios * k) [[unlikely]]
    return too_small("try_simulate_factor_returns", out.size(), n_scenarios * k);
  Rng rng(seed);
  const double t_scale = dist == McDist::kStudentT ? std::sqrt((df - 2.0) / df) : 1.0;
  for (std::size_t s = 0; s < n_scenarios; ++s) {
    double* row = out.data() + s * k;
    // Draw z into the row, then x = L z in place from the last row up:
    // row i needs z_0..z_i only, and z_i is overwritten after its last use.
    for (std::size_t j = 0; j < k; ++j) row[j] = rng.normal();
    for (std::size_t i = k; i-- > 0;) {
      const double* li = lower.data() + i * k;
      double acc = 0.0;
      for (std::size_t j = 0; j <= i; ++j) acc += li[j] * row[j];
      row[i] = acc;
    }
    if (dist == McDist::kStudentT) {
      const double w = rng.chisq(df) / df;
      const double m = t_scale / std::sqrt(w);
      for (std::size_t i = 0; i < k; ++i) row[i] *= m;
    }
  }
  return {};
}

}  // namespace fxvar
