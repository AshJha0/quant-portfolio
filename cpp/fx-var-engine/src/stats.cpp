#include "fxvar/stats.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "qp/normal.hpp"
#include "qp/special.hpp"

namespace fxvar {

namespace {
using qp::kPi;
}  // namespace

double validate_alpha(double alpha) {
  if (!(alpha > 0.0 && alpha < 0.5)) {
    std::ostringstream msg;
    msg << "alpha is the tail probability (e.g. 0.01 for 99 %); must be in "
           "(0, 0.5), got "
        << alpha;
    throw std::invalid_argument(msg.str());
  }
  return alpha;
}

double validate_horizon(double horizon_days) {
  if (!(horizon_days > 0.0) || !std::isfinite(horizon_days)) {
    std::ostringstream msg;
    msg << "horizon_days must be > 0, got " << horizon_days;
    throw std::invalid_argument(msg.str());
  }
  return horizon_days;
}

// ------------------------------------------------------------ log-gamma
// Lanczos (g = 7, n = 9), ~1e-15 relative. Thread-safe: std::lgamma writes
// glibc's global signgam, so the library never calls it.
double log_gamma(double x) {
  if (!(x > 0.0) || !std::isfinite(x))
    throw std::invalid_argument("log_gamma: x must be finite and > 0");
  return qp::log_gamma_lanczos(x);
}

double norm_pdf(double x) { return qp::norm_pdf(x); }

double norm_cdf(double x) { return qp::norm_cdf(x); }

double norm_ppf(double p) {
  if (!(p > 0.0 && p < 1.0))
    throw std::invalid_argument("norm_ppf: p must be in (0, 1)");

  // Acklam plus one Halley step: takes the ~1e-9 approximation to < 1e-13.
  return qp::norm_ppf_unchecked<1>(p);
}

// ---------------------------------------------------------------- gamma
namespace {
constexpr double kEps = 1e-16;  // series / continued-fraction tolerance
constexpr int kBetaMaxIter = 500;
}  // namespace

double reg_lower_gamma(double a, double x) {
  if (!(a > 0.0) || !(x >= 0.0) || !std::isfinite(a) || !std::isfinite(x))
    throw std::invalid_argument("reg_lower_gamma: require finite a > 0 and x >= 0");
  if (x == 0.0) return 0.0;
  return (x < a + 1.0) ? qp::gamma_p_series(a, x, kEps) : 1.0 - qp::gamma_q_cf(a, x, kEps);
}

double reg_upper_gamma(double a, double x) {
  if (!(a > 0.0) || !(x >= 0.0) || !std::isfinite(a) || !std::isfinite(x))
    throw std::invalid_argument("reg_upper_gamma: require finite a > 0 and x >= 0");
  if (x == 0.0) return 1.0;
  return (x < a + 1.0) ? 1.0 - qp::gamma_p_series(a, x, kEps) : qp::gamma_q_cf(a, x, kEps);
}

double chi2_sf(double x, double df) {
  if (!(df > 0.0) || !std::isfinite(df))
    throw std::invalid_argument("chi2_sf: df must be finite and > 0");
  if (std::isnan(x)) throw std::invalid_argument("chi2_sf: x is NaN");
  if (x <= 0.0) return 1.0;
  if (std::isinf(x)) return 0.0;
  return reg_upper_gamma(0.5 * df, 0.5 * x);
}

// ----------------------------------------------------------------- beta

double reg_inc_beta(double a, double b, double x) {
  if (!(a > 0.0) || !(b > 0.0) || !std::isfinite(a) || !std::isfinite(b))
    throw std::invalid_argument("reg_inc_beta: require finite a, b > 0");
  if (!(x >= 0.0 && x <= 1.0))
    throw std::invalid_argument("reg_inc_beta: x must be in [0, 1]");
  if (x == 0.0) return 0.0;
  if (x == 1.0) return 1.0;
  return qp::betainc_unchecked(a, b, x, kEps, kBetaMaxIter);
}

// ------------------------------------------------------------ Student-t
double t_pdf(double x, double df) {
  if (!(df > 0.0) || !std::isfinite(df) || !std::isfinite(x))
    throw std::invalid_argument("t_pdf: need finite x and finite df > 0");
  const double lognorm = log_gamma(0.5 * (df + 1.0)) -
                         log_gamma(0.5 * df) -
                         0.5 * std::log(df * kPi);
  return std::exp(lognorm - 0.5 * (df + 1.0) * std::log1p(x * x / df));
}

double t_cdf(double x, double df) {
  if (!(df > 0.0) || !std::isfinite(df) || std::isnan(x))
    throw std::invalid_argument("t_cdf: need non-NaN x and finite df > 0");
  if (std::isinf(x)) return x > 0.0 ? 1.0 : 0.0;
  if (x == 0.0) return 0.5;
  const double z = df / (df + x * x);
  const double tail = 0.5 * reg_inc_beta(0.5 * df, 0.5, z);
  return (x > 0.0) ? 1.0 - tail : tail;
}

double t_ppf(double p, double df) {
  if (!(p > 0.0 && p < 1.0))
    throw std::invalid_argument("t_ppf: p must be in (0, 1)");
  if (!(df > 0.0) || !std::isfinite(df))
    throw std::invalid_argument("t_ppf: df must be finite and > 0");
  if (p == 0.5) return 0.0;

  // Start from the normal quantile with a first-order fat-tail correction,
  // then bracketed Newton on the CDF.
  const double z = norm_ppf(p);
  double x = z + (z * z * z + z) / (4.0 * df);
  double lo = -std::numeric_limits<double>::infinity();
  double hi = std::numeric_limits<double>::infinity();
  for (int it = 0; it < 100; ++it) {
    const double f = t_cdf(x, df) - p;
    if (f > 0.0)
      hi = x;
    else
      lo = x;
    const double dens = t_pdf(x, df);
    double step = f / std::max(dens, 1e-300);
    double xn = x - step;
    if (!(xn > lo && xn < hi))
      xn = std::isinf(lo)   ? hi - std::max(1.0, std::abs(hi))
           : std::isinf(hi) ? lo + std::max(1.0, std::abs(lo))
                            : 0.5 * (lo + hi);
    if (std::abs(xn - x) < 1e-14 * (1.0 + std::abs(x))) {
      x = xn;
      break;
    }
    x = xn;
  }
  return x;
}

double binom_cdf(int k, int n, double p) {
  if (n < 0) throw std::invalid_argument("binom_cdf: n must be >= 0");
  if (!(p >= 0.0 && p <= 1.0))
    throw std::invalid_argument("binom_cdf: p must be in [0, 1]");
  if (k < 0) return 0.0;
  if (k >= n) return 1.0;
  if (p == 0.0) return 1.0;
  if (p == 1.0) return 0.0;  // k < n here
  return qp::binomial_cdf_unchecked(k, n, p, kEps, kBetaMaxIter);
}

Moments sample_moments(const std::vector<double>& x) {
  const std::size_t n = x.size();
  if (n < 2)
    throw std::invalid_argument("sample_moments: need at least 2 observations");
  for (double v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument("sample_moments: input contains NaN or inf");
  const double dn = static_cast<double>(n);
  double mean = 0.0;
  for (double v : x) mean += v;
  mean /= dn;
  double m2 = 0.0, m3 = 0.0, m4 = 0.0;
  for (double v : x) {
    const double d = v - mean;
    const double d2 = d * d;
    m2 += d2;
    m3 += d2 * d;
    m4 += d2 * d2;
  }
  Moments out;
  out.mean = mean;
  out.stdev = std::sqrt(m2 / (dn - 1.0));
  m2 /= dn;
  m3 /= dn;
  m4 /= dn;
  if (m2 > 0.0) {
    out.skewness = m3 / std::pow(m2, 1.5);
    out.excess_kurtosis = m4 / (m2 * m2) - 3.0;
  }
  return out;
}

double sample_std(const std::vector<double>& x) {
  return sample_moments(x).stdev;
}

}  // namespace fxvar
