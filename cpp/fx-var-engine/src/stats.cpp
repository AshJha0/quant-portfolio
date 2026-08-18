#include "fxvar/stats.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace fxvar {

namespace {
constexpr double kSqrt2 = 1.4142135623730950488016887242097;
constexpr double kInvSqrt2Pi = 0.39894228040143267793994605993438;
constexpr double kPi = 3.1415926535897932384626433832795;
}  // namespace

double validate_alpha(double alpha) {
  if (!(alpha > 0.0 && alpha < 1.0)) {
    std::ostringstream msg;
    msg << "alpha must be in (0, 1), got " << alpha;
    throw std::invalid_argument(msg.str());
  }
  return alpha;
}

double validate_horizon(double horizon_days) {
  if (!(horizon_days > 0.0)) {
    std::ostringstream msg;
    msg << "horizon_days must be > 0, got " << horizon_days;
    throw std::invalid_argument(msg.str());
  }
  return horizon_days;
}

double norm_pdf(double x) { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

double norm_cdf(double x) { return 0.5 * std::erfc(-x / kSqrt2); }

double norm_ppf(double p) {
  if (!(p > 0.0 && p < 1.0))
    throw std::invalid_argument("norm_ppf: p must be in (0, 1)");

  // Acklam's rational approximation.
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                             -2.759285104469687e+02, 1.383577518672690e+02,
                             -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                             -1.556989798598866e+02, 6.680131188771972e+01,
                             -1.328068155288572e+01};
  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                             -2.400758277161838e+00, -2.549732539343734e+00,
                             4.374664141464968e+00,  2.938163982698783e+00};
  static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                             2.445134137142996e+00, 3.754408661907416e+00};
  const double plow = 0.02425;
  double x;
  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  } else if (p <= 1.0 - plow) {
    const double q = p - 0.5;
    const double r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
        q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
  } else {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }

  // One Halley refinement step: takes the ~1e-9 approximation to < 1e-13.
  const double e = norm_cdf(x) - p;
  const double u = e / norm_pdf(x);          // Newton step
  x -= u / (1.0 + 0.5 * x * u);              // Halley correction
  return x;
}

// ---------------------------------------------------------------- gamma
namespace {

double gamma_p_series(double a, double x) {
  double ap = a;
  double sum = 1.0 / a;
  double del = sum;
  for (int n = 0; n < 500; ++n) {
    ap += 1.0;
    del *= x / ap;
    sum += del;
    if (std::abs(del) < std::abs(sum) * 1e-16) break;
  }
  return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

double gamma_q_contfrac(double a, double x) {
  const double tiny = 1e-300;
  double b = x + 1.0 - a;
  double c = 1.0 / tiny;
  double d = 1.0 / b;
  double h = d;
  for (int i = 1; i <= 500; ++i) {
    const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
    b += 2.0;
    d = an * d + b;
    if (std::abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (std::abs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::abs(del - 1.0) < 1e-16) break;
  }
  return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

}  // namespace

double reg_lower_gamma(double a, double x) {
  if (a <= 0.0 || x < 0.0)
    throw std::invalid_argument("reg_lower_gamma: require a > 0 and x >= 0");
  if (x == 0.0) return 0.0;
  return (x < a + 1.0) ? gamma_p_series(a, x) : 1.0 - gamma_q_contfrac(a, x);
}

double reg_upper_gamma(double a, double x) {
  if (a <= 0.0 || x < 0.0)
    throw std::invalid_argument("reg_upper_gamma: require a > 0 and x >= 0");
  if (x == 0.0) return 1.0;
  return (x < a + 1.0) ? 1.0 - gamma_p_series(a, x) : gamma_q_contfrac(a, x);
}

double chi2_sf(double x, double df) {
  if (df <= 0.0) throw std::invalid_argument("chi2_sf: df must be > 0");
  if (x <= 0.0) return 1.0;
  return reg_upper_gamma(0.5 * df, 0.5 * x);
}

// ----------------------------------------------------------------- beta
namespace {

double betacf(double a, double b, double x) {
  const double tiny = 1e-300;
  const double qab = a + b;
  const double qap = a + 1.0;
  const double qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::abs(d) < tiny) d = tiny;
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m <= 500; ++m) {
    const double dm = static_cast<double>(m);
    const double m2 = 2.0 * dm;
    double aa = dm * (b - dm) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < tiny) d = tiny;
    c = 1.0 + aa / c;
    if (std::abs(c) < tiny) c = tiny;
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + dm) * (qab + dm) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < tiny) d = tiny;
    c = 1.0 + aa / c;
    if (std::abs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::abs(del - 1.0) < 1e-16) break;
  }
  return h;
}

}  // namespace

double reg_inc_beta(double a, double b, double x) {
  if (a <= 0.0 || b <= 0.0)
    throw std::invalid_argument("reg_inc_beta: require a, b > 0");
  if (x < 0.0 || x > 1.0)
    throw std::invalid_argument("reg_inc_beta: x must be in [0, 1]");
  if (x == 0.0) return 0.0;
  if (x == 1.0) return 1.0;
  const double lnfront = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                         a * std::log(x) + b * std::log1p(-x);
  const double front = std::exp(lnfront);
  if (x < (a + 1.0) / (a + b + 2.0)) return front * betacf(a, b, x) / a;
  return 1.0 - front * betacf(b, a, 1.0 - x) / b;
}

// ------------------------------------------------------------ Student-t
double t_pdf(double x, double df) {
  if (df <= 0.0) throw std::invalid_argument("t_pdf: df must be > 0");
  const double lognorm = std::lgamma(0.5 * (df + 1.0)) -
                         std::lgamma(0.5 * df) -
                         0.5 * std::log(df * kPi);
  return std::exp(lognorm - 0.5 * (df + 1.0) * std::log1p(x * x / df));
}

double t_cdf(double x, double df) {
  if (df <= 0.0) throw std::invalid_argument("t_cdf: df must be > 0");
  if (x == 0.0) return 0.5;
  const double z = df / (df + x * x);
  const double tail = 0.5 * reg_inc_beta(0.5 * df, 0.5, z);
  return (x > 0.0) ? 1.0 - tail : tail;
}

double t_ppf(double p, double df) {
  if (!(p > 0.0 && p < 1.0))
    throw std::invalid_argument("t_ppf: p must be in (0, 1)");
  if (df <= 0.0) throw std::invalid_argument("t_ppf: df must be > 0");
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
  if (p < 0.0 || p > 1.0)
    throw std::invalid_argument("binom_cdf: p must be in [0, 1]");
  if (k < 0) return 0.0;
  if (k >= n) return 1.0;
  if (p == 0.0) return 1.0;
  if (p == 1.0) return 0.0;  // k < n here
  const double logp = std::log(p);
  const double log1mp = std::log1p(-p);
  double cum = 0.0;
  for (int i = 0; i <= k; ++i) {
    const double logterm = std::lgamma(n + 1.0) - std::lgamma(i + 1.0) -
                           std::lgamma(n - i + 1.0) + i * logp +
                           (n - i) * log1mp;
    cum += std::exp(logterm);
  }
  return std::min(cum, 1.0);
}

Moments sample_moments(const std::vector<double>& x) {
  const std::size_t n = x.size();
  if (n < 2)
    throw std::invalid_argument("sample_moments: need at least 2 observations");
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
