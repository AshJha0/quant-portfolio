#include "eqvar/stats.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "qp/normal.hpp"
#include "qp/special.hpp"

namespace eqvar {

namespace {
using qp::kPi;
}  // namespace

// --------------------------------------------------------------------------
// Normal
// --------------------------------------------------------------------------

double normal_pdf(double x) noexcept { return qp::norm_pdf(x); }

double normal_cdf(double x) noexcept { return qp::norm_cdf(x); }

double normal_ppf(double p) {
    if (!(p > 0.0 && p < 1.0)) {
        throw std::invalid_argument("normal_ppf: p must be in (0, 1), got " + std::to_string(p));
    }
    // Route the upper half through the lower tail: 1 - p is EXACT for doubles
    // p in [0.5, 1) (Sterbenz), and the erfc-based CDF used by the Halley
    // refinement is accurate in the lower tail but suffers cancellation near 1.
    // This also makes Phi^{-1}(p) = -Phi^{-1}(1-p) hold bitwise.
    if (p > 0.5) return -normal_ppf(1.0 - p);
    // Acklam (2003) plus one Halley step against the erfc-based cdf.
    return qp::norm_ppf_unchecked<1>(p);
}

// --------------------------------------------------------------------------
// Log-gamma (Lanczos, g = 7, n = 9). Thread-safe: std::lgamma writes the
// global signgam in glibc, so the library never calls it.
// --------------------------------------------------------------------------

double log_gamma(double x) {
    if (!(x > 0.0) || !std::isfinite(x)) {
        throw std::invalid_argument("log_gamma: x must be finite and > 0");
    }
    return qp::log_gamma_lanczos(x);
}

// --------------------------------------------------------------------------
// Regularized incomplete beta (continued fraction, modified Lentz)
// --------------------------------------------------------------------------

namespace {
constexpr double kBetaEps = 3e-16;
constexpr int kBetaMaxIter = 300;
constexpr double kGammaEps = 3e-16;
}  // namespace

double betainc_reg(double a, double b, double x) {
    if (!(a > 0.0) || !(b > 0.0) || !std::isfinite(a) || !std::isfinite(b)) {
        throw std::invalid_argument("betainc_reg: a and b must be finite and > 0");
    }
    if (!(x >= 0.0 && x <= 1.0)) {
        throw std::invalid_argument("betainc_reg: x must be in [0, 1], got " + std::to_string(x));
    }
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;
    return qp::betainc_unchecked(a, b, x, kBetaEps, kBetaMaxIter);
}

// --------------------------------------------------------------------------
// Regularized incomplete gamma
// --------------------------------------------------------------------------

double regularized_gamma_p(double a, double x) {
    if (!(a > 0.0) || !std::isfinite(a)) {
        throw std::invalid_argument("regularized_gamma_p: a must be finite and > 0");
    }
    if (!(x >= 0.0) || !std::isfinite(x)) {
        throw std::invalid_argument("regularized_gamma_p: x must be finite and >= 0");
    }
    if (x == 0.0) return 0.0;
    return (x < a + 1.0) ? qp::gamma_p_series(a, x, kGammaEps)
                         : 1.0 - qp::gamma_q_cf(a, x, kGammaEps);
}

double regularized_gamma_q(double a, double x) {
    if (!(a > 0.0) || !std::isfinite(a)) {
        throw std::invalid_argument("regularized_gamma_q: a must be finite and > 0");
    }
    if (!(x >= 0.0) || !std::isfinite(x)) {
        throw std::invalid_argument("regularized_gamma_q: x must be finite and >= 0");
    }
    if (x == 0.0) return 1.0;
    return (x < a + 1.0) ? 1.0 - qp::gamma_p_series(a, x, kGammaEps)
                         : qp::gamma_q_cf(a, x, kGammaEps);
}

double chi2_sf(double x, double df) {
    if (!(df > 0.0) || !std::isfinite(df)) {
        throw std::invalid_argument("chi2_sf: df must be finite and > 0");
    }
    if (std::isnan(x)) throw std::invalid_argument("chi2_sf: x is NaN");
    if (x <= 0.0) return 1.0;
    if (std::isinf(x)) return 0.0;
    return regularized_gamma_q(0.5 * df, 0.5 * x);
}

double binomial_cdf(int k, int n, double p) {
    if (n < 1) throw std::invalid_argument("binomial_cdf: n must be >= 1");
    if (!(p >= 0.0 && p <= 1.0)) {
        throw std::invalid_argument("binomial_cdf: p must be in [0, 1]");
    }
    if (k < 0) return 0.0;
    if (k >= n) return 1.0;
    if (p == 0.0) return 1.0;
    if (p == 1.0) return 0.0;
    return qp::binomial_cdf_unchecked(k, n, p, kBetaEps, kBetaMaxIter);
}

// --------------------------------------------------------------------------
// Student-t
// --------------------------------------------------------------------------

double student_t_pdf(double x, double df) {
    if (!(df > 0.0) || !std::isfinite(df) || !std::isfinite(x)) {
        throw std::invalid_argument("student_t_pdf: need finite x and finite df > 0");
    }
    const double ln = log_gamma(0.5 * (df + 1.0)) - log_gamma(0.5 * df) - 0.5 * std::log(df * kPi) -
                      0.5 * (df + 1.0) * std::log1p(x * x / df);
    return std::exp(ln);
}

double student_t_cdf(double x, double df) {
    if (!(df > 0.0) || !std::isfinite(df) || std::isnan(x)) {
        throw std::invalid_argument("student_t_cdf: need non-NaN x and finite df > 0");
    }
    if (std::isinf(x)) return x > 0.0 ? 1.0 : 0.0;
    if (x == 0.0) return 0.5;
    const double half_ib = 0.5 * betainc_reg(0.5 * df, 0.5, df / (df + x * x));
    return (x > 0.0) ? 1.0 - half_ib : half_ib;
}

double student_t_ppf(double p, double df) {
    if (!(p > 0.0 && p < 1.0)) {
        throw std::invalid_argument("student_t_ppf: p must be in (0, 1), got " + std::to_string(p));
    }
    if (!(df > 0.0) || !std::isfinite(df)) {
        throw std::invalid_argument("student_t_ppf: df must be finite and > 0");
    }
    if (p == 0.5) return 0.0;
    // Exploit symmetry: solve for the lower tail and mirror.
    const bool upper = p > 0.5;
    const double pl = upper ? 1.0 - p : p;
    // Bracket the (negative) quantile: expand until CDF(lo) < pl.
    double lo = -1.0, hi = 0.0;
    while (student_t_cdf(lo, df) > pl) {
        hi = lo;
        lo *= 2.0;
        if (lo < -1e12) break;  // pl astronomically small; bisect what we have
    }
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (mid == lo || mid == hi) break;  // machine-precision bracket
        if (student_t_cdf(mid, df) < pl) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const double q = 0.5 * (lo + hi);
    return upper ? -q : q;
}

// --------------------------------------------------------------------------
// Sample moments
// --------------------------------------------------------------------------

namespace {

void require_finite(std::span<const double> x, const char* who) {
    for (double v : x) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument(std::string(who) + ": input contains NaN or inf");
        }
    }
}

void central_moments(std::span<const double> x, double& mu, double& m2, double& m3, double& m4) {
    const auto n = static_cast<double>(x.size());
    mu = 0.0;
    for (double v : x) mu += v;
    mu /= n;
    m2 = m3 = m4 = 0.0;
    for (double v : x) {
        const double d = v - mu;
        const double d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;
}

}  // namespace

double mean(std::span<const double> x) {
    if (x.empty()) throw std::invalid_argument("mean: empty input");
    require_finite(x, "mean");
    double s = 0.0;
    for (double v : x) s += v;
    return s / static_cast<double>(x.size());
}

double stdev(std::span<const double> x) {
    if (x.size() < 2) throw std::invalid_argument("stdev: need at least 2 observations");
    const double mu = mean(x);
    double s = 0.0;
    for (double v : x) s += (v - mu) * (v - mu);
    return std::sqrt(s / static_cast<double>(x.size() - 1));
}

double skewness(std::span<const double> x) {
    if (x.size() < 3) throw std::invalid_argument("skewness: need at least 3 observations");
    require_finite(x, "skewness");
    double mu, m2, m3, m4;
    central_moments(x, mu, m2, m3, m4);
    if (m2 <= 0.0) return 0.0;
    return m3 / std::pow(m2, 1.5);
}

double excess_kurtosis(std::span<const double> x) {
    if (x.size() < 4) throw std::invalid_argument("excess_kurtosis: need at least 4 observations");
    require_finite(x, "excess_kurtosis");
    double mu, m2, m3, m4;
    central_moments(x, mu, m2, m3, m4);
    if (m2 <= 0.0) return 0.0;
    return m4 / (m2 * m2) - 3.0;
}

}  // namespace eqvar
