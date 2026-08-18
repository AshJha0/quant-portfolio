#include "eqvar/stats.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace eqvar {

namespace {
constexpr double kSqrt2 = 1.4142135623730950488016887242097;
constexpr double kInvSqrt2Pi = 0.39894228040143267793994605993438;
constexpr double kPi = 3.1415926535897932384626433832795;
}  // namespace

// --------------------------------------------------------------------------
// Normal
// --------------------------------------------------------------------------

double normal_pdf(double x) noexcept { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

double normal_cdf(double x) noexcept { return 0.5 * std::erfc(-x / kSqrt2); }

double normal_ppf(double p) {
    if (!(p > 0.0 && p < 1.0)) {
        throw std::invalid_argument("normal_ppf: p must be in (0, 1), got " + std::to_string(p));
    }
    // Route the upper half through the lower tail: 1 - p is EXACT for doubles
    // p in [0.5, 1) (Sterbenz), and the erfc-based CDF used by the Halley
    // refinement is accurate in the lower tail but suffers cancellation near 1.
    // This also makes Phi^{-1}(p) = -Phi^{-1}(1-p) hold bitwise.
    if (p > 0.5) return -normal_ppf(1.0 - p);
    // Acklam's rational approximation (2003), |relative error| < 1.15e-9.
    static constexpr double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                   -2.759285104469687e+02, 1.383577518672690e+02,
                                   -3.066479806614716e+01, 2.506628277459239e+00};
    static constexpr double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                   -1.556989798598866e+02, 6.680131188771972e+01,
                                   -1.328068155288572e+01};
    static constexpr double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                   -2.400758277161838e+00, -2.549732539343734e+00,
                                   4.374664141464968e+00,  2.938163982698783e+00};
    static constexpr double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                                   2.445134137142996e+00, 3.754408661907416e+00};
    constexpr double p_low = 0.02425;

    double x;
    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 1.0 - p_low) {
        const double q = p - 0.5;
        const double r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    // One Halley refinement step drives the error to ~machine precision.
    const double e = normal_cdf(x) - p;
    const double u = e / normal_pdf(x);
    x -= u / (1.0 + 0.5 * x * u);
    return x;
}

// --------------------------------------------------------------------------
// Regularized incomplete beta (continued fraction, modified Lentz)
// --------------------------------------------------------------------------

namespace {

double betacf(double a, double b, double x) {
    constexpr int kMaxIter = 300;
    constexpr double kEps = 3e-16;
    constexpr double kFpMin = 1e-300;
    const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < kFpMin) d = kFpMin;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= kMaxIter; ++m) {
        const double m2 = 2.0 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (std::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (std::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < kEps) return h;
    }
    return h;  // converged to working precision in practice
}

}  // namespace

double betainc_reg(double a, double b, double x) {
    if (!(a > 0.0) || !(b > 0.0)) {
        throw std::invalid_argument("betainc_reg: a and b must be > 0");
    }
    if (!(x >= 0.0 && x <= 1.0)) {
        throw std::invalid_argument("betainc_reg: x must be in [0, 1], got " + std::to_string(x));
    }
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;
    const double ln_front = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
                            a * std::log(x) + b * std::log1p(-x);
    const double front = std::exp(ln_front);
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return front * betacf(a, b, x) / a;
    }
    return 1.0 - front * betacf(b, a, 1.0 - x) / b;
}

// --------------------------------------------------------------------------
// Regularized incomplete gamma
// --------------------------------------------------------------------------

namespace {

/// Series representation of P(a, x), valid/fast for x < a + 1.
double gamma_p_series(double a, double x) {
    constexpr int kMaxIter = 500;
    constexpr double kEps = 3e-16;
    double ap = a;
    double sum = 1.0 / a;
    double del = sum;
    for (int n = 0; n < kMaxIter; ++n) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (std::abs(del) < std::abs(sum) * kEps) break;
    }
    return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

/// Continued-fraction representation of Q(a, x), valid/fast for x >= a + 1.
double gamma_q_cf(double a, double x) {
    constexpr int kMaxIter = 500;
    constexpr double kEps = 3e-16;
    constexpr double kFpMin = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / kFpMin;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= kMaxIter; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < kFpMin) d = kFpMin;
        c = b + an / c;
        if (std::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < kEps) break;
    }
    return h * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

}  // namespace

double regularized_gamma_p(double a, double x) {
    if (!(a > 0.0)) throw std::invalid_argument("regularized_gamma_p: a must be > 0");
    if (!(x >= 0.0)) throw std::invalid_argument("regularized_gamma_p: x must be >= 0");
    if (x == 0.0) return 0.0;
    return (x < a + 1.0) ? gamma_p_series(a, x) : 1.0 - gamma_q_cf(a, x);
}

double regularized_gamma_q(double a, double x) {
    if (!(a > 0.0)) throw std::invalid_argument("regularized_gamma_q: a must be > 0");
    if (!(x >= 0.0)) throw std::invalid_argument("regularized_gamma_q: x must be >= 0");
    if (x == 0.0) return 1.0;
    return (x < a + 1.0) ? 1.0 - gamma_p_series(a, x) : gamma_q_cf(a, x);
}

double chi2_sf(double x, double df) {
    if (!(df > 0.0)) throw std::invalid_argument("chi2_sf: df must be > 0");
    if (x <= 0.0) return 1.0;
    return regularized_gamma_q(0.5 * df, 0.5 * x);
}

double binomial_cdf(int k, int n, double p) {
    if (n < 1) throw std::invalid_argument("binomial_cdf: n must be >= 1");
    if (!(p >= 0.0 && p <= 1.0)) {
        throw std::invalid_argument("binomial_cdf: p must be in [0, 1]");
    }
    if (k < 0) return 0.0;
    if (k >= n) return 1.0;
    // P(X <= k) = I_{1-p}(n - k, k + 1).
    return betainc_reg(static_cast<double>(n - k), static_cast<double>(k + 1), 1.0 - p);
}

// --------------------------------------------------------------------------
// Student-t
// --------------------------------------------------------------------------

double student_t_pdf(double x, double df) {
    if (!(df > 0.0)) throw std::invalid_argument("student_t_pdf: df must be > 0");
    const double ln = std::lgamma(0.5 * (df + 1.0)) - std::lgamma(0.5 * df) -
                      0.5 * std::log(df * kPi) -
                      0.5 * (df + 1.0) * std::log1p(x * x / df);
    return std::exp(ln);
}

double student_t_cdf(double x, double df) {
    if (!(df > 0.0)) throw std::invalid_argument("student_t_cdf: df must be > 0");
    if (x == 0.0) return 0.5;
    const double half_ib = 0.5 * betainc_reg(0.5 * df, 0.5, df / (df + x * x));
    return (x > 0.0) ? 1.0 - half_ib : half_ib;
}

double student_t_ppf(double p, double df) {
    if (!(p > 0.0 && p < 1.0)) {
        throw std::invalid_argument("student_t_ppf: p must be in (0, 1), got " +
                                    std::to_string(p));
    }
    if (!(df > 0.0)) throw std::invalid_argument("student_t_ppf: df must be > 0");
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

void central_moments(std::span<const double> x, double& mu, double& m2, double& m3, double& m4) {
    const double n = static_cast<double>(x.size());
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
    double mu, m2, m3, m4;
    central_moments(x, mu, m2, m3, m4);
    if (m2 <= 0.0) return 0.0;
    return m3 / std::pow(m2, 1.5);
}

double excess_kurtosis(std::span<const double> x) {
    if (x.size() < 4) throw std::invalid_argument("excess_kurtosis: need at least 4 observations");
    double mu, m2, m3, m4;
    central_moments(x, mu, m2, m3, m4);
    if (m2 <= 0.0) return 0.0;
    return m4 / (m2 * m2) - 3.0;
}

}  // namespace eqvar
