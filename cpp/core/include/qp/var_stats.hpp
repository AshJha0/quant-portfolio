// qp/var_stats.hpp: the checked special functions and sample moments shared
// by the two VaR engines (plan W20, Round 8).
//
// Before Round 8, eqvar/stats.cpp and fxvar/stats.cpp wrapped the unchecked
// cores in qp/special.hpp with different stopping constants and two
// different quantile solvers. This header is the one checked layer; both
// engines' stats.cpp files are now thin wrappers that keep their own names.
// It takes the equity choices, which were at least as accurate in every row
// of the W20 characterisation corpus:
//
//   * series / continued-fraction tolerance 3e-16 (about one ulp at 1; the
//     FX value 1e-16 sat below half an ulp, so the loops ran to the cap);
//   * normal_ppf solves p > 0.5 as -normal_ppf(1 - p). 1 - p is exact for
//     p in [0.5, 1) (Sterbenz), and the Halley step's CDF is accurate in the
//     lower tail, so the upper tail no longer loses digits (FX was 2.2e-12
//     relative off at p = 0.999999);
//   * student_t_ppf bisects the lower-tail CDF down to adjacent doubles.
//
// Worked example: qp::var::normal_ppf(0.01) is -2.32634787404084 and
// qp::var::chi2_sf(3.84, 1) is 0.0500435 (the Kupiec 95% critical value).
//
// All functions are pure; invalid inputs throw std::invalid_argument.

#pragma once

#include <cmath>
#include <span>
#include <stdexcept>
#include <string>

#include "qp/constants.hpp"
#include "qp/normal.hpp"
#include "qp/special.hpp"

namespace qp::var {

inline constexpr double kSeriesEps = 3e-16;
inline constexpr int kBetaMaxIter = 300;

[[nodiscard]] inline double log_gamma(double x) {
    if (!(x > 0.0) || !std::isfinite(x)) {
        throw std::invalid_argument("log_gamma: x must be finite and > 0");
    }
    return log_gamma_lanczos(x);
}

[[nodiscard]] inline double normal_ppf(double p) {
    if (!(p > 0.0 && p < 1.0)) {
        throw std::invalid_argument("normal_ppf: p must be in (0, 1), got " + std::to_string(p));
    }
    if (p > 0.5) return -normal_ppf(1.0 - p);
    return norm_ppf_unchecked<1>(p);
}

[[nodiscard]] inline double betainc_reg(double a, double b, double x) {
    if (!(a > 0.0) || !(b > 0.0) || !std::isfinite(a) || !std::isfinite(b)) {
        throw std::invalid_argument("betainc_reg: a and b must be finite and > 0");
    }
    if (!(x >= 0.0 && x <= 1.0)) {
        throw std::invalid_argument("betainc_reg: x must be in [0, 1], got " + std::to_string(x));
    }
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;
    return betainc_unchecked(a, b, x, kSeriesEps, kBetaMaxIter);
}

inline void check_gamma_args(double a, double x, const char* who) {
    if (!(a > 0.0) || !std::isfinite(a)) {
        throw std::invalid_argument(std::string(who) + ": a must be finite and > 0");
    }
    if (!(x >= 0.0) || !std::isfinite(x)) {
        throw std::invalid_argument(std::string(who) + ": x must be finite and >= 0");
    }
}

[[nodiscard]] inline double regularized_gamma_p(double a, double x) {
    check_gamma_args(a, x, "regularized_gamma_p");
    if (x == 0.0) return 0.0;
    return (x < a + 1.0) ? gamma_p_series(a, x, kSeriesEps) : 1.0 - gamma_q_cf(a, x, kSeriesEps);
}

[[nodiscard]] inline double regularized_gamma_q(double a, double x) {
    check_gamma_args(a, x, "regularized_gamma_q");
    if (x == 0.0) return 1.0;
    return (x < a + 1.0) ? 1.0 - gamma_p_series(a, x, kSeriesEps) : gamma_q_cf(a, x, kSeriesEps);
}

[[nodiscard]] inline double chi2_sf(double x, double df) {
    if (!(df > 0.0) || !std::isfinite(df)) {
        throw std::invalid_argument("chi2_sf: df must be finite and > 0");
    }
    if (std::isnan(x)) throw std::invalid_argument("chi2_sf: x is NaN");
    if (x <= 0.0) return 1.0;
    if (std::isinf(x)) return 0.0;
    return regularized_gamma_q(0.5 * df, 0.5 * x);
}

/// P(X <= k), X ~ Binomial(n, p): k < 0 gives 0, k >= n gives 1.
[[nodiscard]] inline double binomial_cdf(int k, int n, double p) {
    if (n < 0) throw std::invalid_argument("binomial_cdf: n must be >= 0");
    if (!(p >= 0.0 && p <= 1.0)) throw std::invalid_argument("binomial_cdf: p must be in [0, 1]");
    if (k < 0) return 0.0;
    if (k >= n) return 1.0;
    if (p == 0.0) return 1.0;
    if (p == 1.0) return 0.0;
    return binomial_cdf_unchecked(k, n, p, kSeriesEps, kBetaMaxIter);
}

[[nodiscard]] inline double student_t_pdf(double x, double df) {
    if (!(df > 0.0) || !std::isfinite(df) || !std::isfinite(x)) {
        throw std::invalid_argument("student_t_pdf: need finite x and finite df > 0");
    }
    const double ln = log_gamma(0.5 * (df + 1.0)) - log_gamma(0.5 * df) - 0.5 * std::log(df * kPi) -
                      0.5 * (df + 1.0) * std::log1p(x * x / df);
    return std::exp(ln);
}

[[nodiscard]] inline double student_t_cdf(double x, double df) {
    if (!(df > 0.0) || !std::isfinite(df) || std::isnan(x)) {
        throw std::invalid_argument("student_t_cdf: need non-NaN x and finite df > 0");
    }
    if (std::isinf(x)) return x > 0.0 ? 1.0 : 0.0;
    if (x == 0.0) return 0.5;
    const double half_ib = 0.5 * betainc_reg(0.5 * df, 0.5, df / (df + x * x));
    return (x > 0.0) ? 1.0 - half_ib : half_ib;
}

/// Student-t quantile: bisection on the lower-tail CDF to adjacent doubles,
/// mirrored for p > 0.5.
[[nodiscard]] inline double student_t_ppf(double p, double df) {
    if (!(p > 0.0 && p < 1.0)) {
        throw std::invalid_argument("student_t_ppf: p must be in (0, 1), got " + std::to_string(p));
    }
    if (!(df > 0.0) || !std::isfinite(df)) {
        throw std::invalid_argument("student_t_ppf: df must be finite and > 0");
    }
    if (p == 0.5) return 0.0;
    const bool upper = p > 0.5;
    const double pl = upper ? 1.0 - p : p;
    double lo = -1.0, hi = 0.0;
    while (student_t_cdf(lo, df) > pl) {
        hi = lo;
        lo *= 2.0;
        if (lo < -1e12) break;  // pl astronomically small; bisect what we have
    }
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (mid == lo || mid == hi) break;  // adjacent doubles
        if (student_t_cdf(mid, df) < pl) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const double q = 0.5 * (lo + hi);
    return upper ? -q : q;
}

inline void require_finite_sample(std::span<const double> x, const char* who) {
    for (double v : x) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument(std::string(who) + ": input contains NaN or inf");
        }
    }
}

[[nodiscard]] inline double mean(std::span<const double> x) {
    if (x.empty()) throw std::invalid_argument("mean: empty input");
    require_finite_sample(x, "mean");
    double s = 0.0;
    for (double v : x) s += v;
    return s / static_cast<double>(x.size());
}

/// Sample moments: ddof = 1 standard deviation; skewness and excess
/// kurtosis as biased moment ratios (scipy bias=True), 0 at zero variance.
struct Moments {
    double mean = 0.0;
    double stdev = 0.0;
    double skewness = 0.0;
    double excess_kurtosis = 0.0;
};

[[nodiscard]] inline Moments moments(std::span<const double> x) {
    if (x.size() < 2) throw std::invalid_argument("moments: need at least 2 observations");
    require_finite_sample(x, "moments");
    const auto n = static_cast<double>(x.size());
    Moments out;
    for (double v : x) out.mean += v;
    out.mean /= n;
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (double v : x) {
        const double d = v - out.mean;
        const double d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    out.stdev = std::sqrt(m2 / (n - 1.0));
    m2 /= n;
    m3 /= n;
    m4 /= n;
    if (m2 > 0.0) {
        out.skewness = m3 / std::pow(m2, 1.5);
        out.excess_kurtosis = m4 / (m2 * m2) - 3.0;
    }
    return out;
}

}  // namespace qp::var
