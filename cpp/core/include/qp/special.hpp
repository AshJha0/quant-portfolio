// qp/special.hpp: special functions used by the VaR engines.
//
// These are the unchecked numerical cores. Each engine keeps its own public
// wrapper, which validates the arguments with its own messages and then
// calls the function here. Where the two VaR engines used different
// stopping constants (series tolerance, iteration cap) the constant is a
// parameter, so each engine keeps its exact numbers.

#pragma once

#include <algorithm>
#include <cmath>

#include "qp/constants.hpp"

namespace qp {

/// ln Gamma(x) for finite x > 0: Lanczos (g = 7, n = 9), about 1e-15
/// relative, with reflection below 0.5. std::lgamma is avoided because glibc
/// writes the global signgam, which is not thread-safe.
[[nodiscard]] inline double log_gamma_lanczos(double x) noexcept {
    if (x < 0.5) return std::log(kPi / std::sin(kPi * x)) - log_gamma_lanczos(1.0 - x);
    static constexpr double c[] = {
        0.99999999999980993,  676.5203681218851,     -1259.1392167224028,
        771.32342877765313,   -176.61502916214059,   12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
    const double z = x - 1.0;
    double a = c[0];
    const double t = z + 7.5;
    for (int i = 1; i < 9; ++i) a += c[i] / (z + static_cast<double>(i));
    return 0.5 * std::log(2.0 * kPi) + (z + 0.5) * std::log(t) - t + std::log(a);
}

/// Continued fraction for the regularised incomplete beta (modified Lentz).
[[nodiscard]] inline double betacf(double a, double b, double x, double eps,
                                   int max_iter) noexcept {
    constexpr double kFpMin = 1e-300;
    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < kFpMin) d = kFpMin;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= max_iter; ++m) {
        const auto dm = static_cast<double>(m);
        const double m2 = 2.0 * dm;
        double aa = dm * (b - dm) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (std::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + dm) * (qab + dm) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::abs(d) < kFpMin) d = kFpMin;
        c = 1.0 + aa / c;
        if (std::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < eps) break;
    }
    return h;
}

/// I_x(a, b) for finite a, b > 0 and x in (0, 1).
[[nodiscard]] inline double betainc_unchecked(double a, double b, double x, double eps,
                                              int max_iter) noexcept {
    const double ln_front = log_gamma_lanczos(a + b) - log_gamma_lanczos(a) - log_gamma_lanczos(b) +
                            a * std::log(x) + b * std::log1p(-x);
    const double front = std::exp(ln_front);
    if (x < (a + 1.0) / (a + b + 2.0)) return front * betacf(a, b, x, eps, max_iter) / a;
    return 1.0 - front * betacf(b, a, 1.0 - x, eps, max_iter) / b;
}

/// Series for the regularised lower incomplete gamma P(a, x); use for x < a + 1.
[[nodiscard]] inline double gamma_p_series(double a, double x, double eps) noexcept {
    double ap = a;
    double sum = 1.0 / a;
    double del = sum;
    for (int n = 0; n < 500; ++n) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (std::abs(del) < std::abs(sum) * eps) break;
    }
    return sum * std::exp(-x + a * std::log(x) - log_gamma_lanczos(a));
}

/// Continued fraction for the regularised upper incomplete gamma Q(a, x);
/// use for x >= a + 1.
[[nodiscard]] inline double gamma_q_cf(double a, double x, double eps) noexcept {
    constexpr double kFpMin = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / kFpMin;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= 500; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < kFpMin) d = kFpMin;
        c = b + an / c;
        if (std::abs(c) < kFpMin) c = kFpMin;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < eps) break;
    }
    return h * std::exp(-x + a * std::log(x) - log_gamma_lanczos(a));
}

/// P(X <= k) for X ~ Binomial(n, p), 0 <= k < n, 0 < p < 1. Sums the pmf
/// recursion pmf(j+1) = pmf(j) (n-j)/(j+1) p/(1-p); when (1-p)^n underflows
/// it uses P(X <= k) = I_{1-p}(n - k, k + 1).
[[nodiscard]] inline double binomial_cdf_unchecked(int k, int n, double p, double eps,
                                                   int max_iter) noexcept {
    double pmf = std::exp(static_cast<double>(n) * std::log1p(-p));
    if (pmf == 0.0) {
        return betainc_unchecked(static_cast<double>(n - k), static_cast<double>(k + 1), 1.0 - p,
                                 eps, max_iter);
    }
    const double odds = p / (1.0 - p);
    double cdf = pmf;
    for (int j = 0; j < k; ++j) {
        pmf *= static_cast<double>(n - j) / static_cast<double>(j + 1) * odds;
        cdf += pmf;
    }
    return std::min(cdf, 1.0);
}

}  // namespace qp
