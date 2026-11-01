// qp/normal.hpp: standard normal pdf, cdf and inverse cdf.
//
// The cdf is 0.5 erfc(-x / sqrt 2), which keeps full relative accuracy in
// the left tail. The inverse is Acklam's rational approximation (relative
// error below 1.15e-9) followed by a fixed number of Halley steps against
// that cdf. The engines differ only in how many Halley steps they take and
// in how they validate p, so those stay in the engines.

#pragma once

#include <cmath>

#include "qp/config.hpp"
#include "qp/constants.hpp"

namespace qp {

/// Standard normal density.
[[nodiscard]] inline double norm_pdf(double x) noexcept {
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

/// Standard normal cdf, 0.5 erfc(-x / sqrt 2).
[[nodiscard]] inline double norm_cdf(double x) noexcept QP_POST(r, !(r < 0.0) && !(r > 1.0)) {
    return 0.5 * std::erfc(-x / kSqrt2);
}

/// The same cdf written as a multiplication by the rounded 1/sqrt(2). It
/// can differ from norm_cdf() in the last bit. The equity options engine
/// has always used this form and its golden values are pinned to it.
[[nodiscard]] inline double norm_cdf_rcp(double x) noexcept
    QP_POST(r, !(r < 0.0) && !(r > 1.0)) {
    return 0.5 * std::erfc(-x * kInvSqrt2);
}

/// Acklam's rational approximation to the inverse normal cdf, without
/// refinement. The caller guarantees 0 < p < 1.
[[nodiscard]] inline double acklam_ppf(double p) noexcept QP_PRE(p > 0.0 && p < 1.0) {
    constexpr double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                             -2.759285104469687e+02, 1.383577518672690e+02,
                             -3.066479806614716e+01, 2.506628277459239e+00};
    constexpr double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                             -1.556989798598866e+02, 6.680131188771972e+01,
                             -1.328068155288572e+01};
    constexpr double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                             -2.400758277161838e+00, -2.549732539343734e+00,
                             4.374664141464968e+00,  2.938163982698783e+00};
    constexpr double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                             2.445134137142996e+00, 3.754408661907416e+00};
    constexpr double p_low = 0.02425;
    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p <= 1.0 - p_low) {
        const double q = p - 0.5;
        const double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
}

/// One Halley step for norm_cdf(x) = p: with u = (N(x) - p) / n(x),
/// x - u / (1 + x u / 2).
[[nodiscard]] inline double ppf_halley_step(double x, double p) noexcept {
    const double e = norm_cdf(x) - p;
    const double u = e / norm_pdf(x);
    return x - u / (1.0 + 0.5 * x * u);
}

/// Inverse normal cdf: Acklam plus `Steps` Halley steps. One step gives
/// about 1e-13 absolute error in x, two give about 1e-15. The caller
/// guarantees 0 < p < 1.
template <int Steps>
[[nodiscard]] inline double norm_ppf_unchecked(double p) noexcept QP_PRE(p > 0.0 && p < 1.0) {
    double x = acklam_ppf(p);
    for (int i = 0; i < Steps; ++i) x = ppf_halley_step(x, p);
    return x;
}

}  // namespace qp
