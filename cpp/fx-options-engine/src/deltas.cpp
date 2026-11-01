#include "fxopt/deltas.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <utility>

#include "fxopt/garman_kohlhagen.hpp"
#include "brent.hpp"

namespace fxopt {

namespace {

constexpr double kLnSqrt2Pi = 0.9189385332046728;

}  // namespace

// ln N(x) without underflow. Direct for x > -37 (N(-37) is still a normal
// double); below, the Mills-ratio series
// -x^2/2 - ln(-x) - ln sqrt(2 pi) + ln(1 - 1/x^2 + 3/x^4 - 15/x^6 + 105/x^8)
// has absolute error below 945/x^10 < 2.1e-13.
double log_norm_cdf(double x) {
    if (x > -37.0) return std::log(norm_cdf(x));
    const double x2 = 1.0 / (x * x);
    const double series = 1.0 - x2 * (1.0 - x2 * (3.0 - x2 * (15.0 - 105.0 * x2)));
    return -0.5 * x * x - std::log(-x) - kLnSqrt2Pi + std::log(series);
}

// Bracket for the root of h(x) = ln N(x) + ln v - ln n(x), v = sigma sqrt(T).
// h is increasing. Lower end: for x < 0, N(x) < n(x)/|x| (Mills), so
// h(x) < ln(v/|x|) < 0 whenever |x| > v; take x_lo = -(v + 1). Upper end:
// for x >= 0, N(x) >= 1/2, so h(x) > 0 once n(x) < v/2, i.e.
// x > sqrt(2 ln(2 / (v sqrt(2 pi)))); take that plus 1 (at least 1). Both
// ends are valid for every v > 0.
std::pair<double, double> pa_fold_bracket(double v) {
    const double x_lo = -(v + 1.0);
    const double arg = 2.0 / (v * std::sqrt(2.0 * std::numbers::pi));
    const double x_hi = 1.0 + (arg > 1.0 ? std::sqrt(2.0 * std::log(arg)) : 0.0);
    return {x_lo, x_hi};
}

namespace {

// Strike maximising K*N(d2(K)) -- the fold point of the PA call delta.
//
// Setting d/dK [K N(d2)] = 0 gives N(d2) sigma sqrt(T) = n(d2), a
// one-dimensional root in d2 (unique: LHS increasing, RHS log-concave with
// a single crossing), then K = F exp(-d2 sigma sqrt(T) - sigma^2 T / 2).
// Solved in log form on the derived bracket so no term underflows.
double pa_peak_strike(double F, double T, double sigma) {
    const double v = sigma * std::sqrt(T);
    const double log_v = std::log(v);
    const auto h = [log_v](double x) {
        return log_norm_cdf(x) + log_v + 0.5 * x * x + kLnSqrt2Pi;
    };
    const auto [x_lo, x_hi] = pa_fold_bracket(v);
    const double root = detail::brentq(h, x_lo, x_hi, 1e-14);
    return detail::finite_result(F * std::exp(-root * v - 0.5 * v * v), "PA fold strike");
}

}  // namespace

double delta(double S, double K, double T, double r_d, double r_f,
             double sigma, OptionType type, DeltaConvention convention) {
    const double p = phi(type);
    const double d1v = d1(S, K, T, r_d, r_f, sigma);  // validates S,K,T,sigma
    const double d2v = d1v - sigma * std::sqrt(T);
    const double F = S * std::exp((r_d - r_f) * T);
    switch (convention) {
        case DeltaConvention::Spot:
            return p * std::exp(-r_f * T) * norm_cdf(p * d1v);
        case DeltaConvention::Forward:
            return p * norm_cdf(p * d1v);
        case DeltaConvention::SpotPa:
            return p * std::exp(-r_f * T) * (K / F) * norm_cdf(p * d2v);
        case DeltaConvention::ForwardPa:
        default:
            return p * (K / F) * norm_cdf(p * d2v);
    }
}

double spot_to_forward_delta(double delta_spot, double T, double r_f) {
    validate_inputs(1.0, 1.0, T, 0.0, r_f, 0.0);
    detail::require_finite(delta_spot, "delta_spot");
    return detail::finite_result(delta_spot * std::exp(r_f * T), "spot_to_forward_delta");
}

double forward_to_spot_delta(double delta_forward, double T, double r_f) {
    validate_inputs(1.0, 1.0, T, 0.0, r_f, 0.0);
    detail::require_finite(delta_forward, "delta_forward");
    return detail::finite_result(delta_forward * std::exp(-r_f * T), "forward_to_spot_delta");
}

double premium_adjust_spot_delta(double delta_spot, double price, double S) {
    if (!(S > 0.0) || !std::isfinite(S)) {
        throw std::invalid_argument("Spot S must be positive and finite, got " +
                                    std::format("{}", S));
    }
    detail::require_finite(delta_spot, "delta_spot");
    detail::require_finite(price, "price");
    return delta_spot - price / S;
}

double atm_forward_strike(double S, double T, double r_d, double r_f) {
    validate_inputs(S, S, T, r_d, r_f, 0.0);
    return S * std::exp((r_d - r_f) * T);
}

double atm_dns_strike(double S, double T, double r_d, double r_f, double sigma,
                      DeltaConvention convention) {
    validate_inputs(S, S, T, r_d, r_f, sigma);
    const double F = S * std::exp((r_d - r_f) * T);
    const double sign = is_premium_adjusted(convention) ? -1.0 : 1.0;
    return F * std::exp(sign * 0.5 * sigma * sigma * T);
}

double strike_from_delta(double target_delta, double S, double T, double r_d,
                         double r_f, double sigma, OptionType type,
                         DeltaConvention convention) {
    const double p = phi(type);
    validate_inputs(S, S, T, r_d, r_f, sigma);
    if (T <= 0.0 || sigma <= 0.0) {
        throw std::invalid_argument(
            "strike_from_delta requires T > 0 and sigma > 0");
    }
    detail::require_finite(target_delta, "target_delta");
    if (p * target_delta <= 0.0) {
        throw std::invalid_argument(
            "delta must have sign " + std::format("{}", static_cast<int>(p)) +
            ", got " + std::format("{}", target_delta));
    }

    const double F = S * std::exp((r_d - r_f) * T);
    const double v = sigma * std::sqrt(T);

    if (convention == DeltaConvention::Spot ||
        convention == DeltaConvention::Forward) {
        const double fwd_delta = convention == DeltaConvention::Spot
                                     ? target_delta * std::exp(r_f * T)
                                     : target_delta;
        if (!(p * fwd_delta > 0.0 && p * fwd_delta < 1.0)) {
            throw std::invalid_argument(
                "forward-equivalent delta " + std::format("{}", fwd_delta) +
                " outside (0, 1) range");
        }
        const double z = norm_ppf(p * fwd_delta);  // z = phi * d1
        return F * std::exp(-p * z * v + 0.5 * v * v);
    }

    // Premium-adjusted: solve phi (K/F) N(phi d2(K)) = fwd-equivalent delta.
    const double fwd_delta = convention == DeltaConvention::SpotPa
                                 ? target_delta * std::exp(r_f * T)
                                 : target_delta;

    const auto pa_fwd_delta = [F, v, p](double K) {
        const double d2v = (std::log(F / K) - 0.5 * v * v) / v;
        return p * (K / F) * norm_cdf(p * d2v);
    };

    if (p > 0.0) {  // call: non-monotone; use decreasing branch [K_peak, inf)
        const double k_peak = pa_peak_strike(F, T, sigma);
        const double max_delta = pa_fwd_delta(k_peak);
        if (fwd_delta > max_delta + 1e-14) {
            const double scale = convention == DeltaConvention::SpotPa
                                     ? std::exp(-r_f * T)
                                     : 1.0;
            throw std::invalid_argument(
                "premium-adjusted call delta " + std::format("{}", target_delta) +
                " exceeds the maximum attainable " +
                std::format("{}", max_delta * scale) +
                " for these market inputs");
        }
        double k_hi = k_peak;
        const double k_max = F * std::exp(30.0 * v);
        while (pa_fwd_delta(k_hi) > fwd_delta && k_hi < k_max && std::isfinite(2.0 * k_hi)) {
            k_hi *= 2.0;
        }
        return detail::brentq(
            [&](double K) { return pa_fwd_delta(K) - fwd_delta; }, k_peak,
            k_hi, 1e-14, 200);
    }

    // put: |PA delta| strictly increasing in K -> unique root
    if (!(fwd_delta > -1.0 && fwd_delta < 0.0)) {
        throw std::invalid_argument(
            "premium-adjusted put forward delta " + std::format("{}", fwd_delta) +
            " outside (-1, 0)");
    }
    const double k_lo = F * std::exp(-std::min(30.0 * v, 700.0));
    double k_hi = F;
    const double k_max = F * std::exp(30.0 * v);
    while (pa_fwd_delta(k_hi) > fwd_delta && k_hi < k_max && std::isfinite(2.0 * k_hi)) {
        k_hi *= 2.0;
    }
    return detail::brentq(
        [&](double K) { return pa_fwd_delta(K) - fwd_delta; }, k_lo, k_hi,
        1e-14, 200);
}

}  // namespace fxopt
