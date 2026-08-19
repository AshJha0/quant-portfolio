#include "fxopt/deltas.hpp"

#include <cmath>

#include "fxopt/garman_kohlhagen.hpp"
#include "brent.hpp"

namespace fxopt {

namespace {

// Strike maximising K*N(d2(K)) -- the fold point of the PA call delta.
//
// Setting d/dK [K N(d2)] = 0 gives N(d2) sigma sqrt(T) = n(d2), a
// one-dimensional root in d2 (unique: LHS increasing, RHS log-concave with
// a single crossing), then K = F exp(-d2 sigma sqrt(T) - sigma^2 T / 2).
double pa_peak_strike(double F, double T, double sigma) {
    const double v = sigma * std::sqrt(T);
    const auto g = [v](double x) { return norm_cdf(x) * v - norm_pdf(x); };
    const double root = detail::brentq(g, -20.0, 20.0, 1e-14);
    return F * std::exp(-root * v - 0.5 * v * v);
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
    return delta_spot * std::exp(r_f * T);
}

double forward_to_spot_delta(double delta_forward, double T, double r_f) {
    validate_inputs(1.0, 1.0, T, 0.0, r_f, 0.0);
    return delta_forward * std::exp(-r_f * T);
}

double premium_adjust_spot_delta(double delta_spot, double price, double S) {
    if (!(S > 0.0) || !std::isfinite(S)) {
        throw std::invalid_argument("Spot S must be positive and finite, got " +
                                    std::to_string(S));
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
            "delta must have sign " + std::to_string(static_cast<int>(p)) +
            ", got " + std::to_string(target_delta));
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
                "forward-equivalent delta " + std::to_string(fwd_delta) +
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
                "premium-adjusted call delta " + std::to_string(target_delta) +
                " exceeds the maximum attainable " +
                std::to_string(max_delta * scale) +
                " for these market inputs");
        }
        double k_hi = k_peak;
        const double k_max = F * std::exp(30.0 * v);
        while (pa_fwd_delta(k_hi) > fwd_delta && k_hi < k_max) {
            k_hi *= 2.0;
        }
        return detail::brentq(
            [&](double K) { return pa_fwd_delta(K) - fwd_delta; }, k_peak,
            k_hi, 1e-14, 200);
    }

    // put: |PA delta| strictly increasing in K -> unique root
    if (!(fwd_delta > -1.0 && fwd_delta < 0.0)) {
        throw std::invalid_argument(
            "premium-adjusted put forward delta " + std::to_string(fwd_delta) +
            " outside (-1, 0)");
    }
    const double k_lo = F * std::exp(-30.0 * v);
    double k_hi = F;
    const double k_max = F * std::exp(30.0 * v);
    while (pa_fwd_delta(k_hi) > fwd_delta && k_hi < k_max) {
        k_hi *= 2.0;
    }
    return detail::brentq(
        [&](double K) { return pa_fwd_delta(K) - fwd_delta; }, k_lo, k_hi,
        1e-14, 200);
}

}  // namespace fxopt
