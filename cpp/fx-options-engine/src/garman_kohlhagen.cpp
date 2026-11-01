#include "fxopt/garman_kohlhagen.hpp"

#include <algorithm>
#include <cmath>
#include <format>

namespace fxopt {

double d1(double S, double K, double T, double r_d, double r_f, double sigma) {
    validate_inputs(S, K, T, r_d, r_f, sigma);
    const double vol_sqrt_t = sigma * std::sqrt(T);
    if (vol_sqrt_t <= kMinVol) {
        throw std::invalid_argument(
            "d1 undefined for sigma*sqrt(T)=" + std::format("{}", vol_sqrt_t) +
            "; need sigma>0, T>0");
    }
    return (std::log(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) /
           vol_sqrt_t;
}

double d2(double S, double K, double T, double r_d, double r_f, double sigma) {
    return d1(S, K, T, r_d, r_f, sigma) - sigma * std::sqrt(T);
}

double gk_price(double S, double K, double T, double r_d, double r_f,
                double sigma, OptionType type) {
    const double p = phi(type);
    validate_inputs(S, K, T, r_d, r_f, sigma);
    if (T == 0.0) {
        return std::max(p * (S - K), 0.0);
    }
    const double sqrt_t = std::sqrt(T);
    const double vol_sqrt_t = sigma * sqrt_t;
    const double disc_s = S * std::exp(-r_f * T);
    const double disc_k = K * std::exp(-r_d * T);
    if (vol_sqrt_t <= kMinVol) {
        // e^{-r_d T} max(phi (F - K), 0) as S e^{-r_f T} - K e^{-r_d T}: the
        // forward itself overflows when (r_d - r_f) T > 709.
        return detail::finite_result(std::max(p * (disc_s - disc_k), 0.0), "gk_price");
    }
    const double d1v =
        (std::log(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / vol_sqrt_t;
    const double d2v = d1v - vol_sqrt_t;
    return detail::finite_result(
        p * (disc_s * norm_cdf(p * d1v) - disc_k * norm_cdf(p * d2v)), "gk_price");
}

double gk_call(double S, double K, double T, double r_d, double r_f,
               double sigma) {
    return gk_price(S, K, T, r_d, r_f, sigma, OptionType::Call);
}

double gk_put(double S, double K, double T, double r_d, double r_f,
              double sigma) {
    return gk_price(S, K, T, r_d, r_f, sigma, OptionType::Put);
}

}  // namespace fxopt
