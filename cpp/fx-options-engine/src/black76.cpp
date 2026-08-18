#include "fxopt/black76.hpp"

#include <algorithm>
#include <cmath>

#include "fxopt/forwards.hpp"
#include "fxopt/garman_kohlhagen.hpp"

namespace fxopt {

double black76_price(double F, double K, double T, double r_d, double sigma,
                     OptionType type) {
    const double p = phi(type);
    detail::require_finite(F, "F");
    detail::require_finite(K, "K");
    detail::require_finite(T, "T");
    detail::require_finite(r_d, "r_d");
    detail::require_finite(sigma, "sigma");
    if (F <= 0.0) {
        throw std::invalid_argument("Forward F must be positive, got " +
                                    std::to_string(F));
    }
    if (K <= 0.0) {
        throw std::invalid_argument("Strike K must be positive, got " +
                                    std::to_string(K));
    }
    if (T < 0.0) {
        throw std::invalid_argument(
            "Time to expiry T must be non-negative, got " + std::to_string(T));
    }
    if (sigma < 0.0) {
        throw std::invalid_argument(
            "Volatility sigma must be non-negative, got " +
            std::to_string(sigma));
    }

    const double df = std::exp(-r_d * T);
    const double v = sigma * std::sqrt(T);
    if (T == 0.0 || v <= kMinVol) {
        return df * std::max(p * (F - K), 0.0);
    }
    const double d1v = (std::log(F / K) + 0.5 * v * v) / v;
    const double d2v = d1v - v;
    return p * df * (F * norm_cdf(p * d1v) - K * norm_cdf(p * d2v));
}

double black76_from_spot(double S, double K, double T, double r_d, double r_f,
                         double sigma, OptionType type) {
    const double F = cip_forward(S, T, r_d, r_f);
    return black76_price(F, K, T, r_d, sigma, type);
}

}  // namespace fxopt
