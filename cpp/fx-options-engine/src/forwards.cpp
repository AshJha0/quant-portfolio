#include "fxopt/forwards.hpp"

#include <cmath>
#include <format>
#include <stdexcept>

namespace fxopt {

double cip_forward(double S, double T, double r_d, double r_f) {
    validate_inputs(S, S, T, r_d, r_f, 0.0);
    return detail::finite_result(S * std::exp((r_d - r_f) * T), "cip_forward");
}

double forward_points(double S, double T, double r_d, double r_f,
                      double pip_factor) {
    if (!(pip_factor > 0.0) || !std::isfinite(pip_factor)) {
        throw std::invalid_argument(
            std::format("pip_factor must be positive and finite, got {}", pip_factor));
    }
    return detail::finite_result((cip_forward(S, T, r_d, r_f) - S) * pip_factor,
                                 "forward_points");
}

double synthetic_forward_from_options(double call_price, double put_price,
                                      double K, double T, double r_d) {
    validate_inputs(K, K, T, r_d, 0.0, 0.0);
    detail::require_finite(call_price, "call_price");
    detail::require_finite(put_price, "put_price");
    return detail::finite_result(K + (call_price - put_price) * std::exp(r_d * T),
                                 "synthetic_forward_from_options");
}

}  // namespace fxopt
