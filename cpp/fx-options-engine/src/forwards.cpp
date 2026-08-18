#include "fxopt/forwards.hpp"

#include <cmath>

namespace fxopt {

double cip_forward(double S, double T, double r_d, double r_f) {
    validate_inputs(S, S, T, r_d, r_f, 0.0);
    return S * std::exp((r_d - r_f) * T);
}

double forward_points(double S, double T, double r_d, double r_f,
                      double pip_factor) {
    if (!(pip_factor > 0.0)) {
        throw std::invalid_argument("pip_factor must be positive, got " +
                                    std::to_string(pip_factor));
    }
    return (cip_forward(S, T, r_d, r_f) - S) * pip_factor;
}

double synthetic_forward_from_options(double call_price, double put_price,
                                      double K, double T, double r_d) {
    validate_inputs(K, K, T, r_d, 0.0, 0.0);
    detail::require_finite(call_price, "call_price");
    detail::require_finite(put_price, "put_price");
    return K + (call_price - put_price) * std::exp(r_d * T);
}

}  // namespace fxopt
