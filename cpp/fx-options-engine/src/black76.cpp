#include "fxopt/black76.hpp"

#include "fxopt/forwards.hpp"
#include "fxopt/garman_kohlhagen.hpp"

namespace fxopt {

double black76_price(double F, double K, double T, double r_d, double sigma,
                     OptionType type) {
    return value_or_throw(try_black76_price(F, K, T, r_d, sigma, type));
}

double black76_from_spot(double S, double K, double T, double r_d, double r_f,
                         double sigma, OptionType type) {
    const double F = cip_forward(S, T, r_d, r_f);
    return black76_price(F, K, T, r_d, sigma, type);
}

}  // namespace fxopt
