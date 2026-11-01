#include "eqopt/black76.hpp"

namespace eqopt {

std::pair<double, double> b76_d1_d2(double F, double K, double T,
                                    double sigma) {
    return value_or_throw(try_b76_d1_d2(F, K, T, sigma));
}

double black76_price(double F, double K, double T, double r, double sigma,
                     OptionType type) {
    return value_or_throw(try_black76_price(F, K, T, r, sigma, type));
}

Black76Greeks black76_greeks(double F, double K, double T, double r,
                             double sigma, OptionType type) {
    return value_or_throw(try_black76_greeks(F, K, T, r, sigma, type));
}

}  // namespace eqopt
