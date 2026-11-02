#include "fxopt/garman_kohlhagen.hpp"

#include <cmath>

namespace fxopt {

double d1(double S, double K, double T, double r_d, double r_f, double sigma) {
    return value_or_throw(try_d1(S, K, T, r_d, r_f, sigma));
}

double d2(double S, double K, double T, double r_d, double r_f, double sigma) {
    return d1(S, K, T, r_d, r_f, sigma) - sigma * std::sqrt(T);
}

double gk_price(double S, double K, double T, double r_d, double r_f, double sigma,
                OptionType type) {
    return value_or_throw(try_gk_price(S, K, T, r_d, r_f, sigma, type));
}

double gk_call(double S, double K, double T, double r_d, double r_f, double sigma) {
    return gk_price(S, K, T, r_d, r_f, sigma, OptionType::Call);
}

double gk_put(double S, double K, double T, double r_d, double r_f, double sigma) {
    return gk_price(S, K, T, r_d, r_f, sigma, OptionType::Put);
}

}  // namespace fxopt
