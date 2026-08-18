#include "eqopt/black_scholes.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace eqopt {

namespace {

[[noreturn]] void throw_bad_input(const char* name, double value) {
    std::ostringstream oss;
    oss << name << " must be >= 0 and not NaN, got " << value;
    throw std::invalid_argument(oss.str());
}

}  // namespace

void validate_inputs(double S, double K, double T, double sigma) {
    if (std::isnan(S) || S < 0.0) throw_bad_input("S", S);
    if (std::isnan(K) || K < 0.0) throw_bad_input("K", K);
    if (std::isnan(T) || T < 0.0) throw_bad_input("T", T);
    if (std::isnan(sigma) || sigma < 0.0) throw_bad_input("sigma", sigma);
}

std::pair<double, double> d1_d2(double S, double K, double T, double r,
                                double sigma, double q) {
    validate_inputs(S, K, T, sigma);
    if (S <= 0.0 || K <= 0.0 || T <= 0.0 || sigma <= 0.0) {
        std::ostringstream oss;
        oss << "d1/d2 require strictly positive S, K, T and sigma; got S=" << S
            << ", K=" << K << ", T=" << T << ", sigma=" << sigma;
        throw std::invalid_argument(oss.str());
    }
    const double sqrt_t = std::sqrt(T);
    const double d1 =
        (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) /
        (sigma * sqrt_t);
    return {d1, d1 - sigma * sqrt_t};
}

double bs_price(double S, double K, double T, double r, double sigma,
                double q, OptionType type) {
    validate_inputs(S, K, T, sigma);
    if (T == 0.0) {
        return intrinsic_value(S, K, type);
    }
    if (K == 0.0) {
        // Zero-strike call is a (dividend-adjusted) forward on the stock.
        return type == OptionType::Call ? S * std::exp(-q * T) : 0.0;
    }
    if (S == 0.0) {
        return type == OptionType::Call ? 0.0 : K * std::exp(-r * T);
    }
    if (sigma == 0.0) {
        const double forward = forward_price(S, T, r, q);
        const double sign = type == OptionType::Call ? 1.0 : -1.0;
        return std::exp(-r * T) * std::fmax(sign * (forward - K), 0.0);
    }

    const auto [d1, d2] = d1_d2(S, K, T, r, sigma, q);
    const double disc_s = S * std::exp(-q * T);
    const double disc_k = K * std::exp(-r * T);
    if (type == OptionType::Call) {
        return disc_s * norm_cdf(d1) - disc_k * norm_cdf(d2);
    }
    return disc_k * norm_cdf(-d2) - disc_s * norm_cdf(-d1);
}

}  // namespace eqopt
