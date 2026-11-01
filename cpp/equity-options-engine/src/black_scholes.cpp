#include "eqopt/black_scholes.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace eqopt {

namespace {

[[noreturn]] void throw_bad_input(const char* name, double value) {
    std::ostringstream oss;
    oss << name << " must be finite and >= 0, got " << value;
    throw std::invalid_argument(oss.str());
}

}  // namespace

void validate_inputs(double S, double K, double T, double sigma) {
    if (!std::isfinite(S) || S < 0.0) throw_bad_input("S", S);
    if (!std::isfinite(K) || K < 0.0) throw_bad_input("K", K);
    if (!std::isfinite(T) || T < 0.0) throw_bad_input("T", T);
    if (!std::isfinite(sigma) || sigma < 0.0) throw_bad_input("sigma", sigma);
}

void validate_rates(double r, double q) {
    if (!std::isfinite(r) || !std::isfinite(q)) {
        std::ostringstream oss;
        oss << "r and q must be finite (NaN/Inf rejected), got r=" << r
            << ", q=" << q;
        throw std::invalid_argument(oss.str());
    }
}

std::pair<double, double> d1_d2(double S, double K, double T, double r,
                                double sigma, double q) {
    validate_inputs(S, K, T, sigma);
    validate_rates(r, q);
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
    validate_rates(r, q);
    if (T == 0.0) {
        return intrinsic_value(S, K, type);
    }
    const double disc_s = S * std::exp(-q * T);
    const double disc_k = K * std::exp(-r * T);
    double value;
    if (K == 0.0) {
        // Zero-strike call is a (dividend-adjusted) forward on the stock.
        value = type == OptionType::Call ? disc_s : 0.0;
    } else if (S == 0.0) {
        value = type == OptionType::Call ? 0.0 : disc_k;
    } else if (sigma == 0.0) {
        // e^{-rT} max(+-(F - K), 0) as S e^{-qT} - K e^{-rT}: the forward
        // itself overflows when (r - q) T > 709.
        const double sign = type == OptionType::Call ? 1.0 : -1.0;
        value = std::fmax(sign * (disc_s - disc_k), 0.0);
    } else {
        const auto [d1, d2] = d1_d2(S, K, T, r, sigma, q);
        value = type == OptionType::Call ? disc_s * norm_cdf(d1) - disc_k * norm_cdf(d2)
                                         : disc_k * norm_cdf(-d2) - disc_s * norm_cdf(-d1);
    }
    return require_finite_result(value, "bs_price");
}

double require_finite_result(double value, const char* what) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(
            std::string(what) +
            ": result is not representable as a finite double (a discount factor or forward overflows)");
    }
    return value;
}

}  // namespace eqopt
