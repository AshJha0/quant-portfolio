#include "fxopt/implied_vol.hpp"

#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>

#include "fxopt/iv_core.hpp"

namespace fxopt {

double implied_vol(double price, double S, double K, double T, double r_d,
                   double r_f, OptionType type) {
    const double p = phi(type);
    validate_inputs(S, K, T, r_d, r_f, 0.0);
    if (T <= 0.0) {
        throw std::invalid_argument("implied_vol requires T > 0");
    }
    detail::require_finite(price, "price");

    const double df_d = detail::finite_result(std::exp(-r_d * T), "implied_vol");
    const double df_f = detail::finite_result(std::exp(-r_f * T), "implied_vol");
    const double forward = detail::finite_result(S * df_f / df_d, "forward");
    const double lower = df_d * std::max(p * (forward - K), 0.0);  // sigma->0
    const double upper = p > 0.0 ? S * df_f : K * df_d;            // sigma->inf
    constexpr double eps = std::numeric_limits<double>::epsilon();
    // Rounding of the sigma -> 0 bound df (F - K): a few eps of df max(F, K)
    // when in the money; exactly zero otherwise. Same rule in all engines.
    const double tau_lo = lower > 0.0 ? 4.0 * eps * df_d * std::max(forward, K) : 0.0;
    const double tau = 4.0 * eps * upper;
    if (price < lower - tau_lo || price > upper + tau) {
        throw std::invalid_argument(std::format(
            "price {} outside no-arbitrage bounds [{}, {}]", price, lower, upper));
    }
    if (price - lower <= tau_lo) {
        // Time value below the rounding of the bound: the sigma -> 0 limit.
        return 0.0;
    }
    if (upper - price <= tau) {
        throw std::invalid_argument(std::format(
            "price {} is within {} of the sigma->inf bound {}; no finite implied "
            "volatility is identifiable", price, tau, upper));
    }
    int evals = 0;
    return iv_core::black_implied_vol(price, forward, K, T, df_d, p > 0.0, evals);
}

double implied_vol(double price, double S, double K, double T, double r_d,
                   double r_f, OptionType type, double /*tol*/, int /*max_iter*/) {
    return implied_vol(price, S, K, T, r_d, r_f, type);
}

}  // namespace fxopt
