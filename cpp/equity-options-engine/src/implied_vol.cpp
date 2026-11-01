#include "eqopt/implied_vol.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "eqopt/black_scholes.hpp"
#include "eqopt/iv_core.hpp"

namespace eqopt {

double implied_vol(double price, double S, double K, double T, double r,
                   double q, OptionType type) {
    validate_inputs(S, K, T, 0.0);
    validate_rates(r, q);
    if (!std::isfinite(price)) {
        throw std::invalid_argument("price must be finite");
    }
    if (T <= 0.0) {
        throw std::invalid_argument(
            "implied_vol requires T > 0 (option already expired)");
    }
    if (S <= 0.0 || K <= 0.0) {
        throw std::invalid_argument("implied_vol requires S > 0 and K > 0");
    }

    // No-arbitrage bounds: sigma -> 0 (discounted forward intrinsic) and
    // sigma -> inf.
    const double lower = bs_price(S, K, T, r, 0.0, q, type);
    const double df = require_finite_result(std::exp(-r * T), "implied_vol");
    const double upper = type == OptionType::Call
                             ? require_finite_result(S * std::exp(-q * T), "implied_vol")
                             : K * df;
    if (price <= lower) {
        std::ostringstream oss;
        oss << "price " << price
            << " is at or below the sigma->0 arbitrage bound " << lower
            << "; implied vol is undefined";
        throw std::invalid_argument(oss.str());
    }
    if (price >= upper) {
        std::ostringstream oss;
        oss << "price " << price << " is at or above the sigma->inf bound "
            << upper << "; implied vol is undefined";
        throw std::invalid_argument(oss.str());
    }
    const double forward = require_finite_result(S * std::exp((r - q) * T), "implied_vol");
    int evals = 0;
    return iv_core::black_implied_vol(price, forward, K, T, df, type == OptionType::Call, evals);
}

double implied_vol(double price, double S, double K, double T, double r,
                   double q, OptionType type, double /*tol*/, int /*max_iter*/,
                   double /*sigma_lo*/, double /*sigma_hi*/) {
    return implied_vol(price, S, K, T, r, q, type);
}

}  // namespace eqopt
