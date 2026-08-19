#include "eqopt/black76.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "eqopt/black_scholes.hpp"

namespace eqopt {

std::pair<double, double> b76_d1_d2(double F, double K, double T,
                                    double sigma) {
    validate_inputs(F, K, T, sigma);
    if (F <= 0.0 || K <= 0.0 || T <= 0.0 || sigma <= 0.0) {
        std::ostringstream oss;
        oss << "b76_d1_d2 requires strictly positive F, K, T, sigma; got F="
            << F << ", K=" << K << ", T=" << T << ", sigma=" << sigma;
        throw std::invalid_argument(oss.str());
    }
    const double sqrt_t = std::sqrt(T);
    const double d1 =
        (std::log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_t);
    return {d1, d1 - sigma * sqrt_t};
}

double black76_price(double F, double K, double T, double r, double sigma,
                     OptionType type) {
    validate_inputs(F, K, T, sigma);
    validate_rates(r);
    const double sign = type == OptionType::Call ? 1.0 : -1.0;
    if (T == 0.0) {
        return std::fmax(sign * (F - K), 0.0);
    }
    const double df = std::exp(-r * T);
    if (sigma == 0.0 || K == 0.0 || F == 0.0) {
        return df * std::fmax(sign * (F - K), 0.0);
    }
    const auto [d1, d2] = b76_d1_d2(F, K, T, sigma);
    return df * sign * (F * norm_cdf(sign * d1) - K * norm_cdf(sign * d2));
}

Black76Greeks black76_greeks(double F, double K, double T, double r,
                             double sigma, OptionType type) {
    validate_rates(r);
    const auto [d1, d2] = b76_d1_d2(F, K, T, sigma);
    const double df = std::exp(-r * T);
    const double sqrt_t = std::sqrt(T);
    const double pdf_d1 = norm_pdf(d1);
    const double sign = type == OptionType::Call ? 1.0 : -1.0;

    Black76Greeks g{};
    g.price = df * sign * (F * norm_cdf(sign * d1) - K * norm_cdf(sign * d2));
    g.delta = df * sign * norm_cdf(sign * d1);
    g.gamma = df * pdf_d1 / (F * sigma * sqrt_t);
    g.vega = df * F * pdf_d1 * sqrt_t;
    // theta = dV/dt at fixed F: r*V - decay of the time value.
    g.theta = r * g.price - df * F * pdf_d1 * sigma / (2.0 * sqrt_t);
    g.rho = -T * g.price;
    return g;
}

}  // namespace eqopt
