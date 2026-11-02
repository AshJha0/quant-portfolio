// Charm, speed, colour, zomma and veta for Black-Scholes-Merton (port of
// eq_options.greeks_higher). Haug (2007), "The Complete Guide to Option
// Pricing Formulas", 2nd ed., sections 2.2-2.4. Time derivatives are
// d/dt = -d/dT per year, matching theta.
#ifndef EQOPT_GREEKS_HIGHER_HPP
#define EQOPT_GREEKS_HIGHER_HPP

#include <cmath>

#include "eqopt/black_scholes.hpp"

namespace eqopt {

struct HigherGreeks {
    double charm;   ///< dDelta/dt, per year.
    double speed;   ///< dGamma/dS.
    double colour;  ///< dGamma/dt, per year.
    double zomma;   ///< dGamma/dsigma.
    double veta;    ///< dVega/dt, per year.
};

/// \throws std::invalid_argument unless S, K, T, sigma are strictly positive
///         and finite (via d1_d2).
inline HigherGreeks higher_greeks(double S, double K, double T, double r, double sigma,
                                  double q = 0.0, OptionType type = OptionType::Call) {
    const auto [d1, d2] = d1_d2(S, K, T, r, sigma, q);
    const double sqrt_t = std::sqrt(T);
    const double st = sigma * sqrt_t;
    const double dq = std::exp(-q * T);
    const double phi = norm_pdf(d1);
    const double gamma = dq * phi / (S * st);
    const double drift = (2.0 * (r - q) * T - d2 * st) / (2.0 * T * st);
    HigherGreeks g{};
    g.charm = type == OptionType::Call ? q * dq * norm_cdf(d1) - dq * phi * drift
                                       : -q * dq * norm_cdf(-d1) - dq * phi * drift;
    g.speed = -gamma / S * (d1 / st + 1.0);
    g.colour = dq * phi / (2.0 * S * T * st) *
               (2.0 * q * T + 1.0 + (2.0 * (r - q) * T - d2 * st) * d1 / st);
    g.zomma = gamma * (d1 * d2 - 1.0) / sigma;
    g.veta = S * dq * phi * sqrt_t * (q + (r - q) * d1 / st - (1.0 + d1 * d2) / (2.0 * T));
    return g;
}

}  // namespace eqopt

#endif  // EQOPT_GREEKS_HIGHER_HPP
