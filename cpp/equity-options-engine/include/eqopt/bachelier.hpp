// Bachelier (normal) model: price, Greeks, implied normal vol (port of
// eq_options.bachelier). Schachermayer and Teichmann (2008), Math. Finance
// 18(1), 155-170. F and K may be negative; sigma_n is in price units per
// sqrt(year); theta is -dV/dT per year with F held.
#ifndef EQOPT_BACHELIER_HPP
#define EQOPT_BACHELIER_HPP

#include "eqopt/black_scholes.hpp"

namespace eqopt {

struct BachelierGreeks {
    double price;
    double delta;  ///< dV/dF.
    double gamma;  ///< d2V/dF2.
    double vega;   ///< dV/dsigma_n.
    double theta;  ///< -dV/dT, per year.
};

/// \throws std::invalid_argument on non-finite inputs or T, sigma_n < 0.
double bachelier_price(double F, double K, double T, double sigma_n, double r = 0.0,
                       OptionType type = OptionType::Call);

/// \throws std::invalid_argument unless T > 0 and sigma_n > 0.
BachelierGreeks bachelier_greeks(double F, double K, double T, double sigma_n, double r = 0.0,
                                 OptionType type = OptionType::Call);

/// Safeguarded Newton on the out-of-the-money side, relative tolerance 1e-14.
/// \throws std::invalid_argument if the price is at or below intrinsic.
double bachelier_implied_vol(double price, double F, double K, double T, double r = 0.0,
                             OptionType type = OptionType::Call);

}  // namespace eqopt

#endif  // EQOPT_BACHELIER_HPP
