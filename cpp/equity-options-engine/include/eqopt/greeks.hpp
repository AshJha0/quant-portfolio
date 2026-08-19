/// \file greeks.hpp
/// \brief Analytic Black–Scholes Greeks and templated finite-difference
///        Greeks for any pricer.
///
/// Units and conventions (identical to the Python reference
/// `eq_options/greeks.py`):
///  - delta : dV/dS, dimensionless (per unit of spot);
///  - gamma : d2V/dS2, per currency unit;
///  - vega  : dV/dsigma, currency units per unit of annualised vol
///            (divide by 100 for the market's "per vol point");
///  - theta : dV/dt, currency units per *year* (divide by 365 for per-day);
///  - rho   : dV/dr, currency units per unit of rate;
///  - vanna : d2V/(dS dsigma);
///  - volga : d2V/dsigma2 (a.k.a. vomma).
///
/// All rates continuously compounded, annualised; `T` in years (ACT/365F).

#ifndef EQOPT_GREEKS_HPP
#define EQOPT_GREEKS_HPP

#include <cmath>
#include <stdexcept>

#include "eqopt/black_scholes.hpp"

namespace eqopt {

/// Container for the full Greek set of a European option.
/// Units as documented in the file header.
struct BSGreeks {
    double price;  ///< Present value, currency units.
    double delta;  ///< dV/dS, dimensionless.
    double gamma;  ///< d2V/dS2, per currency unit.
    double vega;   ///< dV/dsigma, per unit of annualised vol.
    double theta;  ///< dV/dt, per year.
    double rho;    ///< dV/dr, per unit of rate.
    double vanna;  ///< d2V/(dS dsigma).
    double volga;  ///< d2V/dsigma2.
};

/// \brief Analytic Black–Scholes–Merton Greeks (continuous dividend yield).
///
/// \param S,K   Spot and strike (currency units), strictly positive.
/// \param T     Time to expiry in years (ACT/365F), strictly positive.
/// \param r     Continuously compounded annualised risk-free rate.
/// \param sigma Annualised volatility, strictly positive.
/// \param q     Continuously compounded annualised dividend yield.
/// \param type  Option payoff direction.
/// \return Price plus delta, gamma, vega, theta, rho, vanna, volga.
/// \throws std::invalid_argument if `S`, `K`, `T`, `sigma` are not strictly
///         positive (Greeks are singular at the boundary).
BSGreeks bs_greeks(double S, double K, double T, double r, double sigma,
                   double q = 0.0, OptionType type = OptionType::Call);

/// \brief Central finite-difference Greeks for *any* pricer.
///
/// The pricer must be callable as `pricer(S, K, T, r, sigma, q, type)` and
/// return a double price (a lambda adapting e.g. `crr_price` works).
/// Central differences are used everywhere; second derivatives use the
/// standard three-point stencil; vanna uses the four-point cross stencil.
/// Theta is reported as `dV/dt = -dV/dT` (per year).
///
/// \tparam Pricer Callable with signature
///         `double(double S, double K, double T, double r, double sigma,
///                 double q, OptionType type)`.
/// \param pricer     Pricing function (e.g. `eqopt::bs_price`).
/// \param S,K,T,r,sigma,q,type Contract and market inputs, as in bs_greeks().
/// \param rel_bump   Relative bump for first derivatives; absolute bumps are
///                   `rel_bump * max(|x|, 1)`. The default 1e-5 balances
///                   truncation vs round-off for analytic pricers.
/// \param rel_bump2  Relative bump for second derivatives (gamma, vanna,
///                   volga), where round-off scales like `eps / h^2` and
///                   needs a larger `h` (optimal `h ~ eps^0.25`).
/// \return Finite-difference price and Greeks (vanna/volga included).
/// \throws std::invalid_argument if inputs are invalid, or `T` (resp.
///         `sigma`) is too small for its central down-bump to stay in the
///         valid domain.
template <typename Pricer>
BSGreeks fd_greeks(Pricer&& pricer, double S, double K, double T, double r,
                   double sigma, double q = 0.0,
                   OptionType type = OptionType::Call, double rel_bump = 1e-5,
                   double rel_bump2 = 2e-4) {
    validate_inputs(S, K, T, sigma);

    auto f = [&](double s, double sig, double t, double rr) -> double {
        return static_cast<double>(pricer(s, K, t, rr, sig, q, type));
    };

    const double h_s = rel_bump * std::fmax(std::fabs(S), 1.0);
    const double h_v = rel_bump * std::fmax(std::fabs(sigma), 1.0);
    const double h_t = rel_bump * std::fmax(std::fabs(T), 1.0);
    const double h_r = rel_bump * std::fmax(std::fabs(r), 1.0);
    const double h_s2 = rel_bump2 * std::fmax(std::fabs(S), 1.0);
    const double h_v2 = rel_bump2 * std::fmax(std::fabs(sigma), 1.0);
    if (T - h_t <= 0.0) {
        throw std::invalid_argument("T too small for a central theta bump");
    }
    if (sigma - std::fmax(h_v, h_v2) <= 0.0) {
        throw std::invalid_argument(
            "sigma too small for a central vega/volga bump");
    }

    const double price = f(S, sigma, T, r);
    BSGreeks g{};
    g.price = price;
    g.delta = (f(S + h_s, sigma, T, r) - f(S - h_s, sigma, T, r)) / (2.0 * h_s);
    g.gamma = (f(S + h_s2, sigma, T, r) - 2.0 * price +
               f(S - h_s2, sigma, T, r)) /
              (h_s2 * h_s2);
    g.vega = (f(S, sigma + h_v, T, r) - f(S, sigma - h_v, T, r)) / (2.0 * h_v);
    g.theta = -(f(S, sigma, T + h_t, r) - f(S, sigma, T - h_t, r)) /
              (2.0 * h_t);
    g.rho = (f(S, sigma, T, r + h_r) - f(S, sigma, T, r - h_r)) / (2.0 * h_r);
    g.vanna = (f(S + h_s2, sigma + h_v2, T, r) -
               f(S + h_s2, sigma - h_v2, T, r) -
               f(S - h_s2, sigma + h_v2, T, r) +
               f(S - h_s2, sigma - h_v2, T, r)) /
              (4.0 * h_s2 * h_v2);
    g.volga = (f(S, sigma + h_v2, T, r) - 2.0 * price +
               f(S, sigma - h_v2, T, r)) /
              (h_v2 * h_v2);
    return g;
}

}  // namespace eqopt

#endif  // EQOPT_GREEKS_HPP
