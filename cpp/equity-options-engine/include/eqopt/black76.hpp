/// \file black76.hpp
/// \brief Black-76 pricing and Greeks for options on forwards/futures.
///
/// Use case: equity *index futures* options (and index options quoted off
/// the forward). Marking off the forward absorbs both the financing rate
/// and the (hard-to-observe) dividend yield into one observable input `F`.
///
/// Conventions (identical to the Python reference `eq_options/black76.py`):
///  - `F` is the forward/futures price for expiry `T` (years, ACT/365F);
///  - `r` is the continuously compounded annualised discount rate applied
///    to the premium (pass `r = 0` for daily-margined futures options);
///  - `sigma` is the annualised volatility of the forward's log-returns;
///  - equivalence: with `F = S exp((r - q) T)`, Black-76 reproduces the
///    Black–Scholes–Merton price exactly.
///
/// Greeks are with respect to the forward `F` (delta, gamma) and per unit
/// of vol / per year / per unit of rate (vega, theta, rho). Rho is the
/// sensitivity of the *discounting only* (`F` held fixed): `rho = -T * V`.

#ifndef EQOPT_BLACK76_HPP
#define EQOPT_BLACK76_HPP

#include <utility>

#include "eqopt/black_scholes.hpp"

namespace eqopt {

/// Analytic Black-76 Greeks (with respect to the forward `F`).
struct Black76Greeks {
    double price;  ///< Present value, currency units.
    double delta;  ///< dV/dF, dimensionless.
    double gamma;  ///< d2V/dF2, per currency unit.
    double vega;   ///< dV/dsigma, per unit of annualised vol.
    double theta;  ///< dV/dt = -dV/dT, per year (calendar decay, F fixed).
    double rho;    ///< dV/dr with F held fixed: `-T * V`.
};

/// \brief Black-76 `d1`/`d2`: `d1 = [ln(F/K) + sigma^2 T/2]/(sigma sqrt(T))`.
/// \param F,K   Forward and strike, strictly positive.
/// \param T     Time to expiry in years, strictly positive.
/// \param sigma Annualised volatility, strictly positive.
/// \return Pair `(d1, d2)` with `d2 = d1 - sigma sqrt(T)`.
/// \throws std::invalid_argument if `F`, `K`, `T` or `sigma` is not
///         strictly positive.
std::pair<double, double> b76_d1_d2(double F, double K, double T,
                                    double sigma);

/// \brief Black-76 present value of a European option on a forward price.
///
/// \param F     Forward/futures price for expiry `T`, `F >= 0`.
/// \param K     Strike, `K >= 0`.
/// \param T     Time to expiry in years (ACT/365F), `T >= 0`.
/// \param r     Continuously compounded annualised discount rate.
/// \param sigma Annualised volatility of the forward, `sigma >= 0`.
/// \param type  Option payoff direction.
/// \return Present value in currency units.
/// \throws std::invalid_argument on negative/NaN inputs.
/// \note `T == 0` returns intrinsic; `sigma == 0` returns the discounted
///       intrinsic `exp(-rT) max(±(F - K), 0)`.
double black76_price(double F, double K, double T, double r, double sigma,
                     OptionType type = OptionType::Call);

/// \brief Analytic Black-76 Greeks with respect to the forward.
/// \param F,K,T,r,sigma As in black76_price(); `F`, `K`, `T`, `sigma` must
///        be strictly positive for finite Greeks.
/// \param type Option payoff direction.
/// \return Struct of price, delta, gamma, vega, theta, rho (units in the
///         struct documentation).
/// \throws std::invalid_argument if inputs are invalid or not strictly
///         positive.
Black76Greeks black76_greeks(double F, double K, double T, double r,
                             double sigma, OptionType type = OptionType::Call);

}  // namespace eqopt

#endif  // EQOPT_BLACK76_HPP
