/// \file implied_vol.hpp
/// \brief Robust implied Black–Scholes volatility inversion.
///
/// Bracketed Newton with bisection fallback, mirroring the Python reference
/// `eq_options.black_scholes.implied_vol`. Newton iterations use analytic
/// vega; every step is kept inside a maintained bracket, and whenever Newton
/// stalls (tiny vega deep ITM/OTM, or a step outside the bracket) the
/// algorithm falls back to bisection, so it is robust across moneyness
/// 0.5x–2.0x and expiries from days to years.

#ifndef EQOPT_IMPLIED_VOL_HPP
#define EQOPT_IMPLIED_VOL_HPP

#include "eqopt/black_scholes.hpp"

namespace eqopt {

/// \brief Implied Black–Scholes volatility from an observed premium.
///
/// \param price Observed option premium, currency units. Must lie strictly
///              between the no-arbitrage bounds: the `sigma -> 0` lower
///              bound (discounted intrinsic on the forward) and
///              `S exp(-qT)` for calls / `K exp(-rT)` for puts.
/// \param S,K   Spot and strike, strictly positive.
/// \param T     Time to expiry in years, strictly positive.
/// \param r     Continuously compounded annualised risk-free rate.
/// \param q     Continuously compounded annualised dividend yield.
/// \param type  Option payoff direction.
/// \param tol   Absolute price tolerance for convergence (default 1e-10).
/// \param max_iter  Maximum Newton/bisection iterations.
/// \param sigma_lo,sigma_hi Initial volatility bracket, annualised; the top
///              is expanded automatically up to 1e3 for extreme premiums.
/// \return Annualised implied volatility.
/// \throws std::invalid_argument if `price` is at or below the `sigma -> 0`
///         lower bound (sub-intrinsic), at or above the `sigma -> inf`
///         upper bound, if `T == 0`, or if any underlying input is invalid.
double implied_vol(double price, double S, double K, double T, double r,
                   double q = 0.0, OptionType type = OptionType::Call,
                   double tol = 1e-10, int max_iter = 200,
                   double sigma_lo = 1e-9, double sigma_hi = 10.0);

}  // namespace eqopt

#endif  // EQOPT_IMPLIED_VOL_HPP
