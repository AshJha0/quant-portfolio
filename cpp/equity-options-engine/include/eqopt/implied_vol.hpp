/// ile implied_vol.hpp
/// rief Implied Black-Scholes volatility, solved on the out-of-the-money side.
///
/// The quote is mapped to normalised Black form; an in-the-money quote is
/// converted to its out-of-the-money counterpart by parity, so the solver
/// always works on time value. The shared core (`eqopt/iv_core.hpp`, the
/// same algorithm as the Python and Rust engines) runs safeguarded Halley
/// iteration on transformed objectives with a relative stopping rule.
///
/// Accuracy: within max(eps * price / vega, 1e-12 * sigma) of the true root
/// over T in [1e-7, 50], sigma in [1e-4, 5], |ln(F/K)| <= 5, in at most 10
/// evaluations (typically 2-4). A deep in-the-money quote carries less
/// information than its out-of-the-money twin: with S=100, K=90, T=0.02,
/// sigma=10% one ulp of the call price moves the vol by about 5e-4; the
/// returned vol then reprices the quote to a few ulps.

#ifndef EQOPT_IMPLIED_VOL_HPP
#define EQOPT_IMPLIED_VOL_HPP

#include "eqopt/black_scholes.hpp"

namespace eqopt {

/// rief Implied Black-Scholes volatility from an observed premium.
///
/// \param price Observed premium, strictly between the sigma -> 0 bound
///              (discounted forward intrinsic) and S e^{-qT} (call) /
///              K e^{-rT} (put).
/// \param S,K,T Spot, strike and expiry, strictly positive.
/// \param r,q   Continuously compounded rate and dividend yield.
/// \param type  Call or put.
/// 	hrows std::invalid_argument on invalid inputs, T == 0, or a price at
///         or outside the bounds; std::runtime_error if the solver does not
///         converge in 100 iterations.
double implied_vol(double price, double S, double K, double T, double r,
                   double q = 0.0, OptionType type = OptionType::Call);

/// rief Former signature with solver tuning parameters, which the new
/// solver does not have. The extra arguments are ignored.
[[deprecated("tol, max_iter, sigma_lo and sigma_hi are ignored; use the 7-argument overload")]]
double implied_vol(double price, double S, double K, double T, double r,
                   double q, OptionType type, double tol, int max_iter = 200,
                   double sigma_lo = 1e-9, double sigma_hi = 10.0);

}  // namespace eqopt

#endif  // EQOPT_IMPLIED_VOL_HPP
