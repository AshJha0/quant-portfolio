// Implied Garman-Kohlhagen volatility from a domestic-currency premium.
//
// The quote is mapped to normalised Black form and solved on the
// out-of-the-money side (an in-the-money quote is converted by parity) by
// the shared core in fxopt/iv_core.hpp: safeguarded Halley iteration on
// transformed objectives with a relative stopping rule. The Python and
// Rust engines run the same algorithm with the same constants.
//
// Bounds use scale-aware tolerances: tau_lo = 4 eps df_d max(F, K) at the
// intrinsic bound when the option is in the money (zero otherwise) and
// tau = 4 eps upper at the sigma -> inf bound. A price within tau_lo of
// the intrinsic bound returns 0.0; a price within tau of the upper bound
// throws, because no finite vol is identifiable there.

#pragma once

#include "fxopt/common.hpp"

namespace fxopt {

/// Implied GK volatility (annualised). Requires T > 0.
///
/// price : observed premium, domestic ccy per unit foreign notional.
///
/// Accuracy: within max(eps * price / vega, 1e-12 * sigma) of the true
/// root, in at most 10 evaluations (typically 2-4).
///
/// Throws std::invalid_argument if the price lies outside
/// (lower - tau_lo, upper - tau), T = 0 or an input is invalid;
/// std::runtime_error if the solver does not converge.
double implied_vol(double price, double S, double K, double T, double r_d, double r_f,
                   OptionType type);

/// Former signature with solver tuning parameters, which the solver no
/// longer has. The extra arguments are ignored.
[[deprecated("tol and max_iter are ignored; use the 7-argument overload")]]
double implied_vol(double price, double S, double K, double T, double r_d, double r_f,
                   OptionType type, double tol, int max_iter = 100);

/// noexcept form of implied_vol(): the same root, or the error the throwing
/// form raises. No allocation.
[[nodiscard]] Result<double> try_implied_vol(double price, double S, double K, double T, double r_d,
                                             double r_f, OptionType type) noexcept;

}  // namespace fxopt
