// Implied Garman-Kohlhagen volatility from a domestic-currency premium.
//
// Strategy (mirroring the Python reference): Newton-Raphson from a
// moneyness-aware initial guess (fast quadratic convergence when vega is
// healthy), falling back to bracketed Brent (guaranteed convergence) if
// Newton stalls or wanders outside the no-arbitrage bracket.
//
// Prices outside the no-arbitrage bounds
// [discounted intrinsic on the forward, discounted forward bound] throw;
// a price whose time value is below double-precision resolution returns
// the sigma -> 0 limit (0.0).

#pragma once

#include "fxopt/common.hpp"

namespace fxopt {

/// Implied GK volatility (annualised).  Requires T > 0.
///
/// price : observed premium, domestic ccy per unit foreign notional.
/// tol   : absolute tolerance on the vol root (default 1e-12; the test
///         suite verifies round trips to 1e-10).
/// max_iter : Newton iteration budget before the Brent fallback.
///
/// Throws std::invalid_argument if the price violates the no-arbitrage
/// bounds, T = 0, or the implied vol exceeds 50 (unattainably high price).
double implied_vol(double price, double S, double K, double T, double r_d,
                   double r_f, OptionType type, double tol = 1e-12,
                   int max_iter = 100);

}  // namespace fxopt
