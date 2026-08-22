// Implied Garman-Kohlhagen volatility from a domestic-currency premium.
//
// Strategy (mirroring the Python reference): Newton-Raphson from a
// moneyness-aware initial guess (fast quadratic convergence when vega is
// healthy), falling back to bracketed Brent (guaranteed convergence) if
// Newton stalls or wanders outside the no-arbitrage bracket.
//
// Prices outside the no-arbitrage bounds
// [discounted intrinsic on the forward, discounted forward bound] throw;
// a price whose time value is below double-precision resolution of the
// sigma -> 0 bound returns that limit (0.0). The symmetric corner -- deep
// ITM + long-dated + high vol, where N(d1)/N(d2) saturate to 0/1 in double
// precision and the price becomes bit-identical to the sigma -> inf bound
// -- is a flat plateau with no unique root, not a single degenerate point
// with a natural finite limit; the solver throws there rather than
// returning an arbitrary point from inside the plateau (see
// docs/VALIDATION.md, failure mode 4, and the Python reference).

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
/// bounds, T = 0, the implied vol exceeds 50 (unattainably high price), or
/// the price sits in the flat plateau near the sigma -> infinity bound
/// where vol is genuinely unrecoverable (see the file-level comment above).
double implied_vol(double price, double S, double K, double T, double r_d,
                   double r_f, OptionType type, double tol = 1e-12,
                   int max_iter = 100);

}  // namespace fxopt
