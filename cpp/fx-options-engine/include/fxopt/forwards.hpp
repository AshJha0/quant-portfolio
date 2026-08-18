// FX forwards via covered interest parity (CIP).
//
//   F = S * exp((r_d - r_f) * T)
//
// A domestic investor can replicate the forward by borrowing domestic cash,
// buying spot foreign currency and depositing it at r_f; absence of
// arbitrage forces the forward to the CIP level (abstracting from the
// cross-currency basis -- see docs/METHODOLOGY.md assumption register).
//
// Forward points are quoted as (F - S) scaled by the pair's pip factor
// (1e4 for most pairs, 1e2 for JPY-quoted pairs).

#pragma once

#include "fxopt/common.hpp"

namespace fxopt {

/// Standard pip scaling: 1e4 for e.g. EURUSD (pip = 0.0001).
inline constexpr double kPipFactorDefault = 1e4;
/// Pip scaling for JPY-quoted pairs, e.g. USDJPY (pip = 0.01).
inline constexpr double kPipFactorJpy = 1e2;

/// Covered-interest-parity forward rate F = S * exp((r_d - r_f) T).
/// F > S when the domestic rate exceeds the foreign rate (forward premium
/// on the base currency), the classic carry relationship.
double cip_forward(double S, double T, double r_d, double r_f);

/// Forward points (F - S) * pip_factor; positive when the base currency
/// trades at a forward premium.  Throws if pip_factor <= 0.
double forward_points(double S, double T, double r_d, double r_f,
                      double pip_factor = kPipFactorDefault);

/// Forward implied by put-call parity: F = K + (C - P) e^{r_d T}.
/// Long call + short put at the same strike replicates a forward purchase
/// of the base currency (a "synthetic forward" / conversion trade).
double synthetic_forward_from_options(double call_price, double put_price,
                                      double K, double T, double r_d);

}  // namespace fxopt
