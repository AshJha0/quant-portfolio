// Black-76 pricing off the FX forward.
//
// FX desks think in forwards: the smile is marked against the forward, and
// Black-76 prices directly off it,
//
//   call = e^{-r_d T} [F N(d1) - K N(d2)],
//   d1 = [ln(F/K) + sigma^2 T / 2] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T).
//
// With the covered-interest-parity forward F = S e^{(r_d - r_f) T},
// Black-76 is algebraically identical to Garman-Kohlhagen -- substituting F
// recovers the GK d1/d2 and price exactly (tested to 1e-12).

#pragma once

#include "fxopt/common.hpp"

namespace fxopt {

/// Black-76 price of a European FX option on the forward.  F, K > 0;
/// T >= 0; sigma >= 0; r_d is used for discounting only.  Premium in
/// domestic currency per unit foreign notional.
double black76_price(double F, double K, double T, double r_d, double sigma,
                     OptionType type);

/// Black-76 with the forward built from spot via CIP.  Equals gk_price to
/// machine precision.
double black76_from_spot(double S, double K, double T, double r_d, double r_f,
                         double sigma, OptionType type);

}  // namespace fxopt
