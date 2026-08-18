// Garman-Kohlhagen pricing for European FX options.
//
// Garman-Kohlhagen (1983) is Black-Scholes with the continuous dividend
// yield replaced by the foreign interest rate: holding the foreign currency
// pays the foreign risk-free rate, exactly as a dividend-paying stock pays
// its yield.
//
// Formulae:
//   d1 = [ln(S/K) + (r_d - r_f + sigma^2/2) T] / (sigma sqrt(T))
//   d2 = d1 - sigma sqrt(T)
//   call = S e^{-r_f T} N(d1) - K e^{-r_d T} N(d2)
//   put  = K e^{-r_d T} N(-d2) - S e^{-r_f T} N(-d1)
//
// Limits handled explicitly (matching the Python reference):
//   T = 0 returns intrinsic value; sigma*sqrt(T) <= 1e-12 returns the
//   discounted intrinsic on the forward, e^{-r_d T} max(phi (F - K), 0).

#pragma once

#include "fxopt/common.hpp"

namespace fxopt {

/// Minimum sigma*sqrt(T) below which d1/d2 are treated as undefined.
inline constexpr double kMinVol = 1e-12;

/// Garman-Kohlhagen d1.  Throws std::invalid_argument on invalid inputs or
/// when sigma*sqrt(T) <= 1e-12 (d1 undefined).
double d1(double S, double K, double T, double r_d, double r_f, double sigma);

/// Garman-Kohlhagen d2 = d1 - sigma*sqrt(T).
double d2(double S, double K, double T, double r_d, double r_f, double sigma);

/// Garman-Kohlhagen price of a European FX option.
///
/// Price is in domestic (quote) currency per unit of foreign (base)
/// notional, e.g. USD per EUR for EURUSD.  T = 0 returns intrinsic;
/// sigma = 0 returns the discounted forward intrinsic.
double gk_price(double S, double K, double T, double r_d, double r_f,
                double sigma, OptionType type);

/// Convenience wrapper: gk_price(..., OptionType::Call).
double gk_call(double S, double K, double T, double r_d, double r_f,
               double sigma);

/// Convenience wrapper: gk_price(..., OptionType::Put).
double gk_put(double S, double K, double T, double r_d, double r_f,
              double sigma);

}  // namespace fxopt
