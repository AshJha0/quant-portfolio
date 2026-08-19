// FX delta conventions: spot, forward, premium-adjusted, and inversions.
//
// Why four deltas?  FX options are quoted interbank in (delta, vol) space,
// so the *meaning* of delta is part of the quote.  Two independent choices:
//
//   1. Spot vs forward hedge: hedge with spot (delta includes the foreign
//      discount factor) or with an outright forward.
//   2. Premium adjustment: if the premium is paid in the base (foreign)
//      currency -- standard for USDJPY and most EM/non-EURUSD pairs -- the
//      premium itself is a position in the underlying and must be
//      subtracted from the hedge.
//
// With phi = +1 call / -1 put and F = S e^{(r_d - r_f) T}:
//
//   spot                 phi e^{-r_f T} N(phi d1)
//   forward              phi N(phi d1)
//   spot  premium-adj    phi e^{-r_f T} (K/F) N(phi d2)
//   forward premium-adj  phi (K/F) N(phi d2)
//
// Relations: delta_forward = delta_spot * e^{r_f T} and
// delta_pa = delta_unadjusted - premium/S (spot form).
//
// Strike from delta: analytic for the unadjusted conventions.  For
// premium-adjusted *calls* the map K -> delta is NOT monotone (it rises
// then falls; (K/F)N(d2) -> 0 both as K -> 0 and K -> inf), so the
// equation has zero, one, or two solutions.  Market convention takes the
// solution on the right (decreasing) branch, i.e. the larger strike --
// implemented by locating the peak of K N(d2) and Brent-solving on
// [K_peak, K_max].  Premium-adjusted put deltas are monotone in K and
// Brent-solve directly.  This mirrors the Python reference exactly.
//
// ATM conventions:
//   * ATM-forward: K = F.
//   * ATM delta-neutral straddle (DNS): strike where call delta + put delta
//     = 0 under the pair's delta convention: K = F e^{+sigma^2 T/2} for
//     unadjusted deltas (d1 = 0), K = F e^{-sigma^2 T/2} for
//     premium-adjusted deltas (d2 = 0).

#pragma once

#include "fxopt/common.hpp"

namespace fxopt {

/// Delta quoting convention.
enum class DeltaConvention { Spot, Forward, SpotPa, ForwardPa };

/// True for the premium-adjusted conventions.
constexpr bool is_premium_adjusted(DeltaConvention c) noexcept {
    return c == DeltaConvention::SpotPa || c == DeltaConvention::ForwardPa;
}

/// FX option delta under a chosen quoting convention, in units of foreign
/// notional.  Requires T > 0 and sigma > 0 (throws otherwise).
double delta(double S, double K, double T, double r_d, double r_f,
             double sigma, OptionType type,
             DeltaConvention convention = DeltaConvention::Spot);

/// Convert spot delta to forward delta: delta_f = delta_s e^{r_f T}.
/// Holds for both plain and premium-adjusted forms.
double spot_to_forward_delta(double delta_spot, double T, double r_f);

/// Convert forward delta to spot delta: delta_s = delta_f e^{-r_f T}.
double forward_to_spot_delta(double delta_forward, double T, double r_f);

/// Premium-adjust a spot delta: delta_pa = delta_spot - V/S, where V is
/// the domestic-currency premium (V/S is the premium converted to base
/// currency -- a long-base position the hedger already holds).
/// Throws std::invalid_argument unless S > 0 and all three arguments are
/// finite (a NaN premium from a bad quote must not silently poison the
/// hedge ratio).
double premium_adjust_spot_delta(double delta_spot, double price, double S);

/// ATM-forward strike: K = F = S e^{(r_d - r_f) T}.
double atm_forward_strike(double S, double T, double r_d, double r_f);

/// ATM delta-neutral-straddle strike under a delta convention:
/// K = F e^{+sigma^2 T/2} (unadjusted, d1 = 0) or K = F e^{-sigma^2 T/2}
/// (premium-adjusted, d2 = 0).  The strike at which the market quotes
/// 'ATM' vol for most pairs.
double atm_dns_strike(double S, double T, double r_d, double r_f, double sigma,
                      DeltaConvention convention = DeltaConvention::Spot);

/// Invert delta -> strike under any of the four conventions.
///
/// Unadjusted conventions are analytic:
/// K = F exp(-phi z sigma sqrt(T) + sigma^2 T / 2) with
/// z = N^{-1}(phi delta e^{r_f T}) (spot) or N^{-1}(phi delta) (forward).
/// Premium-adjusted conventions are solved with Brent; for PA calls the
/// root on the decreasing branch (larger strike, consistent with OTM
/// quoting) is returned, and a target above the fold's maximum attainable
/// PA delta throws std::invalid_argument.
double strike_from_delta(double target_delta, double S, double T, double r_d,
                         double r_f, double sigma, OptionType type,
                         DeltaConvention convention = DeltaConvention::Spot);

}  // namespace fxopt
