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

#include <algorithm>
#include <cmath>

#include "fxopt/common.hpp"
#include "qp/black_scholes.hpp"

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
double gk_price(double S, double K, double T, double r_d, double r_f, double sigma,
                OptionType type);

/// Convenience wrapper: gk_price(..., OptionType::Call).
double gk_call(double S, double K, double T, double r_d, double r_f, double sigma);

/// Convenience wrapper: gk_price(..., OptionType::Put).
double gk_put(double S, double K, double T, double r_d, double r_f, double sigma);

// ------------------------------------------------------------------------
// noexcept kernels. Same arithmetic as the throwing functions above (which
// wrap these), so the values are bit-for-bit the same.
// ------------------------------------------------------------------------

/// noexcept form of d1().
[[nodiscard]] inline Result<double> try_d1(double S, double K, double T, double r_d, double r_f,
                                           double sigma) noexcept {
    if (auto e = try_validate_inputs(S, K, T, r_d, r_f, sigma); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    const double vol_sqrt_t = sigma * std::sqrt(T);
    if (vol_sqrt_t <= kMinVol) [[unlikely]] {
        return std::unexpected(Error{Errc::D1Undefined, "", vol_sqrt_t});
    }
    return qp::carry_d1_d2(S, K, T, r_d, r_f, sigma).d1;
}

/// GK price on inputs the caller has validated (try_validate_inputs). The
/// raw value may be non-finite if a discount factor overflows.
inline double gk_price_unchecked(double S, double K, double T, double r_d, double r_f, double sigma,
                                 OptionType type) noexcept {
    const double p = phi(type);
    if (T == 0.0) [[unlikely]]
        return std::max(p * (S - K), 0.0);
    const double sqrt_t = std::sqrt(T);
    const double vol_sqrt_t = sigma * sqrt_t;
    const double disc_s = S * std::exp(-r_f * T);
    const double disc_k = K * std::exp(-r_d * T);
    if (vol_sqrt_t <= kMinVol) [[unlikely]]
        return std::max(p * (disc_s - disc_k), 0.0);
    // Shared carry-form kernel (qp/black_scholes.hpp); r_f is the carry yield.
    const qp::CarryD d = qp::carry_d1_d2(S, K, T, r_d, r_f, sigma);
    return qp::carry_value<qp::NormCdfDiv>(disc_s, disc_k, d.d1, d.d2, p);
}

/// noexcept form of gk_price(): the same value, or the error the throwing
/// form raises. No allocation.
[[nodiscard]] inline Result<double> try_gk_price(double S, double K, double T, double r_d,
                                                 double r_f, double sigma,
                                                 OptionType type) noexcept {
    if (auto e = try_validate_inputs(S, K, T, r_d, r_f, sigma); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    const double v = gk_price_unchecked(S, K, T, r_d, r_f, sigma, type);
    if (T == 0.0) return v;  // intrinsic, finite by construction
    return try_finite_result(v, "gk_price");
}

}  // namespace fxopt
