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

#include <algorithm>
#include <cmath>

#include "fxopt/common.hpp"
#include "fxopt/garman_kohlhagen.hpp"

namespace fxopt {

/// Black-76 price of a European FX option on the forward.  F, K > 0;
/// T >= 0; sigma >= 0; r_d is used for discounting only.  Premium in
/// domestic currency per unit foreign notional.
double black76_price(double F, double K, double T, double r_d, double sigma, OptionType type);

/// Black-76 with the forward built from spot via CIP.  Equals gk_price to
/// machine precision.
double black76_from_spot(double S, double K, double T, double r_d, double r_f, double sigma,
                         OptionType type);

/// noexcept form of black76_price(): the same value or error, no allocation.
[[nodiscard]] inline Result<double> try_black76_price(double F, double K, double T, double r_d,
                                                      double sigma, OptionType type) noexcept {
    const double p = phi(type);
    if (auto e = try_require_finite(F, "F"); !e) [[unlikely]]
        return std::unexpected(e.error());
    if (auto e = try_require_finite(K, "K"); !e) [[unlikely]]
        return std::unexpected(e.error());
    if (auto e = try_require_finite(T, "T"); !e) [[unlikely]]
        return std::unexpected(e.error());
    if (auto e = try_require_finite(r_d, "r_d"); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    if (auto e = try_require_finite(sigma, "sigma"); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    if (F <= 0.0) [[unlikely]]
        return std::unexpected(Error{Errc::ForwardNonPositive, "F", F});
    if (K <= 0.0) [[unlikely]]
        return std::unexpected(Error{Errc::StrikeNonPositive, "K", K});
    if (T < 0.0) [[unlikely]]
        return std::unexpected(Error{Errc::NegativeExpiry, "T", T});
    if (sigma < 0.0) [[unlikely]] {
        return std::unexpected(Error{Errc::NegativeVol, "sigma", sigma});
    }
    const double df = std::exp(-r_d * T);
    const double v = sigma * std::sqrt(T);
    if (T == 0.0 || v <= kMinVol) [[unlikely]]
        return df * std::max(p * (F - K), 0.0);
    const double d1v = (std::log(F / K) + 0.5 * v * v) / v;
    const double d2v = d1v - v;
    return p * df * (F * norm_cdf(p * d1v) - K * norm_cdf(p * d2v));
}

}  // namespace fxopt
