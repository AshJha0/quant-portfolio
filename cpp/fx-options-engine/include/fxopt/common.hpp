// Shared conventions, validation and normal-distribution utilities.
//
// Conventions (used everywhere in this library, mirroring the Python
// reference package `fx_options`):
//   * FX pairs are quoted BASE/QUOTE: EURUSD = number of USD per 1 EUR.
//   * S is the spot rate in domestic (quote) currency per unit of foreign
//     (base) currency.  All option prices are in domestic currency per unit
//     of foreign notional.
//   * r_d is the domestic (quote-currency) continuously compounded rate,
//     r_f the foreign (base-currency) rate, both annualised, ACT/365F.
//   * T is time to expiry in years, sigma the annualised lognormal vol.
//   * Negative rates are legal (EUR/CHF era); negative T or sigma is not.
//
// Invalid inputs throw std::invalid_argument with an informative message,
// matching the Python reference's ValueError behaviour.

#pragma once

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace fxopt {

/// Option type; phi(Call) = +1, phi(Put) = -1.
enum class OptionType { Call, Put };

/// Sign convention: +1.0 for a call (on the base currency), -1.0 for a put.
constexpr double phi(OptionType type) noexcept {
    return type == OptionType::Call ? 1.0 : -1.0;
}

/// The opposite option type (used by foreign-domestic symmetry).
constexpr OptionType other_type(OptionType type) noexcept {
    return type == OptionType::Call ? OptionType::Put : OptionType::Call;
}

/// Standard normal PDF n(x).
inline double norm_pdf(double x) noexcept {
    constexpr double inv_sqrt_2pi = 0.3989422804014326779399461;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

/// Standard normal CDF N(x), via erfc for full double precision in both tails.
inline double norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x / std::numbers::sqrt2);
}

/// Inverse standard normal CDF (Acklam/Moro-style rational approximation
/// refined with two Halley steps -> ~1e-15 absolute accuracy).
double norm_ppf(double p);

namespace detail {

inline void require_finite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) +
                                    " must be finite, got " +
                                    std::to_string(value));
    }
}

}  // namespace detail

/// Validate common pricing inputs; throws std::invalid_argument on bad data.
///
/// S, K must be positive and finite; T >= 0; sigma >= 0; rates finite
/// (negative rates are allowed).
inline void validate_inputs(double S, double K, double T, double r_d,
                            double r_f, double sigma) {
    detail::require_finite(S, "S");
    detail::require_finite(K, "K");
    detail::require_finite(T, "T");
    detail::require_finite(r_d, "r_d");
    detail::require_finite(r_f, "r_f");
    detail::require_finite(sigma, "sigma");
    if (S <= 0.0) {
        throw std::invalid_argument("Spot S must be positive, got " +
                                    std::to_string(S));
    }
    if (K <= 0.0) {
        throw std::invalid_argument("Strike K must be positive, got " +
                                    std::to_string(K));
    }
    if (T < 0.0) {
        throw std::invalid_argument(
            "Time to expiry T must be non-negative, got " + std::to_string(T));
    }
    if (sigma < 0.0) {
        throw std::invalid_argument(
            "Volatility sigma must be non-negative, got " +
            std::to_string(sigma));
    }
}

}  // namespace fxopt
