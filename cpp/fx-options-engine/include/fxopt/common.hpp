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
// matching the Python reference's ValueError behaviour. Every kernel also
// has a noexcept `try_*` form returning std::expected<T, Error>
// (fxopt/error.hpp) that never throws or allocates.

#pragma once

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

#include "fxopt/error.hpp"

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

/// norm_ppf without the domain check; the caller guarantees 0 < p < 1.
double norm_ppf_unchecked(double p) noexcept;

// ------------------------------------------------------------------------
// noexcept validation. The throwing helpers below are thin wrappers, so the
// checks, their order and the messages are the same in both APIs.
// ------------------------------------------------------------------------

/// Finite check; on failure the error names `field`.
[[nodiscard]] inline Result<void> try_require_finite(double value, const char* field) noexcept {
    if (!std::isfinite(value)) [[unlikely]] {
        return std::unexpected(Error{Errc::NonFinite, field, value});
    }
    return {};
}

/// Finite-result check; on failure the error names the kernel `what`.
[[nodiscard]] inline Result<double> try_finite_result(double value, const char* what) noexcept {
    if (!std::isfinite(value)) [[unlikely]] {
        return std::unexpected(Error{Errc::NonFiniteResult, what});
    }
    return value;
}

/// noexcept form of validate_inputs().
[[nodiscard]] inline Result<void> try_validate_inputs(double S, double K, double T,
                                                      double r_d, double r_f,
                                                      double sigma) noexcept {
    if (auto e = try_require_finite(S, "S"); !e) [[unlikely]] return e;
    if (auto e = try_require_finite(K, "K"); !e) [[unlikely]] return e;
    if (auto e = try_require_finite(T, "T"); !e) [[unlikely]] return e;
    if (auto e = try_require_finite(r_d, "r_d"); !e) [[unlikely]] return e;
    if (auto e = try_require_finite(r_f, "r_f"); !e) [[unlikely]] return e;
    if (auto e = try_require_finite(sigma, "sigma"); !e) [[unlikely]] return e;
    if (S <= 0.0) [[unlikely]] return std::unexpected(Error{Errc::SpotNonPositive, "S", S});
    if (K <= 0.0) [[unlikely]] return std::unexpected(Error{Errc::StrikeNonPositive, "K", K});
    if (T < 0.0) [[unlikely]] return std::unexpected(Error{Errc::NegativeExpiry, "T", T});
    if (sigma < 0.0) [[unlikely]] {
        return std::unexpected(Error{Errc::NegativeVol, "sigma", sigma});
    }
    return {};
}

namespace detail {

inline void require_finite(double value, const char* name) {
    value_or_throw(try_require_finite(value, name));
}

/// Return `value`, or throw std::invalid_argument naming `what` when a
/// finite input produced a non-finite result (a discount factor or forward
/// overflowed).
inline double finite_result(double value, const char* what) {
    return value_or_throw(try_finite_result(value, what));
}

}  // namespace detail

/// Validate common pricing inputs; throws std::invalid_argument on bad data.
///
/// S, K must be positive and finite; T >= 0; sigma >= 0; rates finite
/// (negative rates are allowed).
inline void validate_inputs(double S, double K, double T, double r_d,
                            double r_f, double sigma) {
    value_or_throw(try_validate_inputs(S, K, T, r_d, r_f, sigma));
}

}  // namespace fxopt
