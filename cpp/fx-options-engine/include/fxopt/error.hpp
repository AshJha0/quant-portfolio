// Error values for the noexcept kernel API.
//
// Every `try_*` kernel is `noexcept` and returns `std::expected<T, Error>`.
// An `Error` is a code plus the offending field name (a string literal) and
// up to three numbers, so building one never allocates. The message text is
// formatted only when somebody asks for it (`message()`), which is what the
// throwing wrappers do: they keep the exception types and messages of the
// original API.

#pragma once

#include <cstdint>
#include <expected>
#include <string>

#include <type_traits>
#include <utility>
#include "qp/expected.hpp"

namespace fxopt {

/// What went wrong. Each code maps to one exception message.
enum class Errc : std::uint8_t {
    NonFinite,              ///< `field` (a) is NaN or Inf.
    SpotNonPositive,        ///< S (a) <= 0.
    StrikeNonPositive,      ///< K (a) <= 0.
    ForwardNonPositive,     ///< F (a) <= 0.
    NegativeExpiry,         ///< T (a) < 0.
    NegativeVol,            ///< sigma (a) < 0.
    NonFiniteResult,        ///< `field` names the kernel whose result overflowed.
    D1Undefined,            ///< sigma*sqrt(T) (a) <= 1e-12.
    GreeksDomain,           ///< analytic_greeks with T <= 0 or sigma <= 0.
    BadSteps,               ///< steps (a) < 1.
    ProbabilityOutOfRange,  ///< tree probability (a) outside [0, 1].
    WorkspaceTooSmall,      ///< workspace size (a) below the required (b).
    IvExpired,              ///< implied vol: T <= 0.
    IvOutsideBounds,        ///< implied vol: price a outside [b, c].
    IvNearUpperBound,       ///< implied vol: price a within b of upper bound c.
    AtBound,                ///< implied vol: within rounding of a bound.
    NoConvergence,          ///< implied vol: root finder did not converge.
    McExpired,              ///< MC: T <= 0.
    TooFewPaths,            ///< MC: n_paths a gives b samples, c needed.
    McOverflow,             ///< MC: simulated spots overflow.
    BadThreads,             ///< MC: threads == 0.
    SizeMismatch,           ///< batch: span lengths differ (a vs b).
};

/// A kernel error. Trivially copyable, never allocates.
struct Error {
    Errc code;
    const char* field = "";  ///< static string, never owned
    double a = 0.0, b = 0.0, c = 0.0;
};

/// Result of a noexcept kernel.
template <class T>
using Result = qp::Result<T, Error>;

/// Message identical to the exception text of the throwing API.
/// Allocates; call it on the error path only.
std::string message(const Error& e);

/// Throw what the original API threw for `e`: std::runtime_error for
/// NoConvergence, std::invalid_argument otherwise.
[[noreturn]] void throw_error(const Error& e);

/// Unwrap a result or throw (qp/expected.hpp). Used by the throwing wrappers.
using qp::value_or_throw;

}  // namespace fxopt
