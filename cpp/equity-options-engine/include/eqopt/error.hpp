/// \file error.hpp
/// \brief Error values for the noexcept kernel API.
///
/// Every `try_*` kernel is `noexcept` and returns `std::expected<T, Error>`.
/// An `Error` is a code plus the offending field name (a string literal) and
/// up to four numbers, so building one never allocates. The text of the
/// message is formatted only when somebody asks for it (`message()`), which
/// is what the throwing wrappers do: they keep the exact exception types and
/// messages of the original API.

#ifndef EQOPT_ERROR_HPP
#define EQOPT_ERROR_HPP

#include <cstdint>
#include <expected>
#include <string>

#include "qp/expected.hpp"

namespace eqopt {

/// What went wrong. Each code maps to one exception message.
enum class Errc : std::uint8_t {
    NegativeOrNonFinite,   ///< `field` (a) is NaN, Inf or negative.
    NonFiniteRate,         ///< r (a) or q (b) is NaN or Inf.
    NotStrictlyPositive,   ///< d1/d2: S, K, T, sigma = a, b, c, d.
    B76NotStrictlyPositive,///< b76_d1_d2: F, K, T, sigma = a, b, c, d.
    NonFiniteResult,       ///< `field` names the kernel whose result overflowed.
    BadSteps,              ///< n_steps (a) < 1.
    ProbabilityOutOfRange, ///< CRR risk-neutral p (a) outside (0, 1).
    WorkspaceTooSmall,     ///< workspace size (a) below the required (b).
    PriceNotFinite,        ///< implied vol: quote is NaN or Inf.
    Expired,               ///< implied vol: T <= 0.
    NonPositiveSpotStrike, ///< implied vol: S <= 0 or K <= 0.
    BelowLowerBound,       ///< implied vol: price (a) <= lower bound (b).
    AboveUpperBound,       ///< implied vol: price (a) >= upper bound (b).
    AtBound,               ///< implied vol: within rounding of a bound.
    NoConvergence,         ///< implied vol: root finder did not converge.
    TooFewPaths,           ///< MC: too few independent samples.
    BadThreads,            ///< MC: threads == 0.
    McNotFinite,           ///< MC: estimate overflowed.
    SizeMismatch,          ///< batch: input/output spans differ in length.
};

/// A kernel error. Trivially copyable, never allocates.
struct Error {
    Errc code;
    const char* field = "";  ///< static string, never owned
    double a = 0.0, b = 0.0, c = 0.0, d = 0.0;
};

/// Result of a noexcept kernel.
template <class T>
using Result = qp::Result<T, Error>;

/// \brief Human-readable message, identical to the exception text of the
///        throwing API. Allocates; call it on the error path only.
std::string message(const Error& e);

/// \brief Throw the exception the original API threw for `e`:
///        std::runtime_error for NoConvergence, std::invalid_argument
///        otherwise.
[[noreturn]] void throw_error(const Error& e);

/// Unwrap a result or throw (qp/expected.hpp). Used by the throwing wrappers.
using qp::value_or_throw;

}  // namespace eqopt

#endif  // EQOPT_ERROR_HPP
