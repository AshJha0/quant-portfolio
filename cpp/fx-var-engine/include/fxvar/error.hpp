// Error values for the noexcept kernel API (fxvar/kernels.hpp).
//
// Every try_* kernel is noexcept and returns std::expected<T, Error>. An
// Error is a code, a pointer to a string that outlives the call (a string
// literal, or a factor name owned by the CompiledBook) and two numbers, so
// building one never allocates. message() formats the text on demand; the
// throwing API uses it and keeps its exception type (std::invalid_argument)
// and message text unchanged.

#pragma once

#include <cstdint>
#include <expected>
#include <string>

#include <type_traits>
#include <utility>
#include "qp/expected.hpp"

namespace fxvar {

enum class Errc : std::uint8_t {
  kFixed,              ///< `text` is the whole message.
  kAlphaRange,         ///< alpha (a) outside (0, 0.5).
  kHorizonRange,       ///< horizon_days (a) not finite and > 0.
  kShockNonFinite,     ///< shock for factor `text` is NaN or Inf.
  kWorkspaceTooSmall,  ///< `text`: workspace (a) smaller than needed (b).
  kSizeMismatch,       ///< `text`: span lengths a and b differ.
};

struct Error {
  Errc code = Errc::kFixed;
  const char* text = "";  ///< never owned
  double a = 0.0, b = 0.0;
};

template <class T>
using Result = qp::Result<T, Error>;

/// Shorthand for a fixed-text error.
inline std::unexpected<Error> fail(const char* text) noexcept {
  return std::unexpected(Error{Errc::kFixed, text});
}

/// Message text, identical to the throwing API's exception text. Allocates;
/// call it on the error path only.
std::string message(const Error& e);

/// Throw std::invalid_argument(message(e)).
[[noreturn]] void throw_error(const Error& e);

/// Unwrap a result or throw (qp/expected.hpp). Used by the throwing wrappers.
using qp::value_or_throw;

}  // namespace fxvar
