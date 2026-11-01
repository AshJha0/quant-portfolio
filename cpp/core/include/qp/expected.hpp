// qp/expected.hpp: the error plumbing shared by the noexcept kernel APIs.
//
// Every engine defines its own `Error` (a code plus a few numbers, never
// allocating) and `throw_error(const Error&)`, because the codes and the
// exception texts belong to the engine. What is common is the shape:
// kernels return std::expected<T, Error>, and the throwing API unwraps the
// result with value_or_throw, which finds the engine's throw_error by
// argument-dependent lookup.

#pragma once

#include <expected>
#include <type_traits>
#include <utility>

namespace qp {

/// Result of a noexcept kernel whose error type is E.
template <class T, class E>
using Result = std::expected<T, E>;

/// Unwrap a result or throw. `throw_error(e)` is looked up in E's namespace.
template <class T, class E>
T value_or_throw(std::expected<T, E>&& r) {
    if (!r) [[unlikely]] throw_error(r.error());
    if constexpr (!std::is_void_v<T>) return std::move(*r);
}

}  // namespace qp
