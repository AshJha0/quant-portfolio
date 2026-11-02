/// \file batch.hpp
/// \brief Structure-of-arrays batch pricer: Black-Scholes-Merton prices
///        and Greeks over std::span inputs.
///
/// Each element is priced with exactly the same inline kernel as
/// try_bs_price() / try_bs_greeks(), so every output equals the scalar
/// result bit for bit. The batch runs in two passes:
///  1. validation over the whole batch, written as a branch-free reduction
///     that the compiler vectorises;
///  2. pricing, one element at a time. erfc, exp and log are scalar libm
///     calls; vector libm variants (glibc libmvec) are only enabled under
///     -ffast-math and do not return the same bits, so this loop is not
///     vectorised by design.
/// If validation fails nothing is written and the error names the first bad
/// index. No allocation, no exceptions.

#ifndef EQOPT_BATCH_HPP
#define EQOPT_BATCH_HPP

#include <cstddef>
#include <expected>
#include <span>

#include "eqopt/black_scholes.hpp"
#include "eqopt/error.hpp"
#include "eqopt/greeks.hpp"

namespace eqopt {

/// Batch inputs, one span per field, all the same length.
struct BsBatchInputs {
    std::span<const double> S, K, T, r, sigma, q;
    std::span<const OptionType> type;
};

/// Batch Greek outputs, one span per field, each as long as the inputs.
struct BsGreeksBatchOutputs {
    std::span<double> price, delta, gamma, vega, theta, rho, vanna, volga;
};

/// Error of a batch call: the first offending element and its scalar error.
struct BatchError {
    std::size_t index;
    Error error;
};

/// \brief Prices for every element; `out[i] == *try_bs_price(...)` exactly.
[[nodiscard]] std::expected<void, BatchError> bs_price_batch(const BsBatchInputs& in,
                                                             std::span<double> out) noexcept;

/// \brief Throughput form of bs_price_batch(): the same validation and the
///        same errors, but the pricing loop uses the branch-free exp, log and
///        erfc of qp/vmath.hpp so the compiler can vectorise it. Prices are
///        NOT bit-identical to bs_price(); docs/METHODOLOGY.md states the
///        tested bound (|fast - exact| <= 1e-13 * max(1, S, K) on the test
///        grids). Elements with T, sigma, S or K equal to zero go through the
///        scalar formula and match bs_price() bit for bit.
[[nodiscard]] std::expected<void, BatchError> bs_price_batch_fast(const BsBatchInputs& in,
                                                                  std::span<double> out) noexcept;

/// \brief Greeks for every element; each field equals try_bs_greeks()
///        exactly. Inputs must satisfy the bs_greeks() domain
///        (S, K, T, sigma > 0; r, q finite).
[[nodiscard]] std::expected<void, BatchError> bs_greeks_batch(
    const BsBatchInputs& in, const BsGreeksBatchOutputs& out) noexcept;

}  // namespace eqopt

#endif  // EQOPT_BATCH_HPP
