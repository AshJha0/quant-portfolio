// Structure-of-arrays batch pricer: Garman-Kohlhagen prices and Greeks over
// std::span inputs.
//
// Each element is priced with exactly the same inline kernel as
// try_gk_price() / try_analytic_greeks(), so every output equals the scalar
// result bit for bit. The batch runs in two passes:
//  1. validation over the whole batch, written without control flow so the
//     compiler vectorises it;
//  2. pricing, one element at a time. erfc, exp and log are scalar libm
//     calls; vector libm variants (glibc libmvec) are only enabled under
//     -ffast-math and do not return the same bits, so this loop is not
//     vectorised.
// If validation fails nothing is written and the error names the first bad
// index. No allocation, no exceptions.

#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "fxopt/common.hpp"
#include "fxopt/error.hpp"
#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/greeks.hpp"

namespace fxopt {

/// Batch inputs, one span per field, all the same length.
struct GkBatchInputs {
    std::span<const double> S, K, T, r_d, r_f, sigma;
    std::span<const OptionType> type;
};

/// Batch Greek outputs, one span per GreeksResult field, each as long as
/// the inputs.
struct GkGreeksBatchOutputs {
    std::span<double> price, delta_spot, delta_forward, gamma, vega, theta, rho_domestic,
        rho_foreign, vanna, volga;
};

/// Error of a batch call: the first offending element and its scalar error.
struct BatchError {
    std::size_t index;
    Error error;
};

/// Prices for every element; `out[i] == *try_gk_price(...)` exactly.
[[nodiscard]] std::expected<void, BatchError> gk_price_batch(const GkBatchInputs& in,
                                                             std::span<double> out) noexcept;

/// \brief Throughput form of gk_price_batch(): the same validation and errors,
///        with the pricing loop built on the branch-free exp, log and erfc of
///        qp/vmath.hpp so the compiler can vectorise it. Not bit-identical to
///        gk_price(); the tested bound is |fast - exact| <= 1e-13 * max(1, S, K).
///        Elements with T = 0 or sigma sqrt(T) <= kMinVol take the scalar
///        formula and match gk_price() bit for bit.
[[nodiscard]] std::expected<void, BatchError> gk_price_batch_fast(const GkBatchInputs& in,
                                                                  std::span<double> out) noexcept;

/// Greeks for every element; each field equals try_analytic_greeks()
/// exactly. Inputs must satisfy the analytic_greeks() domain.
[[nodiscard]] std::expected<void, BatchError> gk_greeks_batch(
    const GkBatchInputs& in, const GkGreeksBatchOutputs& out) noexcept;

}  // namespace fxopt
