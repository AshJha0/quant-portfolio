// qp/black_scholes_vm.hpp: the carry-form Black-Scholes kernel (see
// qp/black_scholes.hpp) written with the branch-free exp, log and erfc of
// qp/vmath.hpp, for the engines' _fast batch loops.
//
// The plain idea. A batch loop vectorises only if every call in its body
// can be inlined and has no branches; libm's exp, log and erfc cannot. These
// functions are the per-element body the equity and FX _fast pricers share.
// Each engine keeps its own loop, its own test for which elements need the
// exact scalar formula, and its own target_clones attribute. Both bodies are
// force-inlined (QP_VM_INLINE): as ordinary inline functions GCC 15 kept them
// as calls and the loops ran scalar, about 10x slower (measured).
//
// The formal statement. Same formulas as qp/black_scholes.hpp with
// x = S / K passed in (the caller replaces it with 1 where it is not a normal
// double) and N(y) = 0.5 erfc1(-y * (1 / sqrt 2)), n(y) = (1 / sqrt(2 pi))
// exp1(-y^2 / 2). The results are not bit-identical to the libm path; the
// engines document and test the bound (1e-13 * max(1, S, K) for prices,
// 1e-12 * max(1, S, K, |exact|) per Greek on their test grids).
//
// Worked example. S = K = 100, T = 1, r = 5 %, q = 2 %, sigma = 20 %:
// carry_price_vm(100, 100, 1.0, 1, 0.05, 0.02, 0.2, +1) is 9.2270 to four
// decimals, within 1e-13 of carry_price() (tests/test_core.cpp, "carry").

#pragma once

#include <cmath>

#include "qp/vmath.hpp"

namespace qp {

namespace detail_vm {
inline constexpr double kRsqrt2 = 0x1.6a09e667f3bcdp-1;
inline constexpr double kInvSqrt2Pi = 0x1.9884533d43651p-2;
}  // namespace detail_vm

/// Price phi [S e^{-qT} N(phi d1) - K e^{-rT} N(phi d2)] with x = S / K.
[[nodiscard]] QP_VM_INLINE double carry_price_vm(double S, double K, double x, double T, double r,
                                                 double q, double sigma, double phi) noexcept {
    const double vst = sigma * std::sqrt(T);
    const double d1 = (vm::detail::log1(x) + (r - q + 0.5 * sigma * sigma) * T) / vst;
    const double d2 = d1 - vst;
    const double disc_s = S * vm::detail::exp1(-q * T);
    const double disc_k = K * vm::detail::exp1(-r * T);
    const double n1 = 0.5 * vm::detail::erfc1(-phi * d1 * detail_vm::kRsqrt2);
    const double n2 = 0.5 * vm::detail::erfc1(-phi * d2 * detail_vm::kRsqrt2);
    return phi * (disc_s * n1 - disc_k * n2);
}

/// Price and Greeks from the vmath kernels (fields as in qp::CarryGreeks).
struct CarryGreeksVm {
    double price, delta, delta_forward, gamma, vega, theta, rho, rho_q, vanna, volga;
};

/// Greeks with x = S / K; S, K, T, sigma > 0. Theta is the phi form.
[[nodiscard]] QP_VM_INLINE CarryGreeksVm carry_greeks_vm(double S, double K, double x, double T,
                                                         double r, double q, double sigma,
                                                         double phi) noexcept {
    const double sqrt_t = std::sqrt(T);
    const double vst = sigma * sqrt_t;
    const double d1 = (vm::detail::log1(x) + (r - q + 0.5 * sigma * sigma) * T) / vst;
    const double d2 = d1 - vst;
    const double df_q = vm::detail::exp1(-q * T);
    const double df_r = vm::detail::exp1(-r * T);
    const double pdf = detail_vm::kInvSqrt2Pi * vm::detail::exp1(-0.5 * d1 * d1);
    const double n1 = 0.5 * vm::detail::erfc1(-phi * d1 * detail_vm::kRsqrt2);
    const double n2 = 0.5 * vm::detail::erfc1(-phi * d2 * detail_vm::kRsqrt2);
    const double vega = S * df_q * pdf * sqrt_t;
    return CarryGreeksVm{
        phi * (S * df_q * n1 - K * df_r * n2),
        phi * df_q * n1,
        phi * n1,
        df_q * pdf / (S * vst),
        vega,
        -S * df_q * pdf * sigma / (2.0 * sqrt_t) + phi * (q * S * df_q * n1 - r * K * df_r * n2),
        phi * K * T * df_r * n2,
        -phi * S * T * df_q * n1,
        -df_q * pdf * d2 / sigma,
        vega * d1 * d2 / sigma,
    };
}

}  // namespace qp
