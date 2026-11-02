// qp/vmath.hpp: branch-free batch exp, erfc and normal cdf over SoA spans.
//
// These are fast paths, not replacements. The scalar functions in
// qp/normal.hpp (std::erfc) stay the default everywhere and every golden is
// computed with them. A caller that wants throughput over a structure of
// arrays (one std::span per field) can use these and accept the documented
// tolerance, which the core tests check:
//
//   exp_batch       |rel err| <= 2 ulp for x in [-708, 709]      (tested 4e-16)
//   erfc_batch      |rel err| <= 2e-15 for x in [-6, 26.5]       (measured 1.0e-15)
//   norm_cdf_batch  |abs err| <= 4e-16, |rel err| <= 2e-15 for x >= -37 (2.8e-16, 1.1e-15)
//
// The loops have no data-dependent branch: every lane runs the same
// instructions (clamps and selects compile to min/max/blend), so GCC and
// Clang vectorise them at -O3 on SSE2 and wider without intrinsics. Inputs
// must be finite; NaN is not propagated reliably (min/max clamps).
//
// exp: Cody and Waite range reduction (W. J. Cody and W. Waite, Software
// Manual for the Elementary Functions, 1980, ch. 6) with the fdlibm split of
// ln 2, x = n ln 2 + r, |r| <= ln2 / 2, then a degree-13 Taylor polynomial
// (truncation below 4e-18 relative), then 2^n applied as two powers of two
// built from bits so that subnormal results come out right.
//
// erfc: for z = |x|, erfc(z) = t exp(-z^2 + g(t)) with t = 2 / (2 + z)
// (Press, Teukolsky, Vetterling and Flannery, Numerical Recipes, 3rd ed.,
// 2007, section 6.2.2). g is a 26-term Chebyshev series on z in [0, 27],
// generated in long double by tools/gen_erfc_cheb.cpp. For x < 0,
// erfc(x) = 2 - erfc(-x). z^2 is split exactly (Cody's trick), so the cost
// is two exps per element and the error stays near the exp error.

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

namespace qp::vm {

namespace detail {

inline constexpr double kLog2e = 0x1.71547652b82fep0;
inline constexpr double kLn2Hi = 0x1.62e42fee00000p-1;
inline constexpr double kLn2Lo = 0x1.a39ef35793c76p-33;
inline constexpr double kMagic = 0x1.8p52;  // round-to-nearest-integer shifter
inline constexpr double kExpLo = -746.0;    // below this exp underflows to 0
inline constexpr double kExpHi = 710.0;     // above this exp overflows to inf

/// 2^k for an integer-valued double k in [-1022, 1023], from bits.
[[nodiscard]] inline double pow2_int(double k) noexcept {
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(k + kMagic) -
                               std::bit_cast<std::uint64_t>(kMagic) + 1023U;
    return std::bit_cast<double>(bits << 52);
}

[[nodiscard]] inline double exp1(double x) noexcept {
    x = std::min(std::max(x, kExpLo), kExpHi);
    const double n = (x * kLog2e + kMagic) - kMagic;
    const double r = (x - n * kLn2Hi) - n * kLn2Lo;
    // exp(r) = sum r^k / k!, k = 0..13, Horner.
    double p = 1.0 / 6227020800.0;
    p = p * r + 1.0 / 479001600.0;
    p = p * r + 1.0 / 39916800.0;
    p = p * r + 1.0 / 3628800.0;
    p = p * r + 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    // n in [-1076, 1025]: split so both factors are normal powers of two.
    const double n1 = ((n * 0.5) + kMagic) - kMagic;
    const double n2 = n - n1;
    return (p * pow2_int(n1)) * pow2_int(n2);
}

// Chebyshev coefficients of g(u); see tools/gen_erfc_cheb.cpp.
inline constexpr double kErfcT0 = 2.0 / 29.0;
inline constexpr double kErfcCheb[] = {
    -0x1.38b2eee989714p-1, 0x1.35f18ad2ce0ffp-1, 0x1.c7658ee928cd8p-7, -0x1.0ed39819f4a28p-7,
    -0x1.1c2616503568bp-11, 0x1.2e974e1ab12f5p-12, 0x1.067a324226513p-16, -0x1.cf293647bddcp-17,
    -0x1.20ba820a5fp-28, 0x1.8073bf7bf0a6p-21, -0x1.eff91eecb86p-25, -0x1.2726773d45cp-25,
    0x1.d06e903265p-28, 0x1.4e2cb79bbcp-30, -0x1.260006d808p-31, -0x1.2a97a798p-39,
    0x1.120be3dp-35, -0x1.2f0b8fcp-38, -0x1.3e198ecp-40, 0x1.d67a0f4p-42,
    -0x1.9b2d2p-48, -0x1.843ebcp-46, 0x1.29619p-48, 0x1.752cp-52,
    -0x1.34708p-52, 0x1.99b4p-55,
};

[[nodiscard]] inline double erfc1(double x) noexcept {
    const double z = std::min(x < 0.0 ? -x : x, 28.0);
    const double t = 2.0 / (2.0 + z);
    const double u = (2.0 * t - 1.0 - kErfcT0) / (1.0 - kErfcT0);
    // Clenshaw recurrence for sum c_k T_k(u).
    constexpr std::size_t kN = std::size(kErfcCheb);
    double b1 = 0.0, b2 = 0.0;
    for (std::size_t k = kN - 1; k >= 1; --k) {
        const double b0 = 2.0 * u * b1 - b2 + kErfcCheb[k];
        b2 = b1;
        b1 = b0;
    }
    const double g = kErfcCheb[0] + u * b1 - b2;
    // z^2 = zs^2 + d (z + zs) with zs = z rounded to 1/16: zs^2 is exact and
    // the second term is small, so no rounding of a large z^2 enters (the
    // split Cody uses in his erfc, Math. Comp. 23, 1969).
    const double zs = ((z * 16.0 + kMagic) - kMagic) * 0.0625;
    const double d = z - zs;
    const double r = t * exp1(-zs * zs) * exp1(g - d * (z + zs));
    return x < 0.0 ? 2.0 - r : r;
}

}  // namespace detail

/// out[i] = exp(x[i]). Sizes must match.
inline void exp_batch(std::span<const double> x, std::span<double> out) noexcept {
    const std::size_t n = std::min(x.size(), out.size());
    for (std::size_t i = 0; i < n; ++i) out[i] = detail::exp1(x[i]);
}

/// out[i] = erfc(x[i]). Sizes must match.
inline void erfc_batch(std::span<const double> x, std::span<double> out) noexcept {
    const std::size_t n = std::min(x.size(), out.size());
    for (std::size_t i = 0; i < n; ++i) out[i] = detail::erfc1(x[i]);
}

/// out[i] = N(x[i]) = erfc(-x[i] / sqrt 2) / 2. Sizes must match.
inline void norm_cdf_batch(std::span<const double> x, std::span<double> out) noexcept {
    constexpr double kRsqrt2 = 0x1.6a09e667f3bcdp-1;
    const std::size_t n = std::min(x.size(), out.size());
    for (std::size_t i = 0; i < n; ++i) out[i] = 0.5 * detail::erfc1(-x[i] * kRsqrt2);
}

}  // namespace qp::vm
