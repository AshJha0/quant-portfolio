// qp/simd.hpp: the optional std::simd path for qp::matvec_rows().
//
// QP_HAS_SIMD is 1 when a vector type is available and enabled:
//   * a C++26 build (QP_CXX26, which defines QP_ENABLE_STD_SIMD) whose
//     standard library reports __cpp_lib_simd: std::simd::vec<double>
//     (or std::simd<double> for the pre-2025 spelling);
//   * QP_SIMD_EXPERIMENTAL (test builds only) with the Parallelism TS 2
//     <experimental/simd> that GCC ships: std::experimental::native_simd.
// Otherwise it is 0 and only the scalar loop is compiled.
//
// Bit identity. Lane k of the vector path accumulates row i + k with the
// same operations in the same order as the scalar loop: acc = acc + a*x,
// one rounding for the product and one for the sum, j = 0 .. cols-1. The
// product and the sum are separate statements, so the compiler cannot
// contract them into a fused multiply-add. The scalar loop is not
// contracted either on the default target (x86-64 without FMA); a build
// with -march=native may contract the scalar loop and then the two paths
// can differ in the last bit, as the engines' NATIVE options already state.

#pragma once

#include <cstddef>

#if defined(QP_ENABLE_STD_SIMD) && defined(__has_include)
#if __has_include(<simd>)
#include <simd>
#endif
#endif

#if defined(QP_ENABLE_STD_SIMD) && defined(__cpp_lib_simd)
#define QP_HAS_SIMD 1
#elif defined(QP_SIMD_EXPERIMENTAL) && defined(__has_include)
#if __has_include(<experimental/simd>)
#include <experimental/simd>
#define QP_HAS_SIMD 1
#define QP_SIMD_IS_EXPERIMENTAL 1
#else
#define QP_HAS_SIMD 0
#endif
#else
#define QP_HAS_SIMD 0
#endif

#if QP_HAS_SIMD
namespace qp::simd_detail {

#if defined(QP_SIMD_IS_EXPERIMENTAL)
using vdouble = std::experimental::native_simd<double>;
#elif __cpp_lib_simd >= 202506L
using vdouble = std::simd::vec<double>;
#else
using vdouble = std::simd<double>;
#endif

/// Rows [0, i) of y = A x, W rows per step; returns i (a multiple of W).
inline std::size_t matvec_rows_simd(const double* d, std::size_t rows, std::size_t cols,
                                    const double* xv, double* y) noexcept {
    constexpr std::size_t kW = vdouble::size();
    std::size_t i = 0;
    for (; i + kW <= rows; i += kW) {
        const double* base = d + i * cols;
        vdouble acc(0.0);
        for (std::size_t j = 0; j < cols; ++j) {
            const vdouble col([base, cols, j](auto k) {
                return base[static_cast<std::size_t>(k) * cols + j];
            });
            const vdouble prod = col * vdouble(xv[j]);
            acc = acc + prod;
        }
        for (std::size_t k = 0; k < kW; ++k) y[i + k] = acc[static_cast<int>(k)];
    }
    return i;
}

}  // namespace qp::simd_detail
#endif
