// qp/gemm.hpp: cache-blocked matrix products that keep the naive bits.
//
// Blocking (Lam, Rothberg and Wolf, "The cache performance and optimizations
// of blocked algorithms", ASPLOS 1991) tiles the i, j and k loops so a tile
// of each operand stays in cache. Here the k-tiles are visited in ascending
// order and k ascends inside a tile, and every C element is updated in
// place, so each C(i, j) receives exactly the sequence of roundings
// c = c + a(i, k) b(k, j), k = 0, 1, ..., of the textbook triple loop. The
// product and the sum are separate statements (no fused multiply-add on the
// default target). The result is therefore the same double as the naive
// loop, which the core tests check bit for bit; only the memory traffic
// changes.

#pragma once

#include <algorithm>
#include <cstddef>

#include "qp/config.hpp"

namespace qp {

inline constexpr std::size_t kGemmTile = 64;

/// C += A B with A m x k, B k x n, C m x n, all row-major and dense.
inline void gemm_blocked(const double* a, const double* b, double* c, std::size_t m, std::size_t k,
                         std::size_t n) noexcept
    QP_PRE((a != nullptr && b != nullptr && c != nullptr) || m * k * n == 0) {
    constexpr std::size_t T = kGemmTile;
    for (std::size_t kb = 0; kb < k; kb += T) {
        const std::size_t ke = std::min(kb + T, k);
        for (std::size_t ib = 0; ib < m; ib += T) {
            const std::size_t ie = std::min(ib + T, m);
            for (std::size_t jb = 0; jb < n; jb += T) {
                const std::size_t je = std::min(jb + T, n);
                for (std::size_t i = ib; i < ie; ++i) {
                    double* ci = c + i * n;
                    for (std::size_t p = kb; p < ke; ++p) {
                        const double aip = a[i * k + p];
                        const double* bp = b + p * n;
                        for (std::size_t j = jb; j < je; ++j) {
                            const double prod = aip * bp[j];
                            ci[j] = ci[j] + prod;
                        }
                    }
                }
            }
        }
    }
}

/// Upper triangle (j >= i) of C += D^T D, D t x n row-major, C n x n.
/// Each C(i, j) accumulates d(r, i) d(r, j) for r = 0 .. t-1 in order, so it
/// equals the engines' row-streaming covariance loop bit for bit.
inline void syrk_upper_blocked(const double* d, double* c, std::size_t t, std::size_t n) noexcept
    QP_PRE((d != nullptr && c != nullptr) || t * n == 0) {
    constexpr std::size_t T = kGemmTile;
    for (std::size_t rb = 0; rb < t; rb += T) {
        const std::size_t re = std::min(rb + T, t);
        for (std::size_t ib = 0; ib < n; ib += T) {
            const std::size_t ie = std::min(ib + T, n);
            for (std::size_t jb = ib; jb < n; jb += T) {
                const std::size_t je = std::min(jb + T, n);
                for (std::size_t r = rb; r < re; ++r) {
                    const double* dr = d + r * n;
                    for (std::size_t i = ib; i < ie; ++i) {
                        const double di = dr[i];
                        double* ci = c + i * n;
                        for (std::size_t j = std::max(jb, i); j < je; ++j) {
                            const double prod = di * dr[j];
                            ci[j] = ci[j] + prod;
                        }
                    }
                }
            }
        }
    }
}

}  // namespace qp
