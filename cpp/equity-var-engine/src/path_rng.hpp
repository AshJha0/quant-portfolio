// path_rng.hpp (private to eqvar): the W4(b) per-path Monte Carlo stream.
//
// Path p of a run with seed s draws from Philox4x32-10 stream (key = s,
// stream = p) (qp/philox.hpp; Salmon et al. 2011). Each 64-bit word gives
// one uniform ((w >> 11) + 0.5) 2^-53 on (0, 1); a normal is Wichura's
// AS 241 (PPND16, 1988) inverse CDF of that uniform. A path's n factor
// normals come first, then (Student-t) the Marsaglia-Tsang chi-square on
// the same stream. A path depends only on (s, p), so a run gives the same
// bits on any number of threads.
//
// The Rust twin is eq_var_engine::rng::PathRng with the same stream and the
// same operation order; the library is built with -ffp-contract=off so no
// multiply-add is fused and the two agree to the last bit on the same
// libm (std::log / std::sqrt only matter in the AS 241 tails and the gamma).

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "qp/path_rng.hpp"
#include "qp/philox.hpp"
#include "qp/rng.hpp"

namespace eqvar::detail {

using qp::as241_central_den;
using qp::as241_central_num;
using qp::as241_tail;
using qp::kPathLanes;
using qp::norm_ppf_as241;
using qp::PathRng;

inline double linear_path(const double* v, std::size_t n, bool student, double df,
                          std::uint64_t seed, std::uint64_t p) noexcept {
    PathRng rng(seed, p);
    double z[2 * kPathLanes];
    double acc = 0.0;
    for (std::size_t j0 = 0; j0 < n; j0 += 2 * kPathLanes) {
        const std::size_t len = n - j0 < 2 * kPathLanes ? n - j0 : 2 * kPathLanes;
        rng.fill_normals(z, len);
        for (std::size_t k = 0; k < len; ++k) acc += v[j0 + k] * z[k];
    }
    if (student) {
        const double wmix = rng.chi_squared(df) / df;
        acc *= 1.0 / std::sqrt(wmix);
    }
    return acc;
}

}  // namespace eqvar::detail
