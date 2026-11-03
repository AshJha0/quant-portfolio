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

#include "qp/philox.hpp"
#include "qp/rng.hpp"

namespace eqvar::detail {

inline double as241_central_num(double r) noexcept {
    return ((((((2.5090809287301226727e3 * r + 3.3430575583588128105e4) * r +
                6.7265770927008700853e4) *
                   r +
               4.5921953931549871457e4) *
                  r +
              1.3731693765509461125e4) *
                 r +
             1.9715909503065514427e3) *
                r +
            1.3314166789178437745e2) *
               r +
           3.3871328727963666080e0;
}

inline double as241_central_den(double r) noexcept {
    return ((((((5.2264952788528545610e3 * r + 2.8729085735721942674e4) * r +
                3.9307895800092710610e4) *
                   r +
               2.1213794301586595867e4) *
                  r +
              5.3941960214247511077e3) *
                 r +
             6.8718700749205790830e2) *
                r +
            4.2313330701600911252e1) *
               r +
           1.0;
}

inline double as241_tail(double p, double q) noexcept {
    double r = q < 0.0 ? p : 1.0 - p;
    r = std::sqrt(-std::log(r));
    double val = 0.0;
    if (r <= 5.0) {
        r -= 1.6;
        const double num = ((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r +
                                2.41780725177450611770e-1) *
                                   r +
                               1.27045825245236838258e0) *
                                  r +
                              3.64784832476320460504e0) *
                                 r +
                             5.76949722146069140550e0) *
                                r +
                            4.63033784615654529590e0) *
                               r +
                           1.42343711074968357734e0;
        const double den = ((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r +
                                1.51986665636164571966e-2) *
                                   r +
                               1.48103976427480074590e-1) *
                                  r +
                              6.89767334985100004550e-1) *
                                 r +
                             1.67638483018380384940e0) *
                                r +
                            2.05319162663775882187e0) *
                               r +
                           1.0;
        val = num / den;
    } else {
        r -= 5.0;
        const double num = ((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r +
                                1.24266094738807843860e-3) *
                                   r +
                               2.65321895265761230930e-2) *
                                  r +
                              2.96560571828504891230e-1) *
                                 r +
                             1.78482653991729133580e0) *
                                r +
                            5.46378491116411436990e0) *
                               r +
                           6.65790464350110377720e0;
        const double den = ((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r +
                                1.84631831751005468180e-5) *
                                   r +
                               7.86869131145613259100e-4) *
                                  r +
                              1.48753612908506148525e-2) *
                                 r +
                             1.36929880922735805310e-1) *
                                r +
                            5.99832206555887937690e-1) *
                               r +
                           1.0;
        val = num / den;
    }
    return q < 0.0 ? -val : val;
}

/// Inverse standard normal CDF, AS 241 (PPND16). p in (0, 1).
inline double norm_ppf_as241(double p) noexcept {
    const double q = p - 0.5;
    if (std::abs(q) <= 0.425) {
        const double r = 0.180625 - q * q;
        return q * as241_central_num(r) / as241_central_den(r);
    }
    return as241_tail(p, q);
}

/// Philox blocks computed side by side by PathRng::fill_normals.
inline constexpr std::size_t kPathLanes = 16;

class PathRng {
public:
    PathRng(std::uint64_t seed, std::uint64_t path) noexcept
        : seed_(seed), path_(path), stream_(seed, path) {}

    [[nodiscard]] double uniform() noexcept {
        ++words_;
        return stream_.uniform();
    }

    [[nodiscard]] double normal() noexcept { return norm_ppf_as241(uniform()); }

    /// `len <= 2 * kPathLanes` normals into z: bit-identical to `len` calls
    /// of normal(), computed in a batch (Philox rounds on kPathLanes counters
    /// at once, then the branch-free central AS 241 branch for every lane
    /// and the tail branch only where |p - 0.5| > 0.425).
    void fill_normals(double* z, std::size_t len) noexcept {
        if (words_ % 2 != 0) {
            for (std::size_t k = 0; k < len; ++k) z[k] = normal();
            return;
        }
        philox_uniforms(words_ / 2, z, len);
        double p[2 * kPathLanes];
        bool tail = false;
        for (std::size_t k = 0; k < len; ++k) {
            p[k] = z[k];
            const double q = p[k] - 0.5;
            const double r = 0.180625 - q * q;
            z[k] = q * as241_central_num(r) / as241_central_den(r);
            tail = tail || std::abs(q) > 0.425;
        }
        if (tail) {
            for (std::size_t k = 0; k < len; ++k) {
                if (std::abs(p[k] - 0.5) > 0.425) z[k] = as241_tail(p[k], p[k] - 0.5);
            }
        }
        words_ += len;
        stream_.seek(words_);
    }

    /// Chi-squared(df) for df >= 2 (shape >= 1); NaN otherwise, without
    /// consuming the stream. Never throws.
    [[nodiscard]] double chi_squared(double df) noexcept {
        const double shape = 0.5 * df;
        if (!(shape >= 1.0) || !std::isfinite(shape)) return std::nan("");
        auto u = [this]() noexcept { return uniform(); };
        auto z = [this]() noexcept { return normal(); };
        return 2.0 * qp::gamma_marsaglia_tsang(shape, u, z);
    }

private:
    void philox_uniforms(std::uint64_t block, double* out, std::size_t len) const noexcept {
        constexpr std::uint64_t kM0 = 0xD2511F53U;
        constexpr std::uint64_t kM1 = 0xCD9E8D57U;
        std::uint32_t c0[kPathLanes], c1[kPathLanes], c2[kPathLanes], c3[kPathLanes];
        for (std::size_t l = 0; l < kPathLanes; ++l) {
            const std::uint64_t b = block + l;
            c0[l] = static_cast<std::uint32_t>(path_);
            c1[l] = static_cast<std::uint32_t>(path_ >> 32);
            c2[l] = static_cast<std::uint32_t>(b);
            c3[l] = static_cast<std::uint32_t>(b >> 32);
        }
        auto k0 = static_cast<std::uint32_t>(seed_);
        auto k1 = static_cast<std::uint32_t>(seed_ >> 32);
        for (int r = 0; r < 10; ++r) {
            if (r > 0) {
                k0 += 0x9E3779B9U;
                k1 += 0xBB67AE85U;
            }
            for (std::size_t l = 0; l < kPathLanes; ++l) {
                const std::uint64_t p0 = kM0 * c0[l];
                const std::uint64_t p1 = kM1 * c2[l];
                const auto n0 = static_cast<std::uint32_t>(p1 >> 32) ^ c1[l] ^ k0;
                const auto n2 = static_cast<std::uint32_t>(p0 >> 32) ^ c3[l] ^ k1;
                c0[l] = n0;
                c1[l] = static_cast<std::uint32_t>(p1);
                c2[l] = n2;
                c3[l] = static_cast<std::uint32_t>(p0);
            }
        }
        for (std::size_t k = 0; k < len; ++k) {
            const std::size_t l = k / 2;
            const std::uint64_t w = k % 2 == 0 ? (static_cast<std::uint64_t>(c1[l]) << 32) | c0[l]
                                               : (static_cast<std::uint64_t>(c3[l]) << 32) | c2[l];
            out[k] = qp::uniform_open53(w);
        }
    }

    std::uint64_t seed_;
    std::uint64_t path_;
    qp::PhiloxStream stream_;
    std::uint64_t words_ = 0;
};

/// Scenario P&L of path p for the linear kernel: m_p v . z_p, accumulated
/// left to right over j (the Rust twin forms z first; same order, same sum).
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
