// qp/rng.hpp: seeding and variate transforms shared by the Monte Carlo code.
//
// Every engine draws raw 64-bit words from std::mt19937_64, whose output
// sequence the C++ standard fixes, and turns them into variates with the
// transforms below. No std::*_distribution is used, so a seed gives the same
// stream on every compiler and standard library.

#pragma once

#include <cmath>
#include <cstdint>

#include "qp/constants.hpp"

namespace qp {

/// splitmix64 finaliser (Steele, Lea and Flood 2014). Derives independent
/// per-chunk seeds from (base seed, chunk index) deterministically.
[[nodiscard]] constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

/// 53-bit uniform on the open interval (0, 1): ((u >> 11) + 0.5) 2^-53.
/// Never 0 or 1, so it is a safe input for an inverse cdf.
[[nodiscard]] constexpr double uniform_open53(std::uint64_t u) noexcept {
    return (static_cast<double>(u >> 11) + 0.5) * kTwoPowMinus53;
}

/// Gamma(shape, 1) by Marsaglia and Tsang (2000), with the U^(1/shape)
/// boost for shape < 1. `uniform()` returns a uniform on (0, 1) and
/// `normal()` a standard normal; both come from the caller's stream, so the
/// draw order is part of the result. The caller validates shape > 0.
template <class Uniform, class Normal>
[[nodiscard]] double gamma_marsaglia_tsang(double shape, Uniform& uniform, Normal& normal) {
    if (shape < 1.0) {
        // Boost: Gamma(a) = Gamma(a + 1) * U^(1/a).
        const double u = uniform();
        return gamma_marsaglia_tsang(shape + 1.0, uniform, normal) * std::pow(u, 1.0 / shape);
    }
    // Squeeze method.
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        double x = 0.0;
        double v = 0.0;
        do {
            x = normal();
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        const double u = uniform();
        const double x2 = x * x;
        if (u < 1.0 - 0.0331 * x2 * x2) return d * v;
        if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) return d * v;
    }
}

}  // namespace qp
