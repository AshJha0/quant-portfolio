#include "eqvar/monte_carlo.hpp"

#include <algorithm>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "eqvar/expected_shortfall.hpp"
#include "eqvar/historical.hpp"
#include "eqvar/returns.hpp"
#include "eqvar/stats.hpp"

namespace eqvar {

double RandomStream::uniform() {
    // 53-bit mantissa uniform on (0, 1): (x >> 11) / 2^53 shifted by half a
    // step so 0 and 1 are excluded (safe input for normal_ppf).
    const std::uint64_t bits = engine_() >> 11;
    return (static_cast<double>(bits) + 0.5) * 0x1.0p-53;
}

double RandomStream::gaussian() { return normal_ppf(uniform()); }

double RandomStream::gamma(double shape) {
    if (!(shape > 0.0) || !std::isfinite(shape)) {
        throw std::invalid_argument("RandomStream::gamma: shape must be finite and > 0");
    }
    if (shape < 1.0) {
        // Boost: Gamma(a) = Gamma(a + 1) * U^{1/a}.
        const double u = uniform();
        return gamma(shape + 1.0) * std::pow(u, 1.0 / shape);
    }
    // Marsaglia-Tsang (2000) squeeze method.
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            x = gaussian();
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        const double u = uniform();
        const double x2 = x * x;
        if (u < 1.0 - 0.0331 * x2 * x2) return d * v;
        if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) return d * v;
    }
}

double RandomStream::chi_squared(double df) {
    if (!(df > 0.0) || !std::isfinite(df)) {
        throw std::invalid_argument("RandomStream::chi_squared: df must be finite and > 0");
    }
    return 2.0 * gamma(0.5 * df);
}

Matrix simulate_factor_returns(const Matrix& cov, std::size_t n_paths, Dist dist, double df,
                               std::uint64_t seed, Repair repair) {
    if (n_paths < 1) throw std::invalid_argument("simulate_factor_returns: n_paths must be >= 1");
    const std::size_t n = cov.rows();
    Matrix scale = cov;
    if (dist == Dist::StudentT) {
        if (!(df > 2.0) || !std::isfinite(df)) {
            throw std::invalid_argument(
                "simulate_factor_returns: Student-t df must be > 2 for finite variance, got " +
                std::to_string(df));
        }
        // Scale matrix cov * (df-2)/df so the simulated covariance equals cov.
        for (std::size_t i = 0; i < n * n; ++i) scale.data()[i] *= (df - 2.0) / df;
    }
    const Matrix chol = cholesky(scale, repair).lower;  // capped jitter, optional repair
    RandomStream rng(seed);
    Matrix out(n_paths, n);
    std::vector<double> z(n);
    for (std::size_t p = 0; p < n_paths; ++p) {
        for (std::size_t j = 0; j < n; ++j) z[j] = rng.gaussian();
        // r = L z (lower-triangular product; no allocation in the hot loop).
        double* row = out.data() + p * n;
        for (std::size_t i = 0; i < n; ++i) {
            const double* li = chol.data() + i * n;
            double s = 0.0;
            for (std::size_t j = 0; j <= i; ++j) s += li[j] * z[j];
            row[i] = s;
        }
        if (dist == Dist::StudentT) {
            const double w = rng.chi_squared(df) / df;
            const double m = 1.0 / std::sqrt(w);
            for (std::size_t i = 0; i < n; ++i) row[i] *= m;
        }
    }
    return out;
}






}  // namespace eqvar
