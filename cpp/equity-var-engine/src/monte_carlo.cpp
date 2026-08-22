#include "eqvar/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "eqvar/expected_shortfall.hpp"
#include "eqvar/historical.hpp"
#include "eqvar/returns.hpp"
#include "eqvar/stats.hpp"

namespace {

// Type-7 linear-interpolation VaR quantile (positive for a loss) on an
// ALREADY-SORTED sample. Shared by mc_tail_metrics (point estimate) and
// mc_bootstrap_se (each bootstrap resample) so both use exactly the same
// quantile convention.
double var_from_sorted(const std::vector<double>& sorted, double alpha) {
    const std::size_t n = sorted.size();
    const double h = alpha * static_cast<double>(n - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(h));
    const double frac = h - static_cast<double>(lo);
    const double q = (lo + 1 < n) ? sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
                                  : sorted.back();
    return -q;
}

}  // namespace

namespace eqvar {

double RandomStream::uniform() {
    // 53-bit mantissa uniform on (0, 1): (x >> 11) / 2^53 shifted by half a
    // step so 0 and 1 are excluded (safe input for normal_ppf).
    const std::uint64_t bits = engine_() >> 11;
    return (static_cast<double>(bits) + 0.5) * 0x1.0p-53;
}

double RandomStream::gaussian() { return normal_ppf(uniform()); }

double RandomStream::gamma(double shape) {
    if (!(shape > 0.0)) throw std::invalid_argument("RandomStream::gamma: shape must be > 0");
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
    if (!(df > 0.0)) throw std::invalid_argument("RandomStream::chi_squared: df must be > 0");
    return 2.0 * gamma(0.5 * df);
}

Matrix simulate_factor_returns(const Matrix& cov, std::size_t n_paths, Dist dist, double df,
                               std::uint64_t seed) {
    if (n_paths < 1) throw std::invalid_argument("simulate_factor_returns: n_paths must be >= 1");
    const std::size_t n = cov.rows();
    Matrix scale = cov;
    if (dist == Dist::StudentT) {
        if (df <= 2.0) {
            throw std::invalid_argument(
                "simulate_factor_returns: Student-t df must be > 2 for finite variance, got " +
                std::to_string(df));
        }
        // Scale matrix cov * (df-2)/df so the simulated covariance equals cov.
        for (std::size_t i = 0; i < n * n; ++i) scale.data()[i] *= (df - 2.0) / df;
    }
    const Matrix chol = cholesky(scale).lower;  // jitter fallback inside
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

MonteCarloResult mc_tail_metrics(std::span<const double> pnl, double alpha) {
    validate_alpha(alpha);
    if (pnl.size() < 100) {
        throw std::invalid_argument("mc_tail_metrics: need at least 100 scenarios, got " +
                                    std::to_string(pnl.size()));
    }
    const std::size_t n = pnl.size();
    std::vector<double> sorted(pnl.begin(), pnl.end());
    std::sort(sorted.begin(), sorted.end());

    MonteCarloResult res;
    res.n_paths = n;

    // VaR: linear-interpolated (type 7) quantile on the sorted sample.
    const double h = alpha * static_cast<double>(n - 1);
    res.var = var_from_sorted(sorted, alpha);

    // ES: exact tail integral of the step CDF (same as expected_shortfall).
    const double an = alpha * static_cast<double>(n);
    const std::size_t k = static_cast<std::size_t>(std::floor(an));
    double tail_sum = 0.0;
    for (std::size_t i = 0; i < k; ++i) tail_sum += sorted[i];
    const double f2 = an - static_cast<double>(k);
    if (f2 > 0.0 && k < n) tail_sum += f2 * sorted[k];
    res.es = -tail_sum / an;

    // Order-statistic SE: asymptotic quantile variance alpha(1-alpha)/(n f^2)
    // with f estimated by a symmetric order-statistic finite difference of
    // bandwidth m = ceil(sqrt(alpha n)) around the quantile rank.
    const std::size_t rank = std::min(n - 1, static_cast<std::size_t>(std::llround(h)));
    const std::size_t m = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(std::sqrt(alpha * static_cast<double>(n)))));
    const std::size_t ilo = (rank > m) ? rank - m : 0;
    const std::size_t ihi = std::min(n - 1, rank + m);
    const double dx = sorted[ihi] - sorted[ilo];
    if (dx > 0.0) {
        const double f_hat = (static_cast<double>(ihi - ilo) / static_cast<double>(n)) / dx;
        res.var_se = std::sqrt(alpha * (1.0 - alpha) / static_cast<double>(n)) / f_hat;
    } else {
        res.var_se = 0.0;  // degenerate (e.g. zero-variance portfolio)
    }
    return res;
}

double mc_bootstrap_se(std::span<const double> pnl, double alpha, std::size_t n_boot,
                       std::uint64_t seed) {
    validate_alpha(alpha);
    if (pnl.size() < 10) {
        throw std::invalid_argument("mc_bootstrap_se: need at least 10 observations, got " +
                                    std::to_string(pnl.size()));
    }
    const std::size_t n = pnl.size();
    RandomStream rng(seed);
    std::vector<double> resample(n);
    std::vector<double> boot_vars(n_boot);
    for (std::size_t b = 0; b < n_boot; ++b) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t idx =
                std::min(n - 1, static_cast<std::size_t>(rng.uniform() * static_cast<double>(n)));
            resample[i] = pnl[idx];
        }
        std::sort(resample.begin(), resample.end());
        boot_vars[b] = var_from_sorted(resample, alpha);
    }
    // ddof = 1 sample standard deviation of the bootstrap VaR estimates,
    // matching numpy.std(..., ddof=1) in the Python reference.
    double sum = 0.0;
    for (double v : boot_vars) sum += v;
    const double m = sum / static_cast<double>(n_boot);
    double ss = 0.0;
    for (double v : boot_vars) {
        const double d = v - m;
        ss += d * d;
    }
    return std::sqrt(ss / static_cast<double>(n_boot - 1));
}

MonteCarloResult monte_carlo_var(std::span<const double> exposures, const Matrix& cov,
                                 double alpha, std::size_t n_paths, Dist dist, double df,
                                 std::uint64_t seed) {
    if (exposures.empty()) {
        throw std::invalid_argument("monte_carlo_var: empty portfolio (no exposures)");
    }
    if (!cov.square() || cov.rows() != exposures.size()) {
        throw std::invalid_argument("monte_carlo_var: covariance shape does not match exposures");
    }
    validate_alpha(alpha);
    const Matrix scen = simulate_factor_returns(cov, n_paths, dist, df, seed);
    const std::vector<double> pnl = portfolio_pnl(scen, exposures);
    return mc_tail_metrics(pnl, alpha);
}

}  // namespace eqvar
