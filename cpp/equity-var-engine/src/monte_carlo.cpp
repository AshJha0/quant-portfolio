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

namespace {

std::size_t tail_rank(std::size_t n, double alpha) {
    const double mc = std::ceil(static_cast<double>(n) * alpha - 1e-9);
    return mc < 1.0 ? 1 : std::min(n, static_cast<std::size_t>(mc));
}

void require_finite_pnl(std::span<const double> pnl, const char* who) {
    for (double v : pnl) {
        if (!std::isfinite(v)) throw std::invalid_argument(std::string(who) + ": pnl contains NaN or inf");
    }
}

}  // namespace

double mc_bootstrap_se(std::span<const double> pnl, double alpha, std::size_t n_boot,
                       std::uint64_t seed) {
    validate_alpha(alpha);
    if (pnl.size() < 10) {
        throw std::invalid_argument("mc_bootstrap_se: need at least 10 observations, got " +
                                    std::to_string(pnl.size()));
    }
    if (n_boot < 2) throw std::invalid_argument("mc_bootstrap_se: n_boot must be >= 2");
    require_finite_pnl(pnl, "mc_bootstrap_se");
    const std::size_t n = pnl.size();
    const std::size_t m = tail_rank(n, alpha);
    RandomStream rng(seed);
    // One resample buffer reused per replicate: memory O(n), never n_boot x n.
    std::vector<double> loss(n);
    double sum = 0.0, sum_sq_dev = 0.0;  // Welford
    for (std::size_t b = 0; b < n_boot; ++b) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t idx =
                std::min(n - 1, static_cast<std::size_t>(rng.uniform() * static_cast<double>(n)));
            loss[i] = -pnl[idx];
        }
        std::nth_element(loss.begin(), loss.begin() + static_cast<std::ptrdiff_t>(m - 1), loss.end(),
                         std::greater<double>());
        const double v = loss[m - 1];
        const double prev_mean = (b == 0) ? 0.0 : sum / static_cast<double>(b);
        sum += v;
        const double mean = sum / static_cast<double>(b + 1);
        sum_sq_dev += (v - prev_mean) * (v - mean);
    }
    return std::sqrt(std::max(sum_sq_dev, 0.0) / static_cast<double>(n_boot - 1));
}

double kde_var_se(std::span<const double> pnl, double alpha) {
    validate_alpha(alpha);
    if (pnl.size() < 10) {
        throw std::invalid_argument("kde_var_se: need at least 10 observations, got " +
                                    std::to_string(pnl.size()));
    }
    require_finite_pnl(pnl, "kde_var_se");
    const std::size_t n = pnl.size();
    const double q = -var_es_order_statistic(pnl, alpha).var;  // reported S2 quantile (P&L units)
    double mu = 0.0;
    for (double v : pnl) mu += v;
    mu /= static_cast<double>(n);
    double ss = 0.0;
    for (double v : pnl) ss += (v - mu) * (v - mu);
    const double sd = std::sqrt(ss / static_cast<double>(n - 1));
    std::vector<double> work(pnl.begin(), pnl.end());
    const auto order_stat = [&work](std::size_t i) {
        std::nth_element(work.begin(), work.begin() + static_cast<std::ptrdiff_t>(i), work.end());
        return work[i];
    };
    const double q25 = order_stat(static_cast<std::size_t>(0.25 * static_cast<double>(n - 1)));
    const double q75 = order_stat(static_cast<std::size_t>(0.75 * static_cast<double>(n - 1)));
    double spread = sd;
    if (q75 - q25 > 0.0) spread = std::min(sd, (q75 - q25) / 1.34);
    if (!(spread > 0.0)) return 0.0;  // degenerate (zero-variance) sample
    const double h = 0.9 * spread * std::pow(static_cast<double>(n), -0.2);  // Silverman
    double dens = 0.0;
    for (double x : pnl) dens += normal_pdf((q - x) / h);
    dens /= static_cast<double>(n) * h;
    return std::sqrt(alpha * (1.0 - alpha) / static_cast<double>(n)) / std::max(dens, 1e-300);
}

MonteCarloResult mc_tail_metrics(std::span<const double> pnl, double alpha, std::size_t n_boot,
                                 std::uint64_t seed) {
    validate_alpha(alpha);
    if (pnl.size() < 100) {
        throw std::invalid_argument("mc_tail_metrics: need at least 100 scenarios, got " +
                                    std::to_string(pnl.size()));
    }
    require_finite_pnl(pnl, "mc_tail_metrics");
    const TailRisk tr = var_es_order_statistic(pnl, alpha);
    MonteCarloResult res;
    res.n_paths = pnl.size();
    res.var = tr.var;
    res.es = tr.es;
    res.var_se = mc_bootstrap_se(pnl, alpha, n_boot, seed);
    res.var_se_kde = kde_var_se(pnl, alpha);
    return res;
}

MonteCarloResult monte_carlo_var(std::span<const double> exposures, const Matrix& cov,
                                 double alpha, std::size_t n_paths, Dist dist, double df,
                                 std::uint64_t seed, std::size_t n_boot, Repair repair) {
    validate_exposures(exposures, "monte_carlo_var");
    if (!cov.square() || cov.rows() != exposures.size()) {
        throw std::invalid_argument("monte_carlo_var: covariance shape does not match exposures");
    }
    validate_alpha(alpha);
    // Simulate once; VaR, ES and both SEs come from the same scenarios.
    const Matrix scen = simulate_factor_returns(cov, n_paths, dist, df, seed, repair);
    const std::vector<double> pnl = portfolio_pnl(scen, exposures);
    return mc_tail_metrics(pnl, alpha, n_boot, seed ^ 0x9E3779B97F4A7C15ULL);
}

}  // namespace eqvar
