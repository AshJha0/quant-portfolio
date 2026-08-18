#include "eqvar/historical.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

#include "eqvar/returns.hpp"

namespace eqvar {

double quantile_linear(std::span<const double> x, double q) {
    if (x.empty()) throw std::invalid_argument("quantile_linear: empty sample");
    if (!(q >= 0.0 && q <= 1.0)) {
        throw std::invalid_argument("quantile_linear: q must be in [0, 1], got " +
                                    std::to_string(q));
    }
    std::vector<double> sorted(x.begin(), x.end());
    std::sort(sorted.begin(), sorted.end());
    // NumPy default "linear" (Hyndman-Fan type 7) interpolation.
    const double h = q * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(h));
    if (lo + 1 >= sorted.size()) return sorted.back();
    const double frac = h - static_cast<double>(lo);
    return sorted[lo] + frac * (sorted[lo + 1] - sorted[lo]);
}

double historical_var(std::span<const double> pnl, double alpha) {
    validate_alpha(alpha);
    validate_pnl(pnl, kMinHistObs);
    return -quantile_linear(pnl, alpha);
}

std::vector<double> brw_weights(std::size_t n, double lam) {
    if (!(lam > 0.0 && lam < 1.0)) {
        throw std::invalid_argument("brw_weights: decay lam must be in (0, 1), got " +
                                    std::to_string(lam));
    }
    if (n < 1) throw std::invalid_argument("brw_weights: n must be >= 1");
    // w_i = (1 - lam) lam^{age} / (1 - lam^n), age = n-1-i (0 = oldest).
    std::vector<double> w(n);
    const double norm = (1.0 - lam) / (1.0 - std::pow(lam, static_cast<double>(n)));
    for (std::size_t i = 0; i < n; ++i) {
        w[i] = norm * std::pow(lam, static_cast<double>(n - 1 - i));
    }
    return w;
}

double age_weighted_var(std::span<const double> pnl, double alpha, double lam) {
    validate_alpha(alpha);
    validate_pnl(pnl, kMinHistObs);
    const std::vector<double> w = brw_weights(pnl.size(), lam);
    // Stable argsort by P&L ascending, then invert the weighted step CDF.
    std::vector<std::size_t> order(pnl.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&pnl](std::size_t a, std::size_t b) { return pnl[a] < pnl[b]; });
    double cum = 0.0;
    for (std::size_t r = 0; r < order.size(); ++r) {
        cum += w[order[r]];
        if (cum >= alpha) return -pnl[order[r]];  // searchsorted-left semantics
    }
    return -pnl[order.back()];
}

std::vector<double> ewma_volatility(std::span<const double> x, double lam) {
    if (!(lam > 0.0 && lam < 1.0)) {
        throw std::invalid_argument("ewma_volatility: decay lam must be in (0, 1), got " +
                                    std::to_string(lam));
    }
    if (x.size() < 2) {
        throw std::invalid_argument("ewma_volatility: need at least 2 observations");
    }
    // Seed with the population (ddof = 0) sample variance, matching the
    // Python reference (np.var default).
    double mu = 0.0;
    for (double v : x) mu += v;
    mu /= static_cast<double>(x.size());
    double seed = 0.0;
    for (double v : x) seed += (v - mu) * (v - mu);
    seed /= static_cast<double>(x.size());
    if (seed <= 0.0) seed = 1e-16;  // zero-variance series: avoid division by 0

    std::vector<double> sigma(x.size());
    double sig2 = seed;
    sigma[0] = std::sqrt(std::max(sig2, 1e-32));
    for (std::size_t t = 1; t < x.size(); ++t) {
        sig2 = lam * sig2 + (1.0 - lam) * x[t - 1] * x[t - 1];
        sigma[t] = std::sqrt(std::max(sig2, 1e-32));
    }
    return sigma;
}

double filtered_historical_var(std::span<const double> pnl, double alpha, double lam) {
    validate_alpha(alpha);
    validate_pnl(pnl, kMinHistObs);
    const std::vector<double> sigma = ewma_volatility(pnl, lam);
    const double last = pnl[pnl.size() - 1];
    const double sig_last = sigma.back();
    const double sigma_next = std::sqrt(lam * sig_last * sig_last + (1.0 - lam) * last * last);
    std::vector<double> scenarios(pnl.size());
    for (std::size_t t = 0; t < pnl.size(); ++t) scenarios[t] = pnl[t] / sigma[t] * sigma_next;
    return -quantile_linear(scenarios, alpha);
}

double scale_var_sqrt_time(double var_1d, int horizon_days) {
    if (horizon_days < 1) {
        throw std::invalid_argument("scale_var_sqrt_time: horizon_days must be >= 1, got " +
                                    std::to_string(horizon_days));
    }
    return var_1d * std::sqrt(static_cast<double>(horizon_days));
}

}  // namespace eqvar
