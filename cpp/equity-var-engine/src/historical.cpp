#include "eqvar/historical.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>

#include "eqvar/kernels.hpp"
#include "eqvar/returns.hpp"

namespace eqvar {

namespace {

void require_finite(std::span<const double> x, const char* who) {
    for (double v : x) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument(std::string(who) + ": input contains NaN or inf");
        }
    }
}

void validate_lam(double lam, const char* who) {
    if (!(lam > 0.0 && lam < 1.0)) {
        throw std::invalid_argument(std::string(who) + ": decay lam must be in (0, 1), got " +
                                    std::to_string(lam));
    }
}

// EWMA variance path. Returns the one-step-ahead variance s2_n and fills
// sigma (length n) when given.
double ewma_path(std::span<const double> x, double lam, std::size_t seed_window,
                 std::vector<double>* sigma) {
    validate_lam(lam, "ewma_volatility");
    if (x.size() < 2) throw std::invalid_argument("ewma_volatility: need at least 2 observations");
    if (seed_window < 1) throw std::invalid_argument("ewma_volatility: seed_window must be >= 1");
    require_finite(x, "ewma_volatility");
    const std::size_t w = std::min(seed_window, x.size());
    double s2 = 0.0;
    for (std::size_t i = 0; i < w; ++i) s2 += x[i] * x[i];
    s2 /= static_cast<double>(w);
    if (sigma) sigma->resize(x.size());
    for (std::size_t t = 0; t < x.size(); ++t) {
        if (t > 0) s2 = lam * s2 + (1.0 - lam) * x[t - 1] * x[t - 1];
        if (sigma) (*sigma)[t] = std::sqrt(std::max(s2, 1e-32));
    }
    const double last = x[x.size() - 1];
    return std::max(lam * s2 + (1.0 - lam) * last * last, 1e-32);
}

}  // namespace

TailRisk var_es_order_statistic(std::span<const double> pnl, double alpha) {
    std::vector<double> loss(pnl.size());
    return value_or_throw(try_var_es_order_statistic(pnl, alpha, loss));
}

TailRisk var_es_weighted(std::span<const double> pnl, std::span<const double> weights,
                         double alpha) {
    validate_alpha(alpha);
    if (pnl.empty()) throw std::invalid_argument("var_es_weighted: empty sample");
    if (weights.size() != pnl.size()) {
        throw std::invalid_argument("var_es_weighted: weights must match pnl length");
    }
    require_finite(pnl, "var_es_weighted");
    double wsum = 0.0;
    for (double w : weights) {
        if (!std::isfinite(w) || w < 0.0) {
            throw std::invalid_argument("var_es_weighted: weights must be finite and >= 0");
        }
        wsum += w;
    }
    if (!(wsum > 0.0) || !std::isfinite(wsum)) {
        throw std::invalid_argument("var_es_weighted: weights must sum to a finite value > 0");
    }
    const std::size_t n = pnl.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&pnl](std::size_t a, std::size_t b) { return -pnl[a] > -pnl[b]; });
    double cum = 0.0, tail = 0.0;
    std::size_t m = n - 1;
    for (std::size_t j = 0; j < n; ++j) {
        const double wj = weights[order[j]] / wsum;
        if (cum + wj >= alpha - 1e-12 || j + 1 == n) {
            m = j;
            break;
        }
        cum += wj;
        tail += wj * -pnl[order[j]];
    }
    const double lm = -pnl[order[m]];
    TailRisk r;
    r.var = lm;
    r.es = (tail + std::max(alpha - cum, 0.0) * lm) / alpha;
    return r;
}

double historical_var(std::span<const double> pnl, double alpha) {
    std::vector<double> loss(pnl.size());
    return value_or_throw(try_historical_var(pnl, alpha, loss));
}

double interpolated_var(std::span<const double> pnl, double alpha) {
    validate_alpha(alpha);
    if (pnl.empty()) throw std::invalid_argument("interpolated_var: empty sample");
    require_finite(pnl, "interpolated_var");
    const std::size_t n = pnl.size();
    std::vector<double> loss(n);
    for (std::size_t i = 0; i < n; ++i) loss[i] = -pnl[i];
    const double h = static_cast<double>(n - 1) * alpha;
    const auto lo = static_cast<std::size_t>(std::floor(h));
    const std::size_t hi = std::min(lo + 1, n - 1);
    // Worst first: the hi+1 largest losses, sorted descending.
    std::partial_sort(loss.begin(), loss.begin() + static_cast<std::ptrdiff_t>(hi + 1), loss.end(),
                      std::greater<>());
    return loss[lo] + (h - static_cast<double>(lo)) * (loss[hi] - loss[lo]);
}

double historical_var(std::span<const double> pnl, double alpha, HistRule rule) {
    if (rule == HistRule::kKthWorst) return historical_var(pnl, alpha);
    validate_alpha(alpha);
    validate_pnl(pnl, kMinHistObs);
    return interpolated_var(pnl, alpha);
}

std::vector<double> brw_weights(std::size_t n, double lam) {
    validate_lam(lam, "brw_weights");
    if (n < 1) throw std::invalid_argument("brw_weights: n must be >= 1");
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
    return var_es_weighted(pnl, w, alpha).var;
}

double age_weighted_es(std::span<const double> pnl, double alpha, double lam) {
    validate_alpha(alpha);
    validate_pnl(pnl, kMinHistObs);
    const std::vector<double> w = brw_weights(pnl.size(), lam);
    return var_es_weighted(pnl, w, alpha).es;
}

std::vector<double> ewma_volatility(std::span<const double> x, double lam,
                                    std::size_t seed_window) {
    std::vector<double> sigma;
    (void)ewma_path(x, lam, seed_window, &sigma);
    return sigma;
}

double ewma_volatility_next(std::span<const double> x, double lam, std::size_t seed_window) {
    return std::sqrt(ewma_path(x, lam, seed_window, nullptr));
}

TailRisk filtered_historical_var_es(const Matrix& factor_returns, std::span<const double> exposures,
                                    double alpha, double lam) {
    validate_alpha(alpha);
    validate_lam(lam, "filtered_historical_var_es");
    validate_exposures(exposures, "filtered_historical_var_es");
    const std::size_t t_obs = factor_returns.rows(), n_f = factor_returns.cols();
    if (n_f != exposures.size()) {
        throw std::invalid_argument("filtered_historical_var_es: returns have " +
                                    std::to_string(n_f) + " factors, book has " +
                                    std::to_string(exposures.size()) + " exposures");
    }
    if (t_obs < kMinHistObs) {
        throw std::invalid_argument("filtered_historical_var_es: need at least " +
                                    std::to_string(kMinHistObs) + " observations");
    }
    std::vector<double> pnl(t_obs, 0.0);
    std::vector<double> col(t_obs);
    std::vector<double> sigma;
    for (std::size_t j = 0; j < n_f; ++j) {
        for (std::size_t t = 0; t < t_obs; ++t) col[t] = factor_returns(t, j);
        const double next = std::sqrt(ewma_path(col, lam, kEwmaSeedWindow, &sigma));
        for (std::size_t t = 0; t < t_obs; ++t) {
            pnl[t] += exposures[j] * (col[t] / sigma[t] * next);
        }
    }
    return var_es_order_statistic(pnl, alpha);
}

double filtered_historical_var(std::span<const double> pnl, double alpha, double lam) {
    // Deprecated: kept for one release with its full-sample population-variance
    // seed. The tail is the same order statistic as every other VaR path
    // (m = ceil(n alpha)), matching the Python reference.
    validate_alpha(alpha);
    validate_lam(lam, "filtered_historical_var");
    validate_pnl(pnl, kMinHistObs);
    const std::size_t n = pnl.size();
    double mu = 0.0;
    for (double v : pnl) mu += v;
    mu /= static_cast<double>(n);
    double s2 = 0.0;
    for (double v : pnl) s2 += (v - mu) * (v - mu);
    s2 /= static_cast<double>(n);
    std::vector<double> sigma(n);
    for (std::size_t t = 0; t < n; ++t) {
        if (t > 0) s2 = lam * s2 + (1.0 - lam) * pnl[t - 1] * pnl[t - 1];
        sigma[t] = std::sqrt(std::max(s2, 1e-32));
    }
    const double last = pnl[n - 1];
    const double next = std::sqrt(lam * sigma[n - 1] * sigma[n - 1] + (1.0 - lam) * last * last);
    std::vector<double> scenarios(n);
    for (std::size_t t = 0; t < n; ++t) scenarios[t] = pnl[t] / sigma[t] * next;
    return var_es_order_statistic(scenarios, alpha).var;
}

double scale_var_sqrt_time(double var_1d, int horizon_days) {
    if (!std::isfinite(var_1d)) {
        throw std::invalid_argument("scale_var_sqrt_time: var_1d must be finite");
    }
    if (horizon_days < 1) {
        throw std::invalid_argument("scale_var_sqrt_time: horizon_days must be >= 1, got " +
                                    std::to_string(horizon_days));
    }
    return var_1d * std::sqrt(static_cast<double>(horizon_days));
}

}  // namespace eqvar
