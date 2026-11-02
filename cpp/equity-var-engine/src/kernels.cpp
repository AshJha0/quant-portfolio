#include "eqvar/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

#include "eqvar/returns.hpp"
#include "eqvar/stats.hpp"

namespace eqvar {

// ------------------------------------------------------------- messages

std::string message(const Error& e) {
    const std::string who = e.who;
    switch (e.code) {
        case Errc::BadAlpha:
            return "alpha is the tail probability (e.g. 0.01 for 99 %); must be in (0, 0.5), got " +
                   std::to_string(e.a);
        case Errc::EmptySample:
            return who + ": empty sample";
        case Errc::NonFiniteInput:
            return who + ": input contains NaN or inf";
        case Errc::TooFewPnl:
            return "need at least " + std::to_string(e.n1) + " P&L observations, got " +
                   std::to_string(e.n2) +
                   "; empirical tail quantiles are meaningless on shorter samples";
        case Errc::PnlNotFinite:
            return "pnl contains NaN or infinite values";
        case Errc::BadQuantile:
            return "quantile_linear: q must be in [0, 1], got " + std::to_string(e.a);
        case Errc::NoPositions:
            return who + ": book has no positions";
        case Errc::ExposuresNotFinite:
            return who + ": exposures must be finite";
        case Errc::MatvecShape:
            return "matvec: vector size " + std::to_string(e.n1) + " does not match " +
                   std::to_string(e.n2) + " columns";
        case Errc::QuadFormShape:
            return "quad_form: matrix shape does not match vector size " + std::to_string(e.n1);
        case Errc::SigmaShape:
            return "portfolio_sigma: covariance shape does not match " + std::to_string(e.n1) +
                   " exposures";
        case Errc::SigmaNotFinite:
            return "portfolio_sigma: w'Sw is not finite (covariance contains NaN/Inf "
                   "or overflows)";
        case Errc::NotPsd:
            return "portfolio_sigma: covariance matrix is not positive semi-definite (w'Sw < 0)";
        case Errc::MeanNotFinite:
            return who + ": mean must be finite";
        case Errc::BadHorizon:
            return "parametric_var: horizon_days must be >= 1, got " +
                   std::to_string(static_cast<int>(e.a));
        case Errc::BadDf:
            return "parametric_var: Student-t df must be > 2 for finite variance, got " +
                   std::to_string(e.a);
        case Errc::CovNotSquare:
            return who + ": covariance must be square and non-empty";
        case Errc::CovNotFinite:
            return who + ": covariance contains NaN or infinite entries";
        case Errc::CovNegativeDiag:
            return who + ": negative variance on the diagonal";
        case Errc::CovAsymmetric:
            return who + ": covariance matrix must be symmetric";
        case Errc::CholZeroVar:
            return "cholesky: zero variances with non-zero covariances";
        case Errc::CholIndefinite:
            return "cholesky: covariance is materially indefinite (jitter cap 1e-6 x "
                   "mean(diag) exceeded). Repair it (Repair::Clip or Repair::Higham) "
                   "before use.";
        case Errc::WorkspaceTooSmall:
            return who + ": workspace holds " + std::to_string(e.n1) + " doubles, needs " +
                   std::to_string(e.n2);
        case Errc::McShape:
            return "monte_carlo_var: covariance shape does not match exposures";
        case Errc::McNoPaths:
            return "simulate_factor_returns: n_paths must be >= 1";
        case Errc::McBadDf:
            return "simulate_factor_returns: Student-t df must be > 2 for finite variance, got " +
                   std::to_string(e.a);
        case Errc::McReturnsNotFinite:
            return "portfolio_pnl: returns contain NaN or inf";
        case Errc::McTooFewScenarios:
            return "mc_tail_metrics: need at least 100 scenarios, got " + std::to_string(e.n1);
        case Errc::McPnlNotFinite:
            return who + ": pnl contains NaN or inf";
        case Errc::BootTooFewObs:
            return who + ": need at least 10 observations, got " + std::to_string(e.n1);
        case Errc::BootTooFewReps:
            return "mc_bootstrap_se: n_boot must be >= 2";
    }
    return "eqvar: unknown error";
}

void throw_error(const Error& e) {
    if (e.code == Errc::CholZeroVar || e.code == Errc::CholIndefinite) {
        throw std::runtime_error(message(e));
    }
    throw std::invalid_argument(message(e));
}

namespace {

inline std::unexpected<Error> fail(Errc c, const char* who = "", double a = 0.0, std::size_t n1 = 0,
                                   std::size_t n2 = 0) noexcept {
    return std::unexpected(Error{c, who, a, n1, n2});
}

inline bool all_finite(std::span<const double> x) noexcept {
    for (double v : x) {
        if (!std::isfinite(v)) [[unlikely]]
            return false;
    }
    return true;
}

inline std::size_t tail_rank(std::size_t n, double alpha) noexcept {
    const double mc = std::ceil(static_cast<double>(n) * alpha - 1e-9);
    return mc < 1.0 ? 1 : std::min(n, static_cast<std::size_t>(mc));
}

/// The order-statistic estimator on a validated, finite sample.
TailRisk order_statistic(std::span<const double> pnl, double alpha, double* loss) noexcept {
    const std::size_t n = pnl.size();
    for (std::size_t i = 0; i < n; ++i) loss[i] = -pnl[i];
    const double k = static_cast<double>(n) * alpha;
    const double mc = std::ceil(k - 1e-9);
    const std::size_t m = mc < 1.0 ? 1 : std::min(n, static_cast<std::size_t>(mc));
    // Partition so loss[0..m-1) are the m-1 largest and loss[m-1] = L(m).
    std::nth_element(loss, loss + static_cast<std::ptrdiff_t>(m - 1), loss + n, std::greater<>());
    const double lm = loss[m - 1];
    double tail = 0.0;
    for (std::size_t i = 0; i + 1 < m; ++i) tail += loss[i];
    TailRisk r;
    r.var = lm;
    r.es = (tail + (k - static_cast<double>(m - 1)) * lm) / k;
    return r;
}

/// Plain Cholesky on raw row-major storage; false at a non-positive pivot.
using qp::cholesky_raw;

Result<double> bootstrap_se(std::span<const double> pnl, double alpha, std::size_t n_boot,
                            std::uint64_t seed, double* loss) noexcept {
    const std::size_t n = pnl.size();
    const std::size_t m = tail_rank(n, alpha);
    RandomStream rng(seed);
    double sum = 0.0, sum_sq_dev = 0.0;  // Welford
    for (std::size_t b = 0; b < n_boot; ++b) {
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t idx =
                std::min(n - 1, static_cast<std::size_t>(rng.uniform() * static_cast<double>(n)));
            loss[i] = -pnl[idx];
        }
        std::nth_element(loss, loss + static_cast<std::ptrdiff_t>(m - 1), loss + n,
                         std::greater<>());
        const double v = loss[m - 1];
        const double prev_mean = (b == 0) ? 0.0 : sum / static_cast<double>(b);
        sum += v;
        const double mean = sum / static_cast<double>(b + 1);
        sum_sq_dev += (v - prev_mean) * (v - mean);
    }
    return std::sqrt(std::max(sum_sq_dev, 0.0) / static_cast<double>(n_boot - 1));
}

double kde_se(std::span<const double> pnl, double alpha, double* loss, double* work) noexcept {
    const std::size_t n = pnl.size();
    const double q = -order_statistic(pnl, alpha, loss).var;  // reported quantile (P&L units)
    double mu = 0.0;
    for (double v : pnl) mu += v;
    mu /= static_cast<double>(n);
    double ss = 0.0;
    for (double v : pnl) ss += (v - mu) * (v - mu);
    const double sd = std::sqrt(ss / static_cast<double>(n - 1));
    std::copy(pnl.begin(), pnl.end(), work);
    const auto order_stat = [work, n](std::size_t i) {
        std::nth_element(work, work + static_cast<std::ptrdiff_t>(i), work + n);
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

}  // namespace

// ---------------------------------------------------------------- checks

Result<void> try_validate_alpha(double alpha) noexcept {
    if (!(alpha > 0.0 && alpha < 0.5)) [[unlikely]]
        return fail(Errc::BadAlpha, "", alpha);
    return {};
}

Result<void> try_validate_pnl(std::span<const double> pnl, std::size_t min_obs) noexcept {
    if (pnl.size() < min_obs) [[unlikely]] {
        return fail(Errc::TooFewPnl, "", 0.0, min_obs, pnl.size());
    }
    if (!all_finite(pnl)) [[unlikely]]
        return fail(Errc::PnlNotFinite);
    return {};
}

Result<void> try_validate_exposures(std::span<const double> exposures, const char* who) noexcept {
    if (exposures.empty()) [[unlikely]]
        return fail(Errc::NoPositions, who);
    bool any = false;
    for (double w : exposures) {
        if (!std::isfinite(w)) [[unlikely]]
            return fail(Errc::ExposuresNotFinite, who);
        any = any || w != 0.0;
    }
    if (!any) [[unlikely]]
        return fail(Errc::NoPositions, who);
    return {};
}

// ----------------------------------------------------------- historical

Result<TailRisk> try_var_es_order_statistic(std::span<const double> pnl, double alpha,
                                            std::span<double> workspace) noexcept {
    if (auto e = try_validate_alpha(alpha); !e) [[unlikely]]
        return std::unexpected(e.error());
    if (pnl.empty()) [[unlikely]]
        return fail(Errc::EmptySample, "var_es_order_statistic");
    if (!all_finite(pnl)) [[unlikely]]
        return fail(Errc::NonFiniteInput, "var_es_order_statistic");
    if (workspace.size() < pnl.size()) [[unlikely]] {
        return fail(Errc::WorkspaceTooSmall, "try_var_es_order_statistic", 0.0, workspace.size(),
                    pnl.size());
    }
    return order_statistic(pnl, alpha, workspace.data());
}

Result<double> try_historical_var(std::span<const double> pnl, double alpha,
                                  std::span<double> workspace) noexcept {
    if (auto e = try_validate_alpha(alpha); !e) [[unlikely]]
        return std::unexpected(e.error());
    if (auto e = try_validate_pnl(pnl, kMinHistObs); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    const auto r = try_var_es_order_statistic(pnl, alpha, workspace);
    if (!r) [[unlikely]]
        return std::unexpected(r.error());
    return r->var;
}

Result<double> try_expected_shortfall(std::span<const double> pnl, double alpha,
                                      std::span<double> workspace) noexcept {
    if (auto e = try_validate_alpha(alpha); !e) [[unlikely]]
        return std::unexpected(e.error());
    if (auto e = try_validate_pnl(pnl, 10); !e) [[unlikely]]
        return std::unexpected(e.error());
    const auto r = try_var_es_order_statistic(pnl, alpha, workspace);
    if (!r) [[unlikely]]
        return std::unexpected(r.error());
    return r->es;
}

Result<double> detail::try_quantile_linear(std::span<const double> x, double q,
                                           std::span<double> workspace) noexcept {
    if (x.empty()) [[unlikely]]
        return fail(Errc::EmptySample, "quantile_linear");
    if (!all_finite(x)) [[unlikely]]
        return fail(Errc::NonFiniteInput, "quantile_linear");
    if (!(q >= 0.0 && q <= 1.0)) [[unlikely]]
        return fail(Errc::BadQuantile, "", q);
    const std::size_t n = x.size();
    if (workspace.size() < n) [[unlikely]] {
        return fail(Errc::WorkspaceTooSmall, "try_quantile_linear", 0.0, workspace.size(), n);
    }
    double* w = workspace.data();
    std::copy(x.begin(), x.end(), w);
    const double h = q * static_cast<double>(n - 1);
    const auto lo = static_cast<std::size_t>(std::floor(h));
    if (lo + 1 >= n) return *std::max_element(w, w + n);
    std::nth_element(w, w + static_cast<std::ptrdiff_t>(lo), w + n);
    const double x_lo = w[lo];
    const double x_hi = *std::min_element(w + lo + 1, w + n);
    const double frac = h - static_cast<double>(lo);
    return x_lo + frac * (x_hi - x_lo);
}

// ---------------------------------------------------------------- matrix

Result<void> try_matvec(const Matrix& a, std::span<const double> x, std::span<double> y) noexcept {
    if (x.size() != a.cols()) [[unlikely]] {
        return fail(Errc::MatvecShape, "", 0.0, x.size(), a.cols());
    }
    if (!all_finite(x)) [[unlikely]]
        return fail(Errc::NonFiniteInput, "matvec");
    if (!all_finite({a.data(), a.size()})) [[unlikely]]
        return fail(Errc::NonFiniteInput, "matvec");
    const std::size_t rows = a.rows(), cols = a.cols();
    if (y.size() != rows) [[unlikely]] {
        return fail(Errc::WorkspaceTooSmall, "try_matvec", 0.0, y.size(), rows);
    }
    qp::matvec_rows(a.data(), rows, cols, x.data(), y.data());
    return {};
}

Result<double> try_quad_form(std::span<const double> w, const Matrix& s) noexcept {
    if (!s.square() || s.rows() != w.size()) [[unlikely]] {
        return fail(Errc::QuadFormShape, "", 0.0, w.size());
    }
    if (!all_finite(w)) [[unlikely]]
        return fail(Errc::NonFiniteInput, "quad_form");
    if (!all_finite({s.data(), s.size()})) [[unlikely]]
        return fail(Errc::NonFiniteInput, "quad_form");
    double acc = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        const double* row = s.data() + i * s.cols();
        double inner = 0.0;
        for (std::size_t j = 0; j < w.size(); ++j) inner += row[j] * w[j];
        acc += w[i] * inner;
    }
    return acc;
}

Result<double> try_cholesky_lower(const Matrix& cov, std::span<double> lower,
                                  std::span<double> scratch) noexcept {
    const char* who = "cholesky";
    if (!(cov.rows() == cov.cols() && cov.rows() > 0)) [[unlikely]] {
        return fail(Errc::CovNotSquare, who);
    }
    const std::size_t n = cov.rows();
    if (lower.size() < n * n || scratch.size() < n * n) [[unlikely]] {
        return fail(Errc::WorkspaceTooSmall, "try_cholesky_lower", 0.0,
                    std::min(lower.size(), scratch.size()), n * n);
    }
    const double* a = cov.data();
    double amax = 0.0;
    for (std::size_t k = 0; k < n * n; ++k) {
        if (!std::isfinite(a[k])) [[unlikely]]
            return fail(Errc::CovNotFinite, who);
        amax = std::max(amax, std::abs(a[k]));
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i * n + i] < 0.0) [[unlikely]]
            return fail(Errc::CovNegativeDiag, who);
        for (std::size_t j = i + 1; j < n; ++j) {
            if (std::abs(a[i * n + j] - a[j * n + i]) > 1e-10 * amax) [[unlikely]] {
                return fail(Errc::CovAsymmetric, who);
            }
        }
    }
    double scale = 0.0;
    for (std::size_t i = 0; i < n; ++i) scale += a[i * n + i];
    scale = scale / static_cast<double>(n);
    double* l = lower.data();
    if (scale == 0.0) {
        for (std::size_t k = 0; k < n * n; ++k) {
            if (a[k] != 0.0) [[unlikely]]
                return fail(Errc::CholZeroVar);
        }
        std::fill(l, l + n * n, 0.0);
        return 0.0;
    }
    if (cholesky_raw(a, n, l)) return 0.0;
    double* b = scratch.data();
    for (double eps = 1e-12 * scale; eps <= 1e-6 * scale * (1.0 + 1e-9); eps *= 10.0) {
        std::copy(a, a + n * n, b);
        for (std::size_t i = 0; i < n; ++i) b[i * n + i] += eps;
        if (cholesky_raw(b, n, l)) return eps;
    }
    return fail(Errc::CholIndefinite);
}

// ------------------------------------------------------------ parametric

Result<double> try_portfolio_sigma(std::span<const double> exposures, const Matrix& cov) noexcept {
    if (auto e = try_validate_exposures(exposures, "portfolio_sigma"); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    if (!cov.square() || cov.rows() != exposures.size()) [[unlikely]] {
        return fail(Errc::SigmaShape, "", 0.0, exposures.size());
    }
    const auto q = try_quad_form(exposures, cov);
    if (!q) [[unlikely]]
        return q;
    const double var = *q;
    if (!std::isfinite(var)) [[unlikely]]
        return fail(Errc::SigmaNotFinite);
    double wmax = 0.0;
    for (double w : exposures) wmax = std::max(wmax, std::abs(w));
    if (var < -1e-10 * std::max(1.0, wmax * wmax)) [[unlikely]]
        return fail(Errc::NotPsd);
    return std::sqrt(std::max(var, 0.0));
}

Result<double> try_parametric_var(std::span<const double> exposures, const Matrix& cov,
                                  double alpha, Dist dist, double df, double mean,
                                  int horizon_days) noexcept {
    if (auto e = try_validate_alpha(alpha); !e) [[unlikely]]
        return std::unexpected(e.error());
    if (!std::isfinite(mean)) [[unlikely]]
        return fail(Errc::MeanNotFinite, "parametric_var");
    if (horizon_days < 1) [[unlikely]] {
        return fail(Errc::BadHorizon, "", static_cast<double>(horizon_days));
    }
    const auto s = try_portfolio_sigma(exposures, cov);
    if (!s) [[unlikely]]
        return s;
    const double sigma = *s * std::sqrt(static_cast<double>(horizon_days));
    const double mu = mean * static_cast<double>(horizon_days);
    double z;
    if (dist == Dist::Normal) {
        z = normal_ppf(alpha);  // alpha in (0, 0.5): cannot throw
    } else {
        if (!(df > 2.0) || !std::isfinite(df)) [[unlikely]]
            return fail(Errc::BadDf, "", df);
        // Variance-matched t: quantile rescaled to unit variance.
        z = student_t_ppf(alpha, df) * std::sqrt((df - 2.0) / df);
    }
    return -(mu + z * sigma);
}

// ----------------------------------------------------------- Monte Carlo

Result<MonteCarloResult> try_mc_tail_metrics(std::span<const double> pnl, double alpha,
                                             std::size_t n_boot, std::uint64_t seed,
                                             McVarWorkspace& ws) noexcept {
    if (auto e = try_validate_alpha(alpha); !e) [[unlikely]]
        return std::unexpected(e.error());
    if (pnl.size() < 100) [[unlikely]]
        return fail(Errc::McTooFewScenarios, "", 0.0, pnl.size());
    if (!all_finite(pnl)) [[unlikely]]
        return fail(Errc::McPnlNotFinite, "mc_tail_metrics");
    if (ws.loss.size() < pnl.size() || ws.work.size() < pnl.size()) [[unlikely]] {
        return fail(Errc::WorkspaceTooSmall, "try_mc_tail_metrics", 0.0,
                    std::min(ws.loss.size(), ws.work.size()), pnl.size());
    }
    const TailRisk tr = order_statistic(pnl, alpha, ws.loss.data());
    MonteCarloResult res;
    res.n_paths = pnl.size();
    res.var = tr.var;
    res.es = tr.es;
    if (n_boot < 2) [[unlikely]]
        return fail(Errc::BootTooFewReps);
    res.var_se = *bootstrap_se(pnl, alpha, n_boot, seed, ws.loss.data());
    res.var_se_kde = kde_se(pnl, alpha, ws.loss.data(), ws.work.data());
    return res;
}

Result<MonteCarloResult> try_monte_carlo_var_factored(std::span<const double> exposures,
                                                      std::span<const double> lower, double alpha,
                                                      std::size_t n_paths, Dist dist, double df,
                                                      std::uint64_t seed, std::size_t n_boot,
                                                      McVarWorkspace& ws) noexcept {
    if (auto e = try_validate_exposures(exposures, "monte_carlo_var"); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    const std::size_t n = exposures.size();
    if (lower.size() != n * n) [[unlikely]]
        return fail(Errc::McShape);
    if (auto e = try_validate_alpha(alpha); !e) [[unlikely]]
        return std::unexpected(e.error());
    if (n_paths < 1) [[unlikely]]
        return fail(Errc::McNoPaths);
    const bool student = dist == Dist::StudentT;
    if (student && (!(df > 2.0) || !std::isfinite(df))) [[unlikely]] {
        return fail(Errc::McBadDf, "", df);
    }
    if (ws.z.size() < n || ws.row.size() < n || ws.pnl.size() < n_paths) [[unlikely]] {
        return fail(Errc::WorkspaceTooSmall, "try_monte_carlo_var_factored", 0.0, ws.pnl.size(),
                    n_paths);
    }
    // Same draw order as simulate_factor_returns(); each scenario row is
    // mapped to P&L at once, as portfolio_pnl() would, so the n_paths x n
    // panel is never stored.
    RandomStream rng(seed);
    double* z = ws.z.data();
    double* row = ws.row.data();
    double* pnl = ws.pnl.data();
    const double* l = lower.data();
    const double* w = exposures.data();
    bool finite = true;
    for (std::size_t p = 0; p < n_paths; ++p) {
        for (std::size_t j = 0; j < n; ++j) z[j] = rng.gaussian();
        for (std::size_t i = 0; i < n; ++i) {
            const double* li = l + i * n;
            double s = 0.0;
            for (std::size_t j = 0; j <= i; ++j) s += li[j] * z[j];
            row[i] = s;
        }
        if (student) {
            const double wmix = rng.chi_squared(df) / df;
            const double m = 1.0 / std::sqrt(wmix);
            for (std::size_t i = 0; i < n; ++i) row[i] *= m;
        }
        double acc = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            finite = finite && std::isfinite(row[j]);
            acc += row[j] * w[j];
        }
        pnl[p] = acc;
    }
    if (!finite) [[unlikely]]
        return fail(Errc::McReturnsNotFinite);
    return try_mc_tail_metrics({pnl, n_paths}, alpha, n_boot, seed ^ 0x9E3779B97F4A7C15ULL, ws);
}

// ---------------------------------------------- throwing Monte Carlo API

double mc_bootstrap_se(std::span<const double> pnl, double alpha, std::size_t n_boot,
                       std::uint64_t seed) {
    validate_alpha(alpha);
    if (pnl.size() < 10)
        throw_error(Error{Errc::BootTooFewObs, "mc_bootstrap_se", 0.0, pnl.size()});
    if (n_boot < 2) throw_error(Error{Errc::BootTooFewReps});
    if (!all_finite(pnl)) throw_error(Error{Errc::McPnlNotFinite, "mc_bootstrap_se"});
    // One resample buffer reused per replicate: memory O(n), never n_boot x n.
    std::vector<double> loss(pnl.size());
    return *bootstrap_se(pnl, alpha, n_boot, seed, loss.data());
}

double kde_var_se(std::span<const double> pnl, double alpha) {
    validate_alpha(alpha);
    if (pnl.size() < 10) throw_error(Error{Errc::BootTooFewObs, "kde_var_se", 0.0, pnl.size()});
    if (!all_finite(pnl)) throw_error(Error{Errc::McPnlNotFinite, "kde_var_se"});
    std::vector<double> loss(pnl.size()), work(pnl.size());
    return kde_se(pnl, alpha, loss.data(), work.data());
}

MonteCarloResult mc_tail_metrics(std::span<const double> pnl, double alpha, std::size_t n_boot,
                                 std::uint64_t seed) {
    McVarWorkspace ws;
    ws.reserve(0, pnl.size());
    return value_or_throw(try_mc_tail_metrics(pnl, alpha, n_boot, seed, ws));
}

MonteCarloResult monte_carlo_var(std::span<const double> exposures, const Matrix& cov, double alpha,
                                 std::size_t n_paths, Dist dist, double df, std::uint64_t seed,
                                 std::size_t n_boot, Repair repair) {
    validate_exposures(exposures, "monte_carlo_var");
    if (!cov.square() || cov.rows() != exposures.size()) throw_error(Error{Errc::McShape});
    validate_alpha(alpha);
    // Simulate once; VaR, ES and both SEs come from the same scenarios.
    // Validation order and factorisation are those of simulate_factor_returns().
    if (n_paths < 1) throw_error(Error{Errc::McNoPaths});
    const std::size_t n = cov.rows();
    Matrix scale = cov;
    if (dist == Dist::StudentT) {
        if (!(df > 2.0) || !std::isfinite(df)) throw_error(Error{Errc::McBadDf, "", df});
        for (std::size_t i = 0; i < n * n; ++i) scale.data()[i] *= (df - 2.0) / df;
    }
    const Matrix chol = cholesky(scale, repair).lower;
    McVarWorkspace ws;
    ws.reserve(n, n_paths);
    return value_or_throw(try_monte_carlo_var_factored(exposures, {chol.data(), chol.size()}, alpha,
                                                       n_paths, dist, df, seed, n_boot, ws));
}

}  // namespace eqvar
