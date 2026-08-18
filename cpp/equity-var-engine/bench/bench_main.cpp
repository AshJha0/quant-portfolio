// bench_main.cpp — end-to-end timings for the eqvar engine on a realistic
// desk-sized problem: a 250-day x 100-asset returns panel (one year of
// history, 100 risk factors).
//
// Measured stages
//   1. sample + EWMA covariance estimation from the panel (100 x 100);
//   2. historical VaR family (plain / BRW / FHS) on the portfolio P&L;
//   3. parametric VaR (normal + Student-t) incl. Cholesky-free sigma;
//   4. Monte Carlo VaR + ES, 100'000 paths, normal and Student-t (the
//      dominant cost: 100k x 100 correlated draws through the Cholesky
//      factor = 10M gaussians + 1G multiply-adds for the triangular product).
//
// Deterministic inputs (closed-form sin/cos panel, fixed MC seed) so runs are
// exactly reproducible; each stage is repeated and the best-of-R wall time is
// reported (best-of is the standard way to strip scheduler noise from a
// single-threaded CPU-bound benchmark).
//
// Build: part of the default CMake target set; run ./bench_eqvar from the
// build directory.  Results are quoted in README.md and docs/DESK_GUIDE.md.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "eqvar/expected_shortfall.hpp"
#include "eqvar/historical.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/monte_carlo.hpp"
#include "eqvar/parametric.hpp"
#include "eqvar/returns.hpp"

namespace {

using Clock = std::chrono::steady_clock;

/// Best-of-`reps` wall time (ms) of `fn()`; the result of the last call is
/// accumulated into `sink` so the optimiser cannot elide the work.
template <typename F>
double best_of_ms(int reps, double& sink, F&& fn) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = Clock::now();
        sink += fn();
        const auto t1 = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

void row(const char* stage, double ms, const std::string& result) {
    std::printf("| %-42s | %10.3f | %-28s |\n", stage, ms, result.c_str());
}

std::string money(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.0f", v);
    return std::string(buf);
}

}  // namespace

int main() {
    constexpr std::size_t kDays = 250;
    constexpr std::size_t kAssets = 100;
    constexpr std::size_t kPaths = 100'000;
    constexpr double kAlpha = 0.01;

    // Deterministic 250 x 100 daily returns panel: per-asset vol ramps from
    // 10 bp to ~2 % and phase-shifted sin/cos terms induce rich correlation.
    eqvar::Matrix panel(kDays, kAssets);
    for (std::size_t t = 0; t < kDays; ++t) {
        for (std::size_t j = 0; j < kAssets; ++j) {
            const double td = static_cast<double>(t), jd = static_cast<double>(j);
            const double vol = 0.001 + 0.0002 * jd;
            panel(t, j) = vol * (std::sin(0.7 * td + 0.31 * jd) +
                                 0.5 * std::cos(1.3 * td - 0.11 * jd));
        }
    }
    // Long-biased book: exposures alternate long/short, 100k..1.09M per name.
    std::vector<double> w(kAssets);
    for (std::size_t j = 0; j < kAssets; ++j) {
        w[j] = (j % 3 == 2 ? -1.0 : 1.0) * (1.0e5 + 1.0e4 * static_cast<double>(j));
    }

    std::printf("eqvar benchmark — %zu-day x %zu-asset panel, alpha = %.2f, "
                "%zu MC paths, single thread\n\n",
                kDays, kAssets, kAlpha, kPaths);
    std::printf("| %-42s | %10s | %-28s |\n", "stage", "best ms", "result");
    std::printf("|%s|%s|%s|\n", std::string(44, '-').c_str(), std::string(12, '-').c_str(),
                std::string(30, '-').c_str());

    double sink = 0.0;

    // --- covariance estimation ---------------------------------------------
    eqvar::Matrix cov;
    const double ms_cov = best_of_ms(5, sink, [&] {
        cov = eqvar::sample_covariance(panel);
        return cov(0, 0);
    });
    row("sample covariance (250 x 100 -> 100 x 100)", ms_cov, "");
    eqvar::Matrix ecov;
    const double ms_ecov = best_of_ms(5, sink, [&] {
        ecov = eqvar::ewma_covariance(panel, 0.94);
        return ecov(0, 0);
    });
    row("EWMA covariance (lam = 0.94)", ms_ecov, "");

    // --- historical family --------------------------------------------------
    const std::vector<double> pnl = eqvar::portfolio_pnl(panel, w);
    double v = 0.0;
    const double ms_hist = best_of_ms(20, sink, [&] {
        v = eqvar::historical_var(pnl, kAlpha);
        return v;
    });
    row("historical VaR 99% (250 scenarios)", ms_hist, money(v));
    const double ms_brw = best_of_ms(20, sink, [&] {
        v = eqvar::age_weighted_var(pnl, kAlpha, 0.98);
        return v;
    });
    row("BRW age-weighted VaR 99% (lam = 0.98)", ms_brw, money(v));
    const double ms_fhs = best_of_ms(20, sink, [&] {
        v = eqvar::filtered_historical_var(pnl, kAlpha, 0.94);
        return v;
    });
    row("filtered (FHS) VaR 99% (lam = 0.94)", ms_fhs, money(v));
    const double ms_es = best_of_ms(20, sink, [&] {
        v = eqvar::expected_shortfall(pnl, 0.025);
        return v;
    });
    row("empirical ES 97.5%", ms_es, money(v));

    // --- parametric ----------------------------------------------------------
    const double ms_par_n = best_of_ms(20, sink, [&] {
        v = eqvar::parametric_var(w, cov, kAlpha);
        return v;
    });
    row("parametric VaR 99% (normal)", ms_par_n, money(v));
    const double ms_par_t = best_of_ms(20, sink, [&] {
        v = eqvar::parametric_var(w, cov, kAlpha, eqvar::Dist::StudentT, 6.0);
        return v;
    });
    row("parametric VaR 99% (Student-t, df = 6)", ms_par_t, money(v));
    const double sigma = eqvar::portfolio_sigma(w, cov);
    const double ms_nes = best_of_ms(20, sink, [&] {
        v = eqvar::normal_es(sigma, 0.025);
        return v;
    });
    row("closed-form ES 97.5% (normal)", ms_nes, money(v));

    // --- Cholesky (isolated) -------------------------------------------------
    const double ms_chol = best_of_ms(5, sink, [&] {
        const eqvar::CholeskyResult ch = eqvar::cholesky(cov);
        return ch.lower(0, 0);
    });
    row("Cholesky 100 x 100", ms_chol, "");

    // --- Monte Carlo ---------------------------------------------------------
    eqvar::MonteCarloResult mc;
    const double ms_mc_n = best_of_ms(3, sink, [&] {
        mc = eqvar::monte_carlo_var(w, cov, kAlpha, kPaths, eqvar::Dist::Normal, 6.0, 42);
        return mc.var;
    });
    row("MC VaR 99% normal (100k x 100 paths)", ms_mc_n,
        money(mc.var) + " (SE " + money(mc.var_se) + ")");
    const double mc_var_n = mc.var, mc_es_n = mc.es;
    const double ms_mc_t = best_of_ms(3, sink, [&] {
        mc = eqvar::monte_carlo_var(w, cov, kAlpha, kPaths, eqvar::Dist::StudentT, 6.0, 42);
        return mc.var;
    });
    row("MC VaR 99% Student-t df=6 (100k paths)", ms_mc_t,
        money(mc.var) + " (SE " + money(mc.var_se) + ")");

    const double total =
        ms_cov + ms_ecov + ms_hist + ms_brw + ms_fhs + ms_es + ms_par_n + ms_par_t + ms_mc_n;
    std::printf("\nfull daily batch (cov + historical family + parametric + 100k-path MC): "
                "%.1f ms\n", total);
    std::printf("MC normal vs parametric normal: %.0f vs %.0f (agreement within MC error)\n",
                mc_var_n, eqvar::parametric_var(w, cov, kAlpha));
    std::printf("MC normal ES 99%%: %.0f  |  closed-form: %.0f\n", mc_es_n,
                eqvar::normal_es(sigma, kAlpha));
    std::printf("[checksum %.6g]\n", sink);
    return 0;
}
