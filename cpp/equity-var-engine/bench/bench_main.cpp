// Per-call latency benchmark for the eqvar engine.
//
//   ./bench_eqvar [--out results.json] [--label after] [--quick]
//
// Every kernel is timed call by call (see latency.hpp). Inputs vary per
// call: P&L samples rotate through a table of synthetic series and alpha
// steps through 0.5 % to 5 %, so selection and branch predictors see a
// realistic mix. Sizes: 1,000 and 10,000-observation P&L, a 50-asset book,
// 50 x 50 and 200 x 200 covariance matrices, 10,000 Monte Carlo scenarios.
// Numbers depend on the machine; under WSL2 the hypervisor adds scheduling
// jitter that shows up in p99.9 and max. All data is synthetic.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "eqvar/backtest.hpp"
#include "eqvar/expected_shortfall.hpp"
#include "eqvar/historical.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/monte_carlo.hpp"
#include "eqvar/parametric.hpp"
#include "eqvar/returns.hpp"
#include "latency.hpp"

#if __has_include("eqvar/kernels.hpp")
#include "eqvar/kernels.hpp"
#define EQVAR_BENCH_NEW_API 1
#endif

namespace {

using namespace eqvar;

constexpr std::size_t kSeries = 16;

std::vector<std::vector<double>> pnl_table(std::size_t n, std::uint64_t seed) {
    qpbench::InputGen g(seed);
    std::vector<std::vector<double>> t(kSeries, std::vector<double>(n));
    for (auto& s : t) {
        for (auto& v : s) {
            // Fat-ish tails: a normal-ish body plus occasional large moves.
            const double body = g.uniform(-1.0, 1.0) + g.uniform(-1.0, 1.0) + g.uniform(-1.0, 1.0);
            v = 1e4 * body * (g.uniform(0.0, 1.0) < 0.03 ? 4.0 : 1.0);
        }
    }
    return t;
}

Matrix make_cov(std::size_t n, double rho) {
    Matrix c(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double vi = 0.01 + 0.0002 * static_cast<double>(i);
            const double vj = 0.01 + 0.0002 * static_cast<double>(j);
            c(i, j) = vi * vj * (i == j ? 1.0 : rho);
        }
    return c;
}

double alpha_of(std::size_t i) { return 0.005 + 0.005 * static_cast<double>(i % 10); }

}  // namespace

int main(int argc, char** argv) {
    qpbench::Bench b(argc, argv, "equity-var");
    double acc = 0.0;
    const auto p1k = pnl_table(1000, 1);
    const auto p10k = pnl_table(10000, 2);
    const std::size_t n_assets = 50;
    const Matrix cov50 = make_cov(50, 0.35);
    const Matrix cov200 = make_cov(200, 0.35);
    std::vector<std::vector<double>> books(kSeries, std::vector<double>(n_assets));
    {
        qpbench::InputGen g(3);
        for (auto& w : books)
            for (auto& v : w) v = g.uniform(-2e6, 2e6);
    }
    Matrix panel(10000, n_assets);
    {
        qpbench::InputGen g(4);
        for (std::size_t k = 0; k < panel.size(); ++k) panel.data()[k] = g.uniform(-0.03, 0.03);
    }

    b.run("historical_var n=1000", [&](std::size_t i) {
        acc += historical_var(p1k[i % kSeries], alpha_of(i));
    }, 2000, 100000);
    b.run("historical_var n=10000", [&](std::size_t i) {
        acc += historical_var(p10k[i % kSeries], alpha_of(i));
    }, 200, 20000);
    b.run("expected_shortfall n=10000", [&](std::size_t i) {
        acc += expected_shortfall(p10k[i % kSeries], alpha_of(i));
    }, 200, 20000);
    b.run("age_weighted_var n=1000", [&](std::size_t i) {
        acc += age_weighted_var(p1k[i % kSeries], alpha_of(i), 0.98);
    }, 200, 20000);
    b.run("parametric_var normal 50 assets", [&](std::size_t i) {
        acc += parametric_var(books[i % kSeries], cov50, alpha_of(i));
    }, 2000, 200000);
    b.run("parametric_var student-t 50 assets", [&](std::size_t i) {
        acc += parametric_var(books[i % kSeries], cov50, alpha_of(i), Dist::StudentT, 5.0);
    }, 500, 50000);
    b.run("cholesky 50x50", [&](std::size_t) {
        acc += cholesky(cov50).lower(49, 49);
    }, 200, 20000);
    b.run("cholesky 200x200", [&](std::size_t) {
        acc += cholesky(cov200).lower(199, 199);
    }, 20, 2000);
    b.run("portfolio_pnl 10000x50", [&](std::size_t i) {
        acc += portfolio_pnl(panel, books[i % kSeries])[i % 10000];
    }, 50, 5000);
    b.run("monte_carlo_var 10k scen 50 assets", [&](std::size_t i) {
        acc += monte_carlo_var(books[i % kSeries], cov50, 0.01, 10000, Dist::Normal, 6.0, i, 50)
                   .var;
    }, 3, 100);
    {
        std::vector<std::uint8_t> exc(1000);
        b.run("kupiec + christoffersen n=1000", [&](std::size_t i) {
            for (std::size_t t = 0; t < exc.size(); ++t) {
                exc[t] = static_cast<std::uint8_t>(((t * 2654435761U + i) % 97) < 2);
            }
            const auto cc = christoffersen_cc(exc, 0.01);
            const auto k = kupiec_pof(KupiecInput{1000, static_cast<int>(i % 30), 0.01});
            acc += cc.lr + k.lr;
        }, 1000, 100000);
    }

#ifdef EQVAR_BENCH_NEW_API
    std::vector<double> ws(10000);
    b.run("try_historical_var n=1000 (ws)", [&](std::size_t i) {
        acc += try_historical_var(p1k[i % kSeries], alpha_of(i), ws).value_or(0.0);
    }, 2000, 100000);
    b.run("try_historical_var n=10000 (ws)", [&](std::size_t i) {
        acc += try_historical_var(p10k[i % kSeries], alpha_of(i), ws).value_or(0.0);
    }, 200, 20000);
    b.run("try_parametric_var normal 50 assets", [&](std::size_t i) {
        acc += try_parametric_var(books[i % kSeries], cov50, alpha_of(i)).value_or(0.0);
    }, 2000, 200000);
    std::vector<double> l50(50 * 50), s50(50 * 50), l200(200 * 200), s200(200 * 200);
    b.run("try_cholesky_lower 50x50 (ws)", [&](std::size_t) {
        acc += try_cholesky_lower(cov50, l50, s50).value_or(0.0) + l50.back();
    }, 200, 20000);
    b.run("try_cholesky_lower 200x200 (ws)", [&](std::size_t) {
        acc += try_cholesky_lower(cov200, l200, s200).value_or(0.0) + l200.back();
    }, 20, 2000);
    std::vector<double> y(10000);
    b.run("try_matvec scenario pnl 10000x50", [&](std::size_t i) {
        (void)try_matvec(panel, books[i % kSeries], y);
        acc += y[i % 10000];
    }, 50, 5000);
    McVarWorkspace mws;
    mws.reserve(n_assets, 10000);
    (void)try_cholesky_lower(cov50, l50, s50);
    b.run("try_monte_carlo_var_factored 10k 50", [&](std::size_t i) {
        const auto r = try_monte_carlo_var_factored(books[i % kSeries], l50, 0.01, 10000,
                                                    Dist::Normal, 6.0, i, 50, mws);
        acc += r ? r->var : 0.0;
    }, 3, 100);
#endif

    qpbench::g_sink = acc;
    return 0;
}
