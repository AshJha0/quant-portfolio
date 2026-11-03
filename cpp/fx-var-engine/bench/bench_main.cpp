// Per-call latency benchmark for the fxvar engine.
//
//   ./fxvar_bench [--out results.json] [--label after] [--quick]
//
// Book: 250 positions (200 spots, 50 forwards) on 45 synthetic currencies,
// 50 factors (45 FX + 5 IR). History: 1,000 and 10,000 days simulated from a
// full-rank AR(1) covariance (synthetic data). Every kernel is timed with
// the shared harness (see latency.hpp); inputs change between calls (a different history
// window, P&L vector or exposure scale), so no call repeats the previous
// one. Under WSL2 the hypervisor adds scheduling jitter to p99.9 and max.

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "fxvar/backtest.hpp"
#include "fxvar/book.hpp"
#include "fxvar/expected_shortfall.hpp"
#include "fxvar/historical.hpp"
#include "fxvar/matrix.hpp"
#include "fxvar/monte_carlo.hpp"
#include "fxvar/parametric.hpp"
#include "fxvar/stress.hpp"
#include "qp/bench/latency.hpp"

#if __has_include("fxvar/kernels.hpp")
#include "fxvar/kernels.hpp"
#define FXVAR_BENCH_NEW_API 1
#endif

using namespace fxvar;

namespace {

Matrix ar1(const std::vector<std::string>& factors) {
  const std::size_t k = factors.size();
  Matrix c(k, k);
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j) {
      const double si = factors[i].rfind("FX:", 0) == 0 ? 0.006 : 0.0004;
      const double sj = factors[j].rfind("FX:", 0) == 0 ? 0.006 : 0.0004;
      c(i, j) = si * sj * std::pow(0.5, std::abs(static_cast<double>(i) - static_cast<double>(j)));
    }
  return c;
}

/// Rows [first, first + n) of `r` as a new history (the varied input).
ReturnsMatrix window(const ReturnsMatrix& r, std::size_t first, std::size_t n) {
  ReturnsMatrix w;
  w.factors = r.factors;
  w.data = Matrix(n, r.n_factors());
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < r.n_factors(); ++j) w.data(i, j) = r.data(first + i, j);
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  qpbench::Bench b(argc, argv, "fx-var");

  std::vector<std::string> ccys;
  std::map<std::string, double> spots, rates{{"USD", 0.05}};
  for (int i = 0; i < 45; ++i) {
    std::string c = "C";
    c += static_cast<char>('A' + i / 26);
    c += static_cast<char>('A' + i % 26);
    ccys.push_back(c);
    spots[c] = 0.5 + 0.02 * i;
    rates[c] = 0.01 + 0.001 * (i % 10);
  }
  const Market market(spots, rates);
  std::vector<Position> positions;
  for (int i = 0; i < 200; ++i) {
    const std::string pair = ccys[static_cast<std::size_t>(i % 45)] + "USD";
    positions.push_back(SpotPosition{pair, (i % 2 ? -1.0 : 1.0) * (1e6 + 2e4 * i), {}});
  }
  for (int i = 0; i < 50; ++i) {
    const std::string pair = ccys[static_cast<std::size_t>(i % 4)] + "USD";
    positions.push_back(ForwardPosition{pair, (i % 2 ? -1.0 : 1.0) * 2e6, 0.25 + 0.05 * i, {}});
  }
  const Book book(positions);
  const CompiledBook compiled(book, market);
  FactorCov truth{compiled.factors(), ar1(compiled.factors())};
  const std::size_t k = truth.factors.size();
  const ReturnsMatrix hist = simulate_factor_returns(truth, 12000, McDist::kNormal, 6.0, {}, 7);
  const ReturnsMatrix h1k = window(hist, 0, 1000);
  const std::vector<double> pnl10k = compiled.pnl(window(hist, 0, 10000));
  const std::vector<double> pnl1k(pnl10k.begin(), pnl10k.begin() + 1000);
  const FactorCov cov = sample_cov(window(hist, 0, 1000));
  const std::vector<double> expo = compiled.linear_exposures();
  double acc = 0.0;

  // Rotate the P&L vector by i so each call sees a different input.
  std::vector<double> buf10k(pnl10k.size());
  auto rotated = [&](const std::vector<double>& v, std::size_t i) -> const std::vector<double>& {
    const std::size_t n = v.size(), s = (i * 7919) % n;
    buf10k.resize(n);
    for (std::size_t j = 0; j < n; ++j) buf10k[j] = v[(j + s) % n];
    return buf10k;
  };

  HistoricalOptions hopt;
  hopt.warn_pegs = false;
  b.run(
      "historical_var 1k obs 250 pos",
      [&](std::size_t i) {
        const ReturnsMatrix w = window(hist, i % 2000, 1000);
        acc += historical_var(book, market, w, hopt).var;
      },
      10, 200);

  b.run(
      "empirical_var_es n=1000",
      [&](std::size_t i) { acc += empirical_var_es(rotated(pnl1k, i), 0.01).second; }, 100, 20000);
  b.run(
      "empirical_var_es n=10000",
      [&](std::size_t i) { acc += empirical_var_es(rotated(pnl10k, i), 0.01).second; }, 50, 5000);
  b.run(
      "var_standard_error n=10000",
      [&](std::size_t i) { acc += var_standard_error(rotated(pnl10k, i), 0.01); }, 20, 2000);

  ParametricOptions popt;
  popt.warn_pegs = false;
  b.run(
      "parametric_var 1k obs 50 factors",
      [&](std::size_t i) {
        const ReturnsMatrix w = window(hist, i % 2000, 1000);
        acc += parametric_var(book, market, w, popt).var;
      },
      10, 300);
  std::vector<double> e2(expo);
  b.run(
      "var_covar 50 factors",
      [&](std::size_t i) {
        e2[i % k] = expo[i % k] * (1.0 + 1e-6 * static_cast<double>(i % 97));
        acc += var_covar(e2, cov.cov, 0.01, 10.0).es;
      },
      1000, 100000);

  Matrix c2 = cov.cov;
  b.run(
      "cholesky 50x50",
      [&](std::size_t i) {
        c2(0, 0) = cov.cov(0, 0) * (1.0 + 1e-9 * static_cast<double>(i % 101));
        acc += cholesky(c2).lower(k - 1, k - 1);
      },
      100, 20000);
  b.run(
      "matvec 50x50",
      [&](std::size_t i) {
        e2[i % k] = expo[i % k] + static_cast<double>(i % 13);
        acc += matvec(cov.cov, e2)[0];
      },
      1000, 200000);

  MonteCarloOptions mopt;
  mopt.n_scenarios = 10000;
  mopt.n_boot = 50;
  b.run(
      "monte_carlo_var 10k scen normal",
      [&](std::size_t i) {
        mopt.seed = i;
        acc += monte_carlo_var(book, market, cov, mopt).var;
      },
      3, 60);
  b.run(
      "compiled pnl 10k scen 250 pos",
      [&](std::size_t i) {
        const ReturnsMatrix w = window(hist, i % 2000, 10000);
        acc += compiled.pnl(w)[5];
      },
      3, 60);

  const auto scen = historical_scenarios();
  b.run(
      "run_stress library", [&](std::size_t) { acc += run_stress(book, market, scen).front().pnl; },
      50, 2000);

  std::vector<double> var_fc(1000);
  for (std::size_t i = 0; i < var_fc.size(); ++i)
    var_fc[i] = 2.3 * 1e6 * (1.0 + 0.1 * std::sin(0.1 * static_cast<double>(i)));
  b.run(
      "evaluate_var_backtest 1000 days",
      [&](std::size_t i) {
        acc += evaluate_var_backtest(rotated(pnl1k, i), var_fc, 0.01).kupiec.p;
      },
      50, 5000);

#ifdef FXVAR_BENCH_NEW_API
  std::vector<double> ws(pnl10k.size());
  b.run(
      "try_empirical_var_es n=1000 (ws)",
      [&](std::size_t i) { acc += try_empirical_var_es(rotated(pnl1k, i), 0.01, ws)->second; }, 100,
      20000);
  b.run(
      "try_empirical_var_es n=10000 (ws)",
      [&](std::size_t i) { acc += try_empirical_var_es(rotated(pnl10k, i), 0.01, ws)->second; }, 50,
      5000);
  b.run(
      "try_var_standard_error n=10000",
      [&](std::size_t i) { acc += *try_var_standard_error(rotated(pnl10k, i), 0.01, ws); }, 20,
      2000);
  b.run(
      "try_var_covar 50 factors",
      [&](std::size_t i) {
        e2[i % k] = expo[i % k] * (1.0 + 1e-6 * static_cast<double>(i % 97));
        acc += try_var_covar(e2, cov.cov, 0.01, 10.0)->es;
      },
      1000, 100000);
  std::vector<double> lower(k * k), y(k);
  b.run(
      "try_cholesky_lower 50x50",
      [&](std::size_t i) {
        c2(0, 0) = cov.cov(0, 0) * (1.0 + 1e-9 * static_cast<double>(i % 101));
        acc += try_cholesky_lower(c2, lower).has_value() ? lower[k * k - 1] : 0.0;
      },
      100, 20000);
  b.run(
      "try_matvec 50x50",
      [&](std::size_t i) {
        e2[i % k] = expo[i % k] + static_cast<double>(i % 13);
        acc += try_matvec(cov.cov, e2, y).has_value() ? y[0] : 0.0;
      },
      1000, 200000);
  std::vector<double> sims(10000 * k), out(10000);
  (void)try_cholesky_lower(cov.cov, lower);
  b.run(
      "try_simulate 10k scen 50 factors",
      [&](std::size_t i) {
        acc +=
            try_simulate_factor_returns(lower, k, 10000, McDist::kNormal, 6.0, i, sims).has_value()
                ? sims[7]
                : 0.0;
      },
      3, 60);
  b.run(
      "try_pnl_batch 10k scen 250 pos",
      [&](std::size_t i) {
        const std::size_t off = (i % 2000) * k;
        acc += compiled.try_pnl_batch(
                           std::span<const double>(hist.data.data()).subspan(off, 10000 * k), out)
                       .has_value()
                   ? out[5]
                   : 0.0;
      },
      3, 60);
  b.run(
      "try_pnl 1 scen 250 pos",
      [&](std::size_t i) {
        acc += *compiled.try_pnl(std::span<const double>(hist.data.row(i % 12000), k));
      },
      1000, 200000);
  b.run(
      "MC VaR pipeline 10k (try_*, ws)",
      [&](std::size_t i) {
        (void)try_simulate_factor_returns(lower, k, 10000, McDist::kNormal, 6.0, i, sims);
        (void)compiled.try_pnl_batch(sims, out);
        acc += try_empirical_var_es(out, 0.01, ws)->first;
      },
      3, 60);
#endif

  qpbench::g_sink = acc;
  return 0;
}
