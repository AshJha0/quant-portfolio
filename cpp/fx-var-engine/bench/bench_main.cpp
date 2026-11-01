// Benchmark: 250-position, 50-factor FX book - historical, parametric
// and 100k-scenario Monte Carlo VaR wall times.
//
// The book is deterministic: 45 non-USD currencies (45 FX factors),
// forwards on 4 currency pairs adding 5 IR factors (incl. IR:USD) for a
// 50-factor set; 200 spot positions + 50 forwards = 250 positions.
// History is 500 days simulated from a full-rank AR(1) covariance.
//
// Build:  cmake --build build && ./build/fxvar_bench
// Results are quoted in README.md and docs/VALIDATION.md.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "fxvar/book.hpp"
#include "fxvar/historical.hpp"
#include "fxvar/monte_carlo.hpp"
#include "fxvar/parametric.hpp"

using namespace fxvar;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

int main() {
  // ---- deterministic 45-currency universe --------------------------------
  std::vector<std::string> ccys;
  for (int i = 0; i < 45; ++i) {
    std::string c = "C";
    c += static_cast<char>('A' + i / 26);
    c += static_cast<char>('A' + i % 26);
    ccys.push_back(c);  // CAA, CAB, ... synthetic ISO-like codes
  }
  std::map<std::string, double> spots, rates;
  rates["USD"] = 0.05;
  for (int i = 0; i < 45; ++i) {
    spots[ccys[i]] = 0.5 + 0.02 * i;
    rates[ccys[i]] = 0.01 + 0.001 * (i % 10);
  }
  const Market market(spots, rates);

  // ---- 250 positions: 200 spots + 50 forwards (5 fwd currencies) ---------
  std::vector<Position> positions;
  for (int i = 0; i < 200; ++i) {
    const std::string pair = ccys[i % 45] + "USD";
    const double notional = (i % 2 ? -1.0 : 1.0) * (1e6 + 2e4 * i);
    positions.push_back(SpotPosition{pair, notional, {}});
  }
  for (int i = 0; i < 50; ++i) {
    const std::string pair = ccys[i % 4] + "USD";  // 4 fwd ccys + USD leg
    positions.push_back(
        ForwardPosition{pair, (i % 2 ? -1.0 : 1.0) * 2e6, 0.25 + 0.05 * i, {}});
  }
  const Book book(positions);
  const CompiledBook compiled(book, market);
  const auto factors = compiled.factors();
  std::printf("book: %zu positions, %zu factors\n", positions.size(),
              factors.size());

  // ---- 500-day history drawn from a full-rank covariance ----------------
  // AR(1) correlation rho^|i-j| (rho = 0.5; eigenvalues in [1/3, 3], all
  // positive), FX vol 0.6 %/day, IR vol 0.04 %/day, fixed seed. The sample
  // covariance of 500 days on 50 factors is then full rank as well.
  FactorCov truth;
  truth.factors = factors;
  truth.cov = Matrix(factors.size(), factors.size());
  for (std::size_t i = 0; i < factors.size(); ++i)
    for (std::size_t j = 0; j < factors.size(); ++j) {
      const double si = (factors[i].rfind("FX:", 0) == 0) ? 0.006 : 0.0004;
      const double sj = (factors[j].rfind("FX:", 0) == 0) ? 0.006 : 0.0004;
      truth.cov(i, j) = si * sj *
          std::pow(0.5, std::abs(static_cast<double>(i) - static_cast<double>(j)));
    }
  const ReturnsMatrix rets = simulate_factor_returns(truth, 500, McDist::kNormal, 6.0, {}, 7);

  // ---- historical VaR (500 scenarios, full reval) ------------------------
  {
    HistoricalOptions opts;
    opts.warn_pegs = false;
    auto t0 = Clock::now();
    const int reps = 20;
    HistoricalResult r;
    for (int i = 0; i < reps; ++i) r = historical_var(book, market, rets, opts);
    const double ms = ms_since(t0) / reps;
    std::printf("historical  VaR (500 scen): var=%.0f es=%.0f   %8.3f ms\n",
                r.var, r.es, ms);
  }

  // ---- parametric VaR ----------------------------------------------------
  {
    ParametricOptions opts;
    opts.warn_pegs = false;
    auto t0 = Clock::now();
    const int reps = 20;
    ParametricResult r;
    for (int i = 0; i < reps; ++i) r = parametric_var(book, market, rets, opts);
    const double ms = ms_since(t0) / reps;
    std::printf("parametric  VaR (normal)  : var=%.0f es=%.0f   %8.3f ms\n",
                r.var, r.es, ms);
  }

  // ---- Monte Carlo, 100k scenarios, full reval ---------------------------
  const FactorCov cov = sample_cov(rets);
  {
    auto t0 = Clock::now();
    const int reps = 20;
    CholeskyResult ch;
    for (int i = 0; i < reps; ++i) ch = cholesky(cov.cov);
    std::printf("cholesky %zux%zu          : jitter=%g          %8.3f ms\n",
                cov.cov.rows(), cov.cov.rows(), ch.jitter, ms_since(t0) / reps);
  }
  for (const auto& [dist, label] :
       {std::pair<McDist, const char*>{McDist::kNormal, "normal"},
        std::pair<McDist, const char*>{McDist::kStudentT, "t(5)  "}}) {
    MonteCarloOptions opts;
    opts.n_scenarios = 100000;
    opts.seed = 42;
    opts.dist = dist;
    opts.df = 5.0;
    auto t0 = Clock::now();
    const int reps = 3;
    MonteCarloResult r;
    for (int i = 0; i < reps; ++i) r = monte_carlo_var(book, market, cov, opts);
    const double ms = ms_since(t0) / reps;
    std::printf("monte carlo VaR 100k %s: var=%.0f es=%.0f se=%.0f (kde %.0f) %8.1f ms\n",
                label, r.var, r.es, r.se_var, r.se_var_kde, ms);
  }
  return 0;
}
