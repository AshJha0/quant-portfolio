// Benchmark: 250-position, 50-factor FX book - historical, parametric
// and 100k-scenario Monte Carlo VaR wall times.
//
// The book is deterministic: 45 non-USD currencies (45 FX factors),
// forwards on 4 currency pairs adding 5 IR factors (incl. IR:USD) for a
// 50-factor set; 200 spot positions + 50 forwards = 250 positions.
// History is 500 days of sinusoidal factor returns.
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

  // ---- 500-day deterministic history -------------------------------------
  ReturnsMatrix rets;
  rets.factors = factors;
  rets.data = Matrix(500, factors.size());
  for (std::size_t t = 0; t < 500; ++t)
    for (std::size_t j = 0; j < factors.size(); ++j) {
      const double scale = (factors[j].rfind("FX:", 0) == 0) ? 0.006 : 0.0004;
      rets.data(t, j) =
          scale * std::sin(0.1 * static_cast<double>(t) +
                           0.37 * static_cast<double>(j));
    }

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
    std::printf("monte carlo VaR 100k %s: var=%.0f es=%.0f   %8.1f ms\n",
                label, r.var, r.es, ms);
  }
  return 0;
}
