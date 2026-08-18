#include "fxvar/stress.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>

#include "fxvar/monte_carlo.hpp"  // Rng

namespace fxvar {

double simple_to_log(double pct) { return std::log1p(pct); }

std::map<std::string, Scenario> historical_scenarios() {
  std::map<std::string, Scenario> lib;
  lib["brexit_2016"] = Scenario{
      "GBP flash - Brexit referendum (24 Jun 2016)",
      {
          {fx_factor("GBP"), simple_to_log(-0.081)},
          {fx_factor("EUR"), simple_to_log(-0.024)},
          {fx_factor("JPY"), simple_to_log(+0.039)},
          {fx_factor("CHF"), simple_to_log(+0.015)},
          {fx_factor("AUD"), simple_to_log(-0.019)},
          {ir_factor("GBP"), -0.0025},  // BoE easing repricing
      },
      "Cable -8.1% in a day; safe havens bid; front-end GBP rates "
      "reprice 25bp lower."};
  lib["chf_depeg_2015"] = Scenario{
      "CHF depeg - SNB floor removal (15 Jan 2015)",
      {
          {fx_factor("CHF"), simple_to_log(+0.149)},
          {fx_factor("EUR"), simple_to_log(-0.014)},
          {fx_factor("JPY"), simple_to_log(+0.012)},
          {ir_factor("CHF"), -0.0050},  // SNB cut to -0.75%
      },
      "CHF +14.9% vs USD close-to-close (intraday >+30%); the peg-break "
      "archetype - invisible to a 250d HS window."};
  lib["jpy_1998"] = Scenario{
      "JPY carry unwind (7-8 Oct 1998)",
      {
          {fx_factor("JPY"), simple_to_log(+0.115)},
          {fx_factor("AUD"), simple_to_log(-0.040)},
          {fx_factor("NZD"), simple_to_log(-0.040)},
          {fx_factor("CHF"), simple_to_log(+0.020)},
      },
      "USDJPY 131 -> 117 in two sessions as levered carry unwound."};
  return lib;
}

Scenario usd_broad_move(const std::vector<std::string>& ccys, double pct) {
  if (pct <= -1.0) throw std::invalid_argument("pct must be > -100%");
  Scenario sc;
  {
    std::ostringstream name;
    name << "USD " << (pct >= 0 ? "+" : "") << pct * 100.0 << "% broad move";
    sc.name = name.str();
  }
  sc.description = "Uniform USD move against all book currencies.";
  for (const auto& c : ccys) {
    std::string u = c;
    for (char& ch : u) ch = static_cast<char>(std::toupper(
        static_cast<unsigned char>(ch)));
    if (u == "USD") continue;
    // USD +pct vs CCY means CCYUSD falls by pct/(1+pct) in simple terms.
    sc.shocks[fx_factor(u)] = simple_to_log(-pct / (1.0 + pct));
  }
  return sc;
}

Scenario peg_break_scenario(const std::string& ccy, double jump,
                            const std::map<std::string, double>& contagion) {
  if (jump <= -1.0) throw std::invalid_argument("jump must be > -100%");
  Scenario sc;
  sc.shocks[fx_factor(ccy)] = simple_to_log(jump);
  for (const auto& [c, m] : contagion) sc.shocks[fx_factor(c)] = simple_to_log(m);
  std::string u = ccy;
  for (char& ch : u) ch = static_cast<char>(std::toupper(
      static_cast<unsigned char>(ch)));
  const char* direction = (jump < 0) ? "devaluation" : "revaluation";
  {
    std::ostringstream name;
    name << u << " peg break (" << (jump >= 0 ? "+" : "") << jump * 100.0
         << "% " << direction << ")";
    sc.name = name.str();
  }
  {
    std::ostringstream desc;
    desc << "Managed-currency regime break: " << u << " gaps "
         << (jump >= 0 ? "+" : "") << jump * 100.0
         << "% vs USD with no intermediate prints; HS/parametric VaR see "
            "none of it.";
    sc.description = desc.str();
  }
  return sc;
}

std::vector<StressRow> run_stress(
    const Book& book, const Market& market,
    const std::map<std::string, Scenario>& scenarios) {
  const CompiledBook compiled(book, market);  // throws on empty book
  const std::set<std::string> factors(compiled.factors().begin(),
                                      compiled.factors().end());
  std::vector<StressRow> rows;
  rows.reserve(scenarios.size());
  for (const auto& [key, sc] : scenarios) {
    std::map<std::string, double> filtered;
    for (const auto& [f, v] : sc.shocks)
      if (factors.count(f)) filtered[f] = v;
    rows.push_back({key, sc.name, compiled.pnl(filtered), sc.description});
  }
  std::sort(rows.begin(), rows.end(),
            [](const StressRow& a, const StressRow& b) { return a.pnl < b.pnl; });
  return rows;
}

// ---------------------------------------------------------- reverse stress
namespace {

double checked_sigma_p(const std::vector<double>& w, const Matrix& cov) {
  const double sp2 = quad_form(w, cov);
  const double sp = std::sqrt(std::max(sp2, 0.0));
  if (!(sp > 0.0))
    throw std::invalid_argument(
        "book has zero linear risk; reverse stress undefined");
  return sp;
}

}  // namespace

ReverseStress reverse_stress_linear(const std::vector<double>& exposures,
                                    const Matrix& cov, double radius) {
  if (!(radius > 0.0))
    throw std::invalid_argument("radius / loss_target must be positive");
  const double sp = checked_sigma_p(exposures, cov);
  const std::vector<double> sw = matvec(cov, exposures);
  ReverseStress out;
  out.shocks.resize(exposures.size());
  for (std::size_t i = 0; i < exposures.size(); ++i)
    out.shocks[i] = -radius * sw[i] / sp;
  out.loss = radius * sp;
  return out;
}

ReverseStress reverse_stress_for_loss(const std::vector<double>& exposures,
                                      const Matrix& cov, double loss_target) {
  const double sp = checked_sigma_p(exposures, cov);
  return reverse_stress_linear(exposures, cov, loss_target / sp);
}

ReverseStress reverse_stress_numerical(const std::vector<double>& exposures,
                                       const Matrix& cov, double radius,
                                       std::uint64_t seed) {
  if (!(radius > 0.0)) throw std::invalid_argument("radius must be positive");
  checked_sigma_p(exposures, cov);
  const std::size_t n = exposures.size();

  // Whitened coordinates: dx = L y with L L' = Sigma turns the
  // Mahalanobis ellipsoid dx' Sigma^-1 dx <= k^2 into the sphere
  // |y| <= k.  Maximise loss(y) = -w' L y by projected gradient ascent
  // with finite-difference gradients from a seeded random start - an
  // independent check that never assumes the analytic optimum.
  const CholeskyResult chol = robust_cholesky(cov);
  const Matrix& lower = chol.lower;

  const auto loss_of = [&](const std::vector<double>& y) {
    double loss = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double* li = lower.row(i);
      double dxi = 0.0;
      for (std::size_t j = 0; j <= i; ++j) dxi += li[j] * y[j];
      loss -= exposures[i] * dxi;
    }
    return loss;
  };
  const auto project = [&](std::vector<double>& y) {
    double norm = 0.0;
    for (double v : y) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 0.0)
      for (double& v : y) v *= radius / norm;
  };

  Rng rng(seed);
  std::vector<double> y(n);
  for (double& v : y) v = rng.normal();
  project(y);
  if (loss_of(y) < 0.0)  // start on the profitable side: flip
    for (double& v : y) v = -v;

  const double h = 1e-7 * radius;
  double step = radius;
  double best = loss_of(y);
  std::vector<double> grad(n), trial(n);
  for (int it = 0; it < 200; ++it) {
    for (std::size_t j = 0; j < n; ++j) {
      trial = y;
      trial[j] += h;
      const double up = loss_of(trial);
      trial[j] -= 2.0 * h;
      const double dn = loss_of(trial);
      grad[j] = (up - dn) / (2.0 * h);
    }
    double gnorm = 0.0;
    for (double g : grad) gnorm += g * g;
    gnorm = std::sqrt(gnorm);
    if (gnorm == 0.0) break;
    bool improved = false;
    while (step > 1e-14 * radius) {
      for (std::size_t j = 0; j < n; ++j)
        trial[j] = y[j] + step * radius * grad[j] / gnorm;
      project(trial);
      const double cand = loss_of(trial);
      if (cand > best) {
        y = trial;
        best = cand;
        improved = true;
        break;
      }
      step *= 0.5;
    }
    if (!improved) break;
  }

  ReverseStress out;
  out.shocks.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double* li = lower.row(i);
    double dxi = 0.0;
    for (std::size_t j = 0; j <= i; ++j) dxi += li[j] * y[j];
    out.shocks[i] = dxi;
  }
  out.loss = best;
  return out;
}

}  // namespace fxvar
