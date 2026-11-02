// fxvar/option_reval.cpp: see option_reval.hpp.
#include "fxvar/option_reval.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace fxvar {
namespace {

double ncdf(double x) { return 0.5 * std::erfc(-x / std::numbers::sqrt2); }
double npdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi); }

void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("option_reval: ") + what);
}

void check_leg(const OptionLeg& l) {
  require(std::isfinite(l.quantity) && std::isfinite(l.rate) && std::isfinite(l.carry),
          "quantity, rate and carry must be finite");
  require(std::isfinite(l.spot) && l.spot > 0.0, "spot must be > 0");
  require(std::isfinite(l.strike) && l.strike > 0.0, "strike must be > 0");
  require(std::isfinite(l.tau) && l.tau > 0.0, "tau must be > 0");
  require(std::isfinite(l.vol) && l.vol > 0.0, "vol must be > 0");
}

void check_scenarios(std::span<const OptionLeg> legs, std::span<const double> x,
                     std::span<const double> dv, std::size_t nf) {
  require(!legs.empty(), "legs must not be empty");
  require(nf > 0 && x.size() % nf == 0, "scenario size must be a multiple of n_factors");
  require(dv.empty() || dv.size() == x.size(), "vol_shocks must match spot_log_returns");
  for (const auto& l : legs) {
    check_leg(l);
    require(l.factor < nf, "leg factor index out of range");
  }
  for (double v : x) require(std::isfinite(v), "spot_log_returns must be finite");
  for (double v : dv) require(std::isfinite(v), "vol_shocks must be finite");
}

double price_unchecked(double S, double K, double t, double r, double b, double v,
                       OptionKind kind) {
  const double s = kind == OptionKind::Call ? 1.0 : -1.0;
  if (t <= 0.0) return std::max(s * (S - K), 0.0);
  v = std::max(v, 1e-12);
  const double sq = v * std::sqrt(t);
  const double d1 = (std::log(S / K) + (r - b + 0.5 * v * v) * t) / sq;
  const double d2 = d1 - sq;
  return s * (S * std::exp(-b * t) * ncdf(s * d1) - K * std::exp(-r * t) * ncdf(s * d2));
}

}  // namespace

double bsm_price(double spot, double strike, double tau, double rate, double carry, double vol,
                 OptionKind kind) {
  require(std::isfinite(spot) && spot > 0.0 && std::isfinite(strike) && strike > 0.0,
          "spot and strike must be > 0");
  require(std::isfinite(tau) && std::isfinite(rate) && std::isfinite(carry), "non-finite input");
  require(std::isfinite(vol) && vol > 0.0, "vol must be > 0");
  return price_unchecked(spot, strike, tau, rate, carry, vol, kind);
}

BsmGreeks bsm_greeks(const OptionLeg& l) {
  check_leg(l);
  const double s = l.kind == OptionKind::Call ? 1.0 : -1.0;
  const double rt = std::sqrt(l.tau);
  const double sq = l.vol * rt;
  const double d1 =
      (std::log(l.spot / l.strike) + (l.rate - l.carry + 0.5 * l.vol * l.vol) * l.tau) / sq;
  const double d2 = d1 - sq;
  const double eb = std::exp(-l.carry * l.tau);
  const double er = std::exp(-l.rate * l.tau);
  const double pdf = npdf(d1);
  const double vega = l.spot * eb * pdf * rt;
  BsmGreeks g{};
  g.delta = s * eb * ncdf(s * d1);
  g.gamma = eb * pdf / (l.spot * sq);
  g.vega = vega;
  g.vanna = -eb * pdf * d2 / l.vol;
  g.volga = vega * d1 * d2 / l.vol;
  g.theta = -l.spot * eb * pdf * l.vol / (2.0 * rt) + s * l.carry * l.spot * eb * ncdf(s * d1) -
            s * l.rate * l.strike * er * ncdf(s * d2);
  return g;
}

std::vector<double> full_revaluation_pnl(std::span<const OptionLeg> legs, std::span<const double> x,
                                         std::span<const double> dv, std::size_t nf, double dt) {
  return full_revaluation_pnl(legs, x, dv, nf, dt, VolMode::StickyStrike, {});
}

std::vector<double> full_revaluation_pnl(std::span<const OptionLeg> legs, std::span<const double> x,
                                         std::span<const double> dv, std::size_t nf, double dt,
                                         VolMode mode, std::span<const double> skews) {
  check_scenarios(legs, x, dv, nf);
  require(std::isfinite(dt) && dt >= 0.0, "dt must be >= 0");
  require(skews.empty() || skews.size() == legs.size(), "skews must be empty or one per leg");
  for (double k : skews) require(std::isfinite(k), "skews must be finite");
  require(mode == VolMode::StickyStrike || mode == VolMode::StickyDelta ||
              mode == VolMode::HistoricalAtm,
          "unknown vol mode");
  const std::size_t n = x.size() / nf;
  std::vector<double> pnl(n, 0.0);
  for (std::size_t k = 0; k < legs.size(); ++k) {
    const OptionLeg& l = legs[k];
    const double skew = skews.empty() ? 0.0 : skews[k];
    const double p0 = price_unchecked(l.spot, l.strike, l.tau, l.rate, l.carry, l.vol, l.kind);
    const double t1 = std::max(l.tau - dt, 0.0);
    for (std::size_t r = 0; r < n; ++r) {
      const double xr = x[r * nf + l.factor];
      const double d = dv.empty() ? 0.0 : dv[r * nf + l.factor];
      const double s1 = l.spot * std::exp(xr);
      double v1 = l.vol + d;
      if (mode == VolMode::StickyDelta) v1 = l.vol - skew * xr + d;
      if (mode == VolMode::HistoricalAtm) v1 = l.vol * std::exp(d);
      pnl[r] += l.quantity * (price_unchecked(s1, l.strike, t1, l.rate, l.carry, v1, l.kind) - p0);
    }
  }
  return pnl;
}

std::vector<double> delta_gamma_pnl(std::span<const OptionLeg> legs, std::span<const double> x,
                                    std::size_t nf) {
  check_scenarios(legs, x, {}, nf);
  const std::size_t n = x.size() / nf;
  std::vector<double> pnl(n, 0.0);
  for (const auto& l : legs) {
    const BsmGreeks g = bsm_greeks(l);
    for (std::size_t i = 0; i < n; ++i) {
      const double ds = l.spot * std::expm1(x[i * nf + l.factor]);
      pnl[i] += l.quantity * (g.delta * ds + 0.5 * g.gamma * ds * ds);
    }
  }
  return pnl;
}

OptionBookLayout layout_option_book(const Book& book, const Market& market,
                                    std::span<const OptionLeg> options,
                                    const std::vector<std::string>& option_factors) {
  require(!options.empty(), "no option legs");
  OptionBookLayout out;
  if (!book.empty()) {
    require(book.base() == "USD", "a book holding options must report in USD");
    out.linear.emplace(book, market);
    out.factors = out.linear->factors();
  }
  for (const auto& name : option_factors) {
    require(name.size() > 3 && name.starts_with("FX:") && name != "FX:USD",
            "option factors must be named FX:CCY with CCY != USD");
  }
  out.legs.assign(options.begin(), options.end());
  for (auto& l : out.legs) {
    check_leg(l);
    require(l.factor < option_factors.size(), "leg factor index out of range");
    const std::string& name = option_factors[l.factor];
    auto it = std::find(out.factors.begin(), out.factors.end(), name);
    if (it == out.factors.end()) {
      out.factors.push_back(name);
      it = out.factors.end() - 1;
    }
    l.factor = static_cast<std::size_t>(it - out.factors.begin());
  }
  return out;
}

std::vector<double> option_book_pnl(const OptionBookLayout& layout, const ReturnsMatrix& scenarios,
                                    double dt) {
  const ReturnsMatrix rets = scenarios.select(layout.factors);
  const std::size_t nf = layout.factors.size();
  const std::vector<double>& x = rets.data.data();
  std::vector<double> pnl = full_revaluation_pnl(layout.legs, x, {}, nf, dt);
  if (layout.linear && !layout.linear->factors().empty()) {
    const std::vector<double> lin = layout.linear->pnl(rets);
    for (std::size_t i = 0; i < pnl.size(); ++i) pnl[i] += lin[i];
  }
  return pnl;
}

std::vector<std::string> option_vol_factors(const OptionBookLayout& layout) {
  std::vector<bool> used(layout.factors.size(), false);
  for (const auto& l : layout.legs) used[l.factor] = true;
  std::vector<std::string> out;
  for (std::size_t j = 0; j < layout.factors.size(); ++j)
    if (used[j]) out.push_back("VOL:" + layout.factors[j].substr(3));
  return out;
}

std::vector<std::string> option_combined_factors(const OptionBookLayout& layout) {
  std::vector<std::string> out = layout.factors;
  for (auto& v : option_vol_factors(layout)) out.push_back(std::move(v));
  return out;
}

std::vector<double> option_book_pnl(const OptionBookLayout& layout, const ReturnsMatrix& scenarios,
                                    double dt, const VolScenarios& vol) {
  require(vol.skews.empty() || vol.skews.size() == layout.legs.size(),
          "skews must be empty or one per option leg");
  const ReturnsMatrix rets = scenarios.select(layout.factors);
  const std::vector<std::string> vnames = option_vol_factors(layout);
  const ReturnsMatrix vols = scenarios.select(vnames);
  const std::size_t nf = layout.factors.size();
  const std::size_t n = rets.n_obs();
  // dv on the spot axis: column j holds the vol shock of factor j (zero for
  // factors no option references; full_revaluation_pnl never reads those).
  std::vector<double> dv(n * nf, 0.0);
  std::size_t k = 0;
  for (std::size_t j = 0; j < nf; ++j) {
    if (k < vnames.size() && vnames[k] == "VOL:" + layout.factors[j].substr(3)) {
      for (std::size_t i = 0; i < n; ++i) dv[i * nf + j] = vols.data(i, k);
      ++k;
    }
  }
  const std::vector<double>& x = rets.data.data();
  std::vector<double> pnl = full_revaluation_pnl(layout.legs, x, dv, nf, dt, vol.mode, vol.skews);
  if (layout.linear && !layout.linear->factors().empty()) {
    const std::vector<double> lin = layout.linear->pnl(rets);
    for (std::size_t i = 0; i < pnl.size(); ++i) pnl[i] += lin[i];
  }
  return pnl;
}

}  // namespace fxvar
