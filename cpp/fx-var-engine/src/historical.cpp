#include "fxvar/historical.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "fxvar/expected_shortfall.hpp"
#include "fxvar/stats.hpp"

namespace fxvar {

namespace {

// Peg warnings, scenario weights and (for kFiltered) the per-factor FHS
// rescaling of `rets` in place. Shared by the linear and option drivers.
std::vector<double> prepare_scenarios(ReturnsMatrix& rets, const HistoricalOptions& opts,
                                      HistoricalResult& res) {
  if (opts.warn_pegs) {
    res.flagged_peg_factors = flag_peg_factors(rets);
    for (const auto& f : res.flagged_peg_factors) {
      std::ostringstream msg;
      msg << "peg blindness: factor " << f << " has daily vol < " << kPegVolThreshold
          << " (pegged/managed currency). Historical and parametric VaR are "
             "blind to peg-break risk; add the peg-break stress add-on "
             "(fxvar::peg_break_scenario).";
      res.warnings.push_back(msg.str());
    }
  }

  const std::size_t n = rets.n_obs();
  std::vector<double> weights;
  if (opts.method == HsMethod::kPlain) {
    weights.assign(n, 1.0 / static_cast<double>(n));
  } else if (opts.method == HsMethod::kAge) {
    if (!(opts.decay > 0.0 && opts.decay < 1.0)) {
      std::ostringstream msg;
      msg << "decay must be in (0, 1), got " << opts.decay;
      throw std::invalid_argument(msg.str());
    }
    weights.resize(n);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const auto age = static_cast<double>(n - 1 - i);  // last row = age 0
      weights[i] = std::pow(opts.decay, age);
      sum += weights[i];
    }
    for (double& w : weights) w /= sum;
  } else {  // kFiltered: per-factor FHS
    const EwmaVolatility ev = ewma_volatility(rets, opts.ewma_lambda);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < rets.n_factors(); ++j)
        rets.data(i, j) = rets.data(i, j) / ev.sigma(i, j) * ev.sigma_next[j];
    weights.assign(n, 1.0 / static_cast<double>(n));
  }
  return weights;
}

}  // namespace

HistoricalResult historical_var(const Book& book, const Market& market,
                                const ReturnsMatrix& returns, const HistoricalOptions& opts) {
  validate_alpha(opts.alpha);
  validate_horizon(opts.horizon_days);
  const CompiledBook compiled(book, market);  // throws on empty book

  HistoricalResult res;
  res.alpha = opts.alpha;
  res.horizon_days = opts.horizon_days;
  res.method = opts.method;

  const std::vector<std::string>& factors = compiled.factors();
  if (factors.empty()) {
    // Pure base-ccy (or USD-in-USD-book) cash: zero risk by construction.
    res.pnl.assign(returns.n_obs(), 0.0);
    res.weights.assign(returns.n_obs(),
                       returns.n_obs() ? 1.0 / static_cast<double>(returns.n_obs()) : 0.0);
    return res;
  }

  validate_returns(returns, factors, opts.min_obs);
  ReturnsMatrix rets = returns.select(factors);
  const std::vector<double> weights = prepare_scenarios(rets, opts, res);

  res.pnl = compiled.pnl(rets);
  res.weights = weights;
  const auto [var1, es1] = empirical_var_es(res.pnl, opts.alpha, weights);
  const double scale = std::sqrt(opts.horizon_days);
  res.var = var1 * scale;
  res.es = es1 * scale;
  return res;
}

HistoricalResult historical_var(const Book& book, const Market& market,
                                std::span<const OptionLeg> options,
                                const std::vector<std::string>& option_factors,
                                const ReturnsMatrix& returns, const HistoricalOptions& opts,
                                double dt) {
  if (options.empty()) return historical_var(book, market, returns, opts);
  validate_alpha(opts.alpha);
  validate_horizon(opts.horizon_days);
  if (opts.horizon_days != 1.0)
    throw std::invalid_argument(
        "historical_var with options: sqrt-time scaling is not valid for a non-linear book; "
        "pass horizon_days = 1 and h-day returns instead");
  const OptionBookLayout layout = layout_option_book(book, market, options, option_factors);

  HistoricalResult res;
  res.alpha = opts.alpha;
  res.horizon_days = opts.horizon_days;
  res.method = opts.method;

  validate_returns(returns, layout.factors, opts.min_obs);
  ReturnsMatrix rets = returns.select(layout.factors);
  const std::vector<double> weights = prepare_scenarios(rets, opts, res);

  res.pnl = option_book_pnl(layout, rets, dt);
  res.weights = weights;
  const auto [var1, es1] = empirical_var_es(res.pnl, opts.alpha, weights);
  res.var = var1;
  res.es = es1;
  return res;
}

}  // namespace fxvar
