#include "fxvar/backtest.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "fxvar/stats.hpp"

namespace fxvar {

namespace {

/// x * log(y) with the 0 * log(0) = 0 convention.
double xlogy(double x, double y) {
  if (x == 0.0) return 0.0;
  if (y <= 0.0) return -std::numeric_limits<double>::infinity();
  return x * std::log(y);
}

void check_counts(int x, int n) {
  if (n <= 0 || x < 0 || x > n) {
    std::ostringstream msg;
    msg << "need 0 <= n_exceptions <= n_obs, got x=" << x << ", n=" << n;
    throw std::invalid_argument(msg.str());
  }
}

}  // namespace

LrTest kupiec_pof(const KupiecInput& in) {
  validate_alpha(in.alpha);
  check_counts(in.n_exceptions, in.n_obs);
  const double x = in.n_exceptions;
  const double n = in.n_obs;
  const double p = in.alpha;
  const double pi_hat = x / n;
  const double ll0 = xlogy(n - x, 1.0 - p) + xlogy(x, p);
  const double ll1 = xlogy(n - x, 1.0 - pi_hat) + xlogy(x, pi_hat);
  double lr = -2.0 * (ll0 - ll1);
  lr = std::max(lr, 0.0);
  return {lr, chi2_sf(lr, 1.0)};
}

LrTest kupiec_pof(int n_exceptions, int n_obs, double alpha) {
  return kupiec_pof(
      KupiecInput{.n_obs = n_obs, .n_exceptions = n_exceptions, .alpha = alpha});
}

LrTest christoffersen_independence(const std::vector<int>& exceedances) {
  if (exceedances.size() < 2)
    throw std::invalid_argument(
        "need at least 2 observations for the independence test");
  for (int e : exceedances)
    if (e != 0 && e != 1)
      throw std::invalid_argument(
          "exceedances must be a 0/1 (or boolean) series");
  double n00 = 0, n01 = 0, n10 = 0, n11 = 0;
  for (std::size_t t = 1; t < exceedances.size(); ++t) {
    const int prev = exceedances[t - 1];
    const int curr = exceedances[t];
    if (prev == 0 && curr == 0) n00 += 1;
    if (prev == 0 && curr == 1) n01 += 1;
    if (prev == 1 && curr == 0) n10 += 1;
    if (prev == 1 && curr == 1) n11 += 1;
  }
  const double pi01 = (n00 + n01 > 0) ? n01 / (n00 + n01) : 0.0;
  const double pi11 = (n10 + n11 > 0) ? n11 / (n10 + n11) : 0.0;
  const double pi = (n01 + n11) / (n00 + n01 + n10 + n11);
  const double ll0 = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi);
  const double ll1 = xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) +
                     xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11);
  double lr = -2.0 * (ll0 - ll1);
  lr = std::isfinite(lr) ? std::max(lr, 0.0) : 0.0;
  return {lr, chi2_sf(lr, 1.0)};
}

LrTest conditional_coverage(const std::vector<int>& exceedances, double alpha) {
  validate_alpha(alpha);
  int x = 0;
  for (int e : exceedances) {
    if (e != 0 && e != 1)
      throw std::invalid_argument("exceedances must be a 0/1 (or boolean) series");
    x += e;
  }
  const LrTest uc = kupiec_pof(KupiecInput{
      .n_obs = static_cast<int>(exceedances.size()), .n_exceptions = x, .alpha = alpha});
  const LrTest ind = christoffersen_independence(exceedances);
  const double lr = uc.lr + ind.lr;
  return {lr, chi2_sf(lr, 2.0)};
}

TrafficLight basel_traffic_light(const BaselInput& in) {
  validate_alpha(in.alpha);
  check_counts(in.n_exceptions, in.n_obs);
  TrafficLight tl;
  tl.n_exceptions = in.n_exceptions;
  tl.n_obs = in.n_obs;
  tl.cumulative_prob = binom_cdf(in.n_exceptions, in.n_obs, in.alpha);
  const double f = tl.cumulative_prob;
  if (f < 0.95) {
    tl.zone = Zone::kGreen;
    tl.plus_factor = 0.0;
  } else if (f < 0.9999) {
    tl.zone = Zone::kYellow;
    static constexpr double kPlus[] = {0.40, 0.50, 0.65, 0.75, 0.85};
    int x_eq = 9;
    for (int xp = 5; xp <= 9; ++xp) {
      if (binom_cdf(xp, 250, 0.01) >= f - 1e-12) {
        x_eq = xp;
        break;
      }
    }
    tl.plus_factor = kPlus[x_eq - 5];
  } else {
    tl.zone = Zone::kRed;
    tl.plus_factor = 1.0;
  }
  tl.multiplier = 3.0 + tl.plus_factor;
  return tl;
}

TrafficLight basel_traffic_light(int n_exceptions, int n_obs, double alpha) {
  return basel_traffic_light(
      BaselInput{.n_obs = n_obs, .n_exceptions = n_exceptions, .alpha = alpha});
}

double acerbi_szekely_z2(const std::vector<double>& pnl,
                         const std::vector<double>& var,
                         const std::vector<double>& es, double alpha) {
  validate_alpha(alpha);
  if (pnl.empty()) throw std::invalid_argument("acerbi_szekely_z2: empty pnl");
  const std::size_t n = pnl.size();
  if ((var.size() != 1 && var.size() != n) || (es.size() != 1 && es.size() != n))
    throw std::invalid_argument(
        "acerbi_szekely_z2: var and es must have size 1 or one entry per day");
  for (double v : pnl)
    if (!std::isfinite(v))
      throw std::invalid_argument("acerbi_szekely_z2: pnl must be finite");
  for (double v : var)
    if (!std::isfinite(v) || v < 0.0)
      throw std::invalid_argument("acerbi_szekely_z2: VaR must be finite and >= 0");
  for (double v : es)
    if (!std::isfinite(v) || !(v > 0.0))
      throw std::invalid_argument("acerbi_szekely_z2: ES must be finite and > 0");
  const double t = static_cast<double>(n);
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double v = var.size() == 1 ? var[0] : var[i];
    const double e = es.size() == 1 ? es[0] : es[i];
    if (pnl[i] < -v) sum += pnl[i] / (t * alpha * e);
  }
  return sum + 1.0;
}

BacktestResult evaluate_var_backtest(const std::vector<double>& pnl,
                                     const std::vector<double>& var_forecasts,
                                     double alpha) {
  validate_alpha(alpha);
  if (pnl.size() != var_forecasts.size())
    throw std::invalid_argument("pnl and var_forecasts must have equal length");
  if (pnl.size() < 2)
    throw std::invalid_argument("need at least 2 observations to backtest");
  for (std::size_t i = 0; i < pnl.size(); ++i) {
    if (!std::isfinite(pnl[i]) || !std::isfinite(var_forecasts[i]))
      throw std::invalid_argument(
          "backtest inputs contain NaN or inf (NaN policy: refuse)");
    if (var_forecasts[i] < 0.0)
      throw std::invalid_argument("VaR forecasts must be positive losses (>= 0)");
  }

  BacktestResult res;
  res.n_obs = pnl.size();
  res.exceedances.resize(pnl.size());
  int x = 0;
  for (std::size_t i = 0; i < pnl.size(); ++i) {
    res.exceedances[i] = (-pnl[i] > var_forecasts[i]) ? 1 : 0;
    x += res.exceedances[i];
  }
  res.n_exceptions = x;
  res.exception_rate = static_cast<double>(x) / static_cast<double>(pnl.size());
  res.expected_rate = alpha;
  const KupiecInput counts{.n_obs = static_cast<int>(pnl.size()), .n_exceptions = x,
                           .alpha = alpha};
  res.kupiec = kupiec_pof(counts);
  res.independence = christoffersen_independence(res.exceedances);
  const double lr_cc = res.kupiec.lr + res.independence.lr;
  res.conditional = {lr_cc, chi2_sf(lr_cc, 2.0)};
  res.traffic_light = basel_traffic_light(counts);
  return res;
}

}  // namespace fxvar
