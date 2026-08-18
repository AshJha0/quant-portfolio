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

LrTest kupiec_pof(int n_exceptions, int n_obs, double alpha) {
  validate_alpha(alpha);
  check_counts(n_exceptions, n_obs);
  const double x = n_exceptions;
  const double n = n_obs;
  const double p = 1.0 - alpha;
  const double pi_hat = x / n;
  const double ll0 = xlogy(n - x, 1.0 - p) + xlogy(x, p);
  const double ll1 = xlogy(n - x, 1.0 - pi_hat) + xlogy(x, pi_hat);
  double lr = -2.0 * (ll0 - ll1);
  lr = std::max(lr, 0.0);
  return {lr, chi2_sf(lr, 1.0)};
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
  int x = 0;
  for (int e : exceedances) x += e;
  const LrTest uc = kupiec_pof(x, static_cast<int>(exceedances.size()), alpha);
  const LrTest ind = christoffersen_independence(exceedances);
  const double lr = uc.lr + ind.lr;
  return {lr, chi2_sf(lr, 2.0)};
}

TrafficLight basel_traffic_light(int n_exceptions, int n_obs, double alpha) {
  validate_alpha(alpha);
  check_counts(n_exceptions, n_obs);
  TrafficLight tl;
  tl.n_exceptions = n_exceptions;
  tl.n_obs = n_obs;
  tl.cumulative_prob = binom_cdf(n_exceptions, n_obs, 1.0 - alpha);
  if (tl.cumulative_prob < 0.95) {
    tl.zone = Zone::kGreen;
    tl.multiplier = 3.0;
  } else if (tl.cumulative_prob < 0.9999) {
    tl.zone = Zone::kYellow;
    // 1996 Basel yellow-zone add-ons by exception count (250d, 99%):
    // 5 -> 0.40, 6 -> 0.50, 7 -> 0.65, 8 -> 0.75, 9 -> 0.85.  Yellow at
    // an out-of-table count (non-standard window) takes the nearest
    // boundary add-on, mirroring the Python reference.
    double addon;
    switch (n_exceptions) {
      case 5: addon = 0.40; break;
      case 6: addon = 0.50; break;
      case 7: addon = 0.65; break;
      case 8: addon = 0.75; break;
      case 9: addon = 0.85; break;
      default: addon = (n_exceptions > 9) ? 0.85 : 0.40; break;
    }
    tl.multiplier = 3.0 + addon;
  } else {
    tl.zone = Zone::kRed;
    tl.multiplier = 4.0;
  }
  return tl;
}

BacktestResult evaluate_var_backtest(const std::vector<double>& pnl,
                                     const std::vector<double>& var_forecasts,
                                     double alpha) {
  validate_alpha(alpha);
  if (pnl.size() != var_forecasts.size())
    throw std::invalid_argument("pnl and var_forecasts must have equal length");
  if (pnl.size() < 2)
    throw std::invalid_argument("need at least 2 observations to backtest");
  for (std::size_t i = 0; i < pnl.size(); ++i)
    if (std::isnan(pnl[i]) || std::isnan(var_forecasts[i]))
      throw std::invalid_argument(
          "backtest inputs contain NaNs (NaN policy: refuse)");

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
  res.expected_rate = 1.0 - alpha;
  res.kupiec = kupiec_pof(x, static_cast<int>(pnl.size()), alpha);
  res.independence = christoffersen_independence(res.exceedances);
  const double lr_cc = res.kupiec.lr + res.independence.lr;
  res.conditional = {lr_cc, chi2_sf(lr_cc, 2.0)};
  res.traffic_light =
      basel_traffic_light(x, static_cast<int>(pnl.size()), alpha);
  return res;
}

}  // namespace fxvar
