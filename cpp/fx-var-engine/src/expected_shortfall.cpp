#include "fxvar/expected_shortfall.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "fxvar/kernels.hpp"
#include "fxvar/stats.hpp"

namespace fxvar {

namespace {

void check_pnl(const std::vector<double>& pnl) {
  if (pnl.empty()) throw std::invalid_argument("pnl sample is empty");
  for (double v : pnl)
    if (!std::isfinite(v))
      throw std::invalid_argument(
          "pnl sample contains NaN or inf (NaN policy: refuse)");
}

std::pair<double, double> weighted_var_es(const std::vector<double>& pnl,
                                          double alpha,
                                          const std::vector<double>& weights) {
  const std::size_t n = pnl.size();
  if (weights.size() != n)
    throw std::invalid_argument("weights must match pnl length");
  double sum = 0.0;
  for (double x : weights) {
    if (!std::isfinite(x) || x < 0.0)
      throw std::invalid_argument("weights must be finite, non-negative and sum > 0");
    sum += x;
  }
  if (!(sum > 0.0) || !std::isfinite(sum))
    throw std::invalid_argument("weights must be finite, non-negative and sum > 0");
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [&pnl](std::size_t a, std::size_t b) {
                     return -pnl[a] > -pnl[b];  // losses descending
                   });
  double cum = 0.0, tail = 0.0;
  std::size_t m = n - 1;
  for (std::size_t j = 0; j < n; ++j) {
    const double wj = weights[order[j]] / sum;
    if (cum + wj >= alpha - 1e-12 || j + 1 == n) {
      m = j;
      break;
    }
    cum += wj;
    tail += wj * -pnl[order[j]];
  }
  const double lm = -pnl[order[m]];
  return {lm, (tail + std::max(alpha - cum, 0.0) * lm) / alpha};
}

}  // namespace

std::pair<double, double> empirical_var_es(const std::vector<double>& pnl,
                                           double alpha,
                                           const std::vector<double>& weights) {
  if (weights.empty()) {
    std::vector<double> ws(pnl.size());
    return value_or_throw(try_empirical_var_es(pnl, alpha, ws));
  }
  validate_alpha(alpha);
  check_pnl(pnl);
  return weighted_var_es(pnl, alpha, weights);
}

double empirical_var(const std::vector<double>& pnl, double alpha,
                     const std::vector<double>& weights) {
  return empirical_var_es(pnl, alpha, weights).first;
}

double empirical_es(const std::vector<double>& pnl, double alpha,
                    const std::vector<double>& weights) {
  return empirical_var_es(pnl, alpha, weights).second;
}

double normal_var(double sigma, double alpha, double mean) {
  return value_or_throw(try_normal_var(sigma, alpha, mean));
}

double normal_es(double sigma, double alpha, double mean) {
  return value_or_throw(try_normal_es(sigma, alpha, mean));
}

double t_var(double sigma, double alpha, double df, double mean) {
  return value_or_throw(try_t_var(sigma, alpha, df, mean));
}

double t_es(double sigma, double alpha, double df, double mean) {
  return value_or_throw(try_t_es(sigma, alpha, df, mean));
}

}  // namespace fxvar
