#include "fxvar/expected_shortfall.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>

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

std::pair<double, double> uniform_var_es(const std::vector<double>& pnl,
                                         double alpha) {
  const std::size_t n = pnl.size();
  std::vector<double> loss(n);
  for (std::size_t i = 0; i < n; ++i) loss[i] = -pnl[i];
  const double k = static_cast<double>(n) * alpha;
  const double mc = std::ceil(k - 1e-9);
  const std::size_t m =
      mc < 1.0 ? 1 : std::min(n, static_cast<std::size_t>(mc));
  std::nth_element(loss.begin(),
                   loss.begin() + static_cast<std::ptrdiff_t>(m - 1),
                   loss.end(), std::greater<double>());
  const double lm = loss[m - 1];
  double tail = 0.0;
  for (std::size_t i = 0; i + 1 < m; ++i) tail += loss[i];
  return {lm, (tail + (k - static_cast<double>(m - 1)) * lm) / k};
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

void check_scale(double sigma, double mean) {
  if (!(sigma >= 0.0) || !std::isfinite(sigma))
    throw std::invalid_argument("sigma must be finite and >= 0");
  if (!std::isfinite(mean)) throw std::invalid_argument("mean must be finite");
}

double t_scale(double df) {
  if (!(df > 2.0) || !std::isfinite(df))
    throw std::invalid_argument(
        "Student-t df must be finite and > 2 for finite variance");
  return std::sqrt((df - 2.0) / df);
}

}  // namespace

std::pair<double, double> empirical_var_es(const std::vector<double>& pnl,
                                           double alpha,
                                           const std::vector<double>& weights) {
  validate_alpha(alpha);
  check_pnl(pnl);
  return weights.empty() ? uniform_var_es(pnl, alpha)
                         : weighted_var_es(pnl, alpha, weights);
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
  validate_alpha(alpha);
  check_scale(sigma, mean);
  return -mean + sigma * norm_ppf(1.0 - alpha);
}

double normal_es(double sigma, double alpha, double mean) {
  validate_alpha(alpha);
  check_scale(sigma, mean);
  const double z = norm_ppf(1.0 - alpha);
  return -mean + sigma * norm_pdf(z) / alpha;
}

double t_var(double sigma, double alpha, double df, double mean) {
  validate_alpha(alpha);
  check_scale(sigma, mean);
  const double s = t_scale(df);
  return -mean + sigma * s * t_ppf(1.0 - alpha, df);
}

double t_es(double sigma, double alpha, double df, double mean) {
  validate_alpha(alpha);
  check_scale(sigma, mean);
  const double s = t_scale(df);
  const double q = t_ppf(1.0 - alpha, df);
  const double es_std = t_pdf(q, df) * (df + q * q) / (alpha * (df - 1.0));
  return -mean + sigma * s * es_std;
}

}  // namespace fxvar
