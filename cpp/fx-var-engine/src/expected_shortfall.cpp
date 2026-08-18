#include "fxvar/expected_shortfall.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "fxvar/stats.hpp"

namespace fxvar {

namespace {

constexpr double kWeightTol = 1e-12;

struct Tail {
  std::vector<double> losses;   // descending
  std::vector<double> weights;  // aligned, normalised
  std::size_t idx = 0;          // index of the VaR atom
};

Tail make_tail(const std::vector<double>& pnl, double alpha,
               const std::vector<double>& weights) {
  validate_alpha(alpha);
  if (pnl.empty()) throw std::invalid_argument("pnl sample is empty");
  for (double v : pnl)
    if (std::isnan(v))
      throw std::invalid_argument("pnl sample contains NaNs (NaN policy: refuse)");

  const std::size_t n = pnl.size();
  std::vector<double> w;
  if (weights.empty()) {
    w.assign(n, 1.0 / static_cast<double>(n));
  } else {
    if (weights.size() != n)
      throw std::invalid_argument("weights must match pnl length");
    double sum = 0.0;
    for (double x : weights) {
      if (x < 0.0)
        throw std::invalid_argument("weights must be non-negative and sum > 0");
      sum += x;
    }
    if (!(sum > 0.0))
      throw std::invalid_argument("weights must be non-negative and sum > 0");
    w.resize(n);
    for (std::size_t i = 0; i < n; ++i) w[i] = weights[i] / sum;
  }

  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&pnl](std::size_t a, std::size_t b) {
                     return -pnl[a] > -pnl[b];  // losses descending
                   });

  Tail t;
  t.losses.resize(n);
  t.weights.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    t.losses[i] = -pnl[order[i]];
    t.weights[i] = w[order[i]];
  }
  const double target = 1.0 - alpha;
  double cum = 0.0;
  std::size_t idx = n - 1;
  for (std::size_t i = 0; i < n; ++i) {
    cum += t.weights[i];
    if (cum >= target - kWeightTol) {
      idx = i;
      break;
    }
  }
  t.idx = idx;
  return t;
}

double es_from_tail(const Tail& t, double alpha) {
  const double target = 1.0 - alpha;
  double full = 0.0, cum_before = 0.0;
  for (std::size_t i = 0; i < t.idx; ++i) {
    full += t.losses[i] * t.weights[i];
    cum_before += t.weights[i];
  }
  const double frac = std::max(target - cum_before, 0.0);
  return (full + frac * t.losses[t.idx]) / target;
}

}  // namespace

double empirical_var(const std::vector<double>& pnl, double alpha,
                     const std::vector<double>& weights) {
  const Tail t = make_tail(pnl, alpha, weights);
  return t.losses[t.idx];
}

double empirical_es(const std::vector<double>& pnl, double alpha,
                    const std::vector<double>& weights) {
  const Tail t = make_tail(pnl, alpha, weights);
  return es_from_tail(t, alpha);
}

std::pair<double, double> empirical_var_es(const std::vector<double>& pnl,
                                           double alpha,
                                           const std::vector<double>& weights) {
  const Tail t = make_tail(pnl, alpha, weights);
  return {t.losses[t.idx], es_from_tail(t, alpha)};
}

double normal_var(double sigma, double alpha, double mean) {
  validate_alpha(alpha);
  if (sigma < 0.0) throw std::invalid_argument("sigma must be >= 0");
  return -mean + sigma * norm_ppf(alpha);
}

double normal_es(double sigma, double alpha, double mean) {
  validate_alpha(alpha);
  if (sigma < 0.0) throw std::invalid_argument("sigma must be >= 0");
  const double z = norm_ppf(alpha);
  return -mean + sigma * norm_pdf(z) / (1.0 - alpha);
}

namespace {
double t_scale(double df) {
  if (df <= 2.0)
    throw std::invalid_argument("Student-t df must be > 2 for finite variance");
  return std::sqrt((df - 2.0) / df);
}
}  // namespace

double t_var(double sigma, double alpha, double df, double mean) {
  validate_alpha(alpha);
  if (sigma < 0.0) throw std::invalid_argument("sigma must be >= 0");
  const double q = t_ppf(alpha, df);
  return -mean + sigma * t_scale(df) * q;
}

double t_es(double sigma, double alpha, double df, double mean) {
  validate_alpha(alpha);
  if (sigma < 0.0) throw std::invalid_argument("sigma must be >= 0");
  const double q = t_ppf(alpha, df);
  const double es_std = t_pdf(q, df) * (df + q * q) / ((1.0 - alpha) * (df - 1.0));
  return -mean + sigma * t_scale(df) * es_std;
}

}  // namespace fxvar
