#include "fxvar/returns.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace fxvar {

int ReturnsMatrix::column_index(const std::string& factor) const {
  for (std::size_t j = 0; j < factors.size(); ++j)
    if (factors[j] == factor) return static_cast<int>(j);
  return -1;
}

ReturnsMatrix ReturnsMatrix::select(
    const std::vector<std::string>& wanted) const {
  std::vector<int> idx;
  std::vector<std::string> missing;
  idx.reserve(wanted.size());
  for (const auto& f : wanted) {
    const int j = column_index(f);
    if (j < 0)
      missing.push_back(f);
    else
      idx.push_back(j);
  }
  if (!missing.empty()) {
    std::ostringstream msg;
    msg << "returns is missing required factor columns:";
    for (const auto& f : missing) msg << " " << f;
    throw std::invalid_argument(msg.str());
  }
  ReturnsMatrix out;
  out.factors = wanted;
  out.data = Matrix(n_obs(), wanted.size());
  for (std::size_t i = 0; i < n_obs(); ++i)
    for (std::size_t j = 0; j < idx.size(); ++j)
      out.data(i, j) = data(i, static_cast<std::size_t>(idx[j]));
  return out;
}

FactorCov FactorCov::select(const std::vector<std::string>& wanted) const {
  std::vector<int> idx;
  std::vector<std::string> missing;
  for (const auto& f : wanted) {
    int found = -1;
    for (std::size_t j = 0; j < factors.size(); ++j)
      if (factors[j] == f) {
        found = static_cast<int>(j);
        break;
      }
    if (found < 0)
      missing.push_back(f);
    else
      idx.push_back(found);
  }
  if (!missing.empty()) {
    std::ostringstream msg;
    msg << "cov is missing required factor columns:";
    for (const auto& f : missing) msg << " " << f;
    throw std::invalid_argument(msg.str());
  }
  FactorCov out;
  out.factors = wanted;
  out.cov = Matrix(wanted.size(), wanted.size());
  for (std::size_t i = 0; i < idx.size(); ++i)
    for (std::size_t j = 0; j < idx.size(); ++j)
      out.cov(i, j) = cov(static_cast<std::size_t>(idx[i]),
                          static_cast<std::size_t>(idx[j]));
  return out;
}

void validate_returns(const ReturnsMatrix& returns,
                      const std::vector<std::string>& required,
                      std::size_t min_obs) {
  if (returns.data.cols() != returns.factors.size())
    throw std::invalid_argument(
        "returns: factor labels do not match the data column count");
  if (returns.n_obs() < min_obs) {
    std::ostringstream msg;
    msg << "insufficient history: " << returns.n_obs()
        << " rows < min_obs=" << min_obs
        << "; VaR quantiles are meaningless on this sample";
    throw std::invalid_argument(msg.str());
  }
  std::vector<std::string> missing;
  for (const auto& f : required)
    if (returns.column_index(f) < 0) missing.push_back(f);
  if (!missing.empty()) {
    std::ostringstream msg;
    msg << "returns is missing required factor columns:";
    for (const auto& f : missing) msg << " " << f;
    throw std::invalid_argument(msg.str());
  }
  if (returns.data.has_nan())
    throw std::invalid_argument(
        "returns contains NaNs; clean or drop them explicitly before "
        "calling the engine (NaN policy: refuse, never impute silently)");
}

FactorCov sample_cov(const ReturnsMatrix& returns) {
  const std::size_t n = returns.n_obs();
  const std::size_t k = returns.n_factors();
  if (n < 2)
    throw std::invalid_argument("sample_cov: need at least 2 observations");
  std::vector<double> mean(k, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < k; ++j) mean[j] += returns.data(i, j);
  for (std::size_t j = 0; j < k; ++j) mean[j] /= static_cast<double>(n);
  Matrix cov(k, k, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t a = 0; a < k; ++a) {
      const double da = returns.data(i, a) - mean[a];
      for (std::size_t b = a; b < k; ++b)
        cov(a, b) += da * (returns.data(i, b) - mean[b]);
    }
  }
  const double denom = static_cast<double>(n) - 1.0;
  for (std::size_t a = 0; a < k; ++a)
    for (std::size_t b = a; b < k; ++b) {
      cov(a, b) /= denom;
      cov(b, a) = cov(a, b);
    }
  return FactorCov{returns.factors, std::move(cov)};
}

FactorCov ewma_cov(const ReturnsMatrix& returns, double lam) {
  if (!(lam > 0.0 && lam < 1.0)) {
    std::ostringstream msg;
    msg << "lambda must be in (0, 1), got " << lam;
    throw std::invalid_argument(msg.str());
  }
  FactorCov out = sample_cov(returns);
  const std::size_t n = returns.n_obs();
  const std::size_t k = returns.n_factors();
  for (std::size_t i = 0; i < n; ++i) {
    const double* r = returns.data.row(i);
    for (std::size_t a = 0; a < k; ++a)
      for (std::size_t b = 0; b < k; ++b)
        out.cov(a, b) = lam * out.cov(a, b) + (1.0 - lam) * r[a] * r[b];
  }
  return out;
}

EwmaVolatility ewma_volatility(const ReturnsMatrix& returns, double lam) {
  if (!(lam > 0.0 && lam < 1.0)) {
    std::ostringstream msg;
    msg << "lambda must be in (0, 1), got " << lam;
    throw std::invalid_argument(msg.str());
  }
  const std::size_t n = returns.n_obs();
  const std::size_t k = returns.n_factors();
  if (n < 2)
    throw std::invalid_argument("ewma_volatility: need at least 2 observations");
  // Seed with the ddof=1 sample variance per factor, floored at 1e-18.
  std::vector<double> mean(k, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < k; ++j) mean[j] += returns.data(i, j);
  for (std::size_t j = 0; j < k; ++j) mean[j] /= static_cast<double>(n);
  std::vector<double> sig2(k, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < k; ++j) {
      const double d = returns.data(i, j) - mean[j];
      sig2[j] += d * d;
    }
  for (std::size_t j = 0; j < k; ++j)
    sig2[j] = std::max(sig2[j] / (static_cast<double>(n) - 1.0), 1e-18);

  EwmaVolatility out;
  out.sigma = Matrix(n, k);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      out.sigma(i, j) = std::sqrt(sig2[j]);
      const double r = returns.data(i, j);
      sig2[j] = lam * sig2[j] + (1.0 - lam) * r * r;
    }
  }
  out.sigma_next.resize(k);
  for (std::size_t j = 0; j < k; ++j) out.sigma_next[j] = std::sqrt(sig2[j]);
  return out;
}

std::vector<std::string> flag_peg_factors(const ReturnsMatrix& returns,
                                          double threshold) {
  std::vector<std::string> flagged;
  const std::size_t n = returns.n_obs();
  if (n < 2) return flagged;
  for (std::size_t j = 0; j < returns.n_factors(); ++j) {
    const std::string& f = returns.factors[j];
    if (f.rfind("FX:", 0) != 0) continue;
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean += returns.data(i, j);
    mean /= static_cast<double>(n);
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double d = returns.data(i, j) - mean;
      ss += d * d;
    }
    const double sd = std::sqrt(ss / (static_cast<double>(n) - 1.0));
    if (sd < threshold) flagged.push_back(f);
  }
  return flagged;
}

}  // namespace fxvar
