#include "fxvar/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace fxvar {

Matrix Matrix::from_rows(const std::vector<std::vector<double>>& rows) {
  const std::size_t r = rows.size();
  const std::size_t c = r ? rows.front().size() : 0;
  Matrix m(r, c);
  for (std::size_t i = 0; i < r; ++i) {
    if (rows[i].size() != c)
      throw std::invalid_argument("Matrix::from_rows: ragged rows");
    for (std::size_t j = 0; j < c; ++j) m(i, j) = rows[i][j];
  }
  return m;
}

Matrix Matrix::identity(std::size_t n) {
  Matrix m(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.0;
  return m;
}

Matrix Matrix::transpose() const {
  Matrix t(cols_, rows_);
  for (std::size_t i = 0; i < rows_; ++i)
    for (std::size_t j = 0; j < cols_; ++j) t(j, i) = (*this)(i, j);
  return t;
}

bool Matrix::has_nan() const {
  for (double v : data_)
    if (std::isnan(v)) return true;
  return false;
}

Matrix matmul(const Matrix& a, const Matrix& b) {
  if (a.cols() != b.rows())
    throw std::invalid_argument("matmul: inner dimensions do not match");
  Matrix c(a.rows(), b.cols(), 0.0);
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t k = 0; k < a.cols(); ++k) {
      const double aik = a(i, k);
      if (aik == 0.0) continue;
      const double* brow = b.row(k);
      double* crow = c.row(i);
      for (std::size_t j = 0; j < b.cols(); ++j) crow[j] += aik * brow[j];
    }
  }
  return c;
}

std::vector<double> matvec(const Matrix& a, const std::vector<double>& x) {
  if (a.cols() != x.size())
    throw std::invalid_argument("matvec: dimension mismatch");
  std::vector<double> y(a.rows(), 0.0);
  for (std::size_t i = 0; i < a.rows(); ++i) {
    const double* arow = a.row(i);
    double s = 0.0;
    for (std::size_t j = 0; j < a.cols(); ++j) s += arow[j] * x[j];
    y[i] = s;
  }
  return y;
}

double quad_form(const std::vector<double>& w, const Matrix& a) {
  if (a.rows() != a.cols() || a.rows() != w.size())
    throw std::invalid_argument("quad_form: dimension mismatch");
  double total = 0.0;
  for (std::size_t i = 0; i < a.rows(); ++i) {
    const double* arow = a.row(i);
    double s = 0.0;
    for (std::size_t j = 0; j < a.cols(); ++j) s += arow[j] * w[j];
    total += w[i] * s;
  }
  return total;
}

namespace {

// Plain Cholesky; returns false if a non-positive pivot is met.
bool try_cholesky(const Matrix& a, Matrix& lower) {
  const std::size_t n = a.rows();
  lower = Matrix(n, n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double s = a(i, j);
      for (std::size_t k = 0; k < j; ++k) s -= lower(i, k) * lower(j, k);
      if (i == j) {
        if (s <= 0.0 || !std::isfinite(s)) return false;
        lower(i, i) = std::sqrt(s);
      } else {
        lower(i, j) = s / lower(j, j);
      }
    }
  }
  return true;
}

}  // namespace

CholeskyResult robust_cholesky(const Matrix& cov, int max_tries) {
  const std::size_t n = cov.rows();
  if (n == 0 || cov.cols() != n)
    throw std::invalid_argument("robust_cholesky: cov must be a non-empty square matrix");
  // Largest magnitude entry: the symmetry test must be RELATIVE.  A
  // covariance quoted in large units (JPY notionals, unscaled variances)
  // is symmetric only to ~1e-16 relative, which an absolute 1e-12 test
  // rejects out of hand.
  double cmax = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      const double v = cov(i, j);
      // NaN/Inf would survive the jitter ladder (every pivot test fails)
      // and surface as a misleading "not factorisable" runtime_error.
      if (!std::isfinite(v))
        throw std::invalid_argument(
            "robust_cholesky: cov contains NaN or infinite entries (NaN "
            "policy: refuse)");
      cmax = std::max(cmax, std::abs(v));
    }
  const double sym_tol = 1e-12 * std::max(1.0, cmax);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (std::abs(cov(i, j) - cov(j, i)) > sym_tol)
        throw std::invalid_argument("robust_cholesky: cov must be symmetric");

  CholeskyResult res;
  if (try_cholesky(cov, res.lower)) return res;

  double base = 0.0;
  for (std::size_t i = 0; i < n; ++i) base += cov(i, i);
  base /= static_cast<double>(n);
  if (base <= 0.0) base = 1.0;

  double jitter = 1e-12 * base;
  for (int t = 0; t < max_tries; ++t, jitter *= 10.0) {
    Matrix a = cov;
    for (std::size_t i = 0; i < n; ++i) a(i, i) += jitter;
    if (try_cholesky(a, res.lower)) {
      res.jitter = jitter;
      res.jittered = true;
      std::ostringstream msg;
      msg << "covariance not positive definite; Cholesky computed with "
             "diagonal jitter "
          << jitter << " (singular/pegged factor block)";
      res.warning = msg.str();
      return res;
    }
  }
  throw std::runtime_error(
      "robust_cholesky: covariance matrix is not factorisable even with jitter");
}

}  // namespace fxvar
