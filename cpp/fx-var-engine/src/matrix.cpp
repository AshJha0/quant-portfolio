#include "fxvar/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

#include "fxvar/kernels.hpp"

namespace fxvar {

Matrix Matrix::from_rows(const std::vector<std::vector<double>>& rows) {
  const std::size_t r = rows.size();
  const std::size_t c = r ? rows.front().size() : 0;
  Matrix m(r, c);
  for (std::size_t i = 0; i < r; ++i) {
    if (rows[i].size() != c) throw std::invalid_argument("Matrix::from_rows: ragged rows");
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
  if (a.cols() != b.rows()) throw std::invalid_argument("matmul: inner dimensions do not match");
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
  if (a.cols() != x.size()) throw std::invalid_argument("matvec: dimension mismatch");
  std::vector<double> y(a.rows(), 0.0);
  value_or_throw(try_matvec(a, x, y));
  return y;
}

double quad_form(const std::vector<double>& w, const Matrix& a) {
  return value_or_throw(try_quad_form(w, a));
}

namespace {

qp::linalg::RepairKind kind(Repair r) noexcept {
  switch (r) {
    case Repair::kClip:
      return qp::linalg::RepairKind::Clip;
    case Repair::kHigham:
      return qp::linalg::RepairKind::Higham;
    case Repair::kNone:
      break;
  }
  return qp::linalg::RepairKind::None;
}

}  // namespace

EigenResult symmetric_eigen(const Matrix& a) {
  EigenResult res;
  qp::linalg::symmetric_eigen(a, res.values, res.vectors);
  return res;
}

Matrix repair_covariance(const Matrix& cov, Repair repair) {
  return qp::linalg::repair_covariance(cov, kind(repair));
}

CholeskyResult cholesky(const Matrix& cov, Repair repair) {
  auto out = qp::linalg::cholesky(cov, kind(repair), "Repair::kClip or Repair::kHigham");
  CholeskyResult res;
  res.lower = std::move(out.lower);
  res.jitter = out.jitter;
  res.jittered = out.jitter != 0.0;
  res.repaired = out.repaired;
  res.warning = std::move(out.warning);
  return res;
}

}  // namespace fxvar
