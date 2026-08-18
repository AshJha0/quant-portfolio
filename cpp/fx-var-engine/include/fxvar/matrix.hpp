// Dense matrix and robust Cholesky factorisation.
//
// A deliberately small linear-algebra layer: the engine needs dense
// row-major storage, products, quadratic forms, and Cholesky with jitter
// escalation (singular covariances are a legitimate FX input: two
// currencies pegged to the same anchor are perfectly correlated).  No
// external BLAS - the problem sizes (tens of factors) never justify the
// dependency, and -O2 auto-vectorises the streaming loops that matter
// (see bench/bench_main.cpp).
//
// Covariance estimators live in returns.hpp (they are factor-labelled).
// Mirrors: python/fx/03-var-es-engine fx_var.monte_carlo_var.robust_cholesky.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace fxvar {

/// Dense row-major matrix of double.
class Matrix {
 public:
  Matrix() = default;

  /// rows x cols matrix filled with `fill`.
  Matrix(std::size_t rows, std::size_t cols, double fill = 0.0)
      : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

  /// Build from nested rows (throws std::invalid_argument on ragged input).
  static Matrix from_rows(const std::vector<std::vector<double>>& rows);

  /// n x n identity.
  static Matrix identity(std::size_t n);

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  double& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
  double operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

  /// Pointer to the start of row i (contiguous, cols() doubles).
  const double* row(std::size_t i) const { return data_.data() + i * cols_; }
  double* row(std::size_t i) { return data_.data() + i * cols_; }

  Matrix transpose() const;

  /// True if any element is NaN (the engine's NaN policy is refuse).
  bool has_nan() const;

  const std::vector<double>& data() const { return data_; }

 private:
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<double> data_;
};

/// C = A B (throws std::invalid_argument on dimension mismatch).
Matrix matmul(const Matrix& a, const Matrix& b);

/// y = A x (throws on dimension mismatch).
std::vector<double> matvec(const Matrix& a, const std::vector<double>& x);

/// w' A w for square A (throws on dimension mismatch).
double quad_form(const std::vector<double>& w, const Matrix& a);

/// Result of `robust_cholesky`: the lower factor plus a diagnostic of the
/// jitter that was needed.  The engine surfaces (never hides) numerical
/// fallbacks - a jittered factorisation on a "clean" covariance is a data
/// quality signal the desk should see.
struct CholeskyResult {
  Matrix lower;           ///< L with L L^T = cov (+ jitter I).
  double jitter = 0.0;    ///< Diagonal jitter actually added (0 if none).
  bool jittered = false;  ///< True if any jitter was required.
  std::string warning;    ///< Human-readable diagnostic when jittered.
};

/// Cholesky factorisation with escalating diagonal jitter.
///
/// A covariance containing pegged currencies is routinely singular or
/// numerically indefinite (near-zero-vol factors, perfectly correlated
/// pegs to the same anchor).  On failure, jitter 1e-12 * mean(diag) is
/// added and escalated x10 up to `max_tries` times; the result records
/// the jitter actually used.  Throws std::invalid_argument if `cov` is
/// not square/symmetric and std::runtime_error if it cannot be
/// factorised at maximum jitter.
CholeskyResult robust_cholesky(const Matrix& cov, int max_tries = 8);

}  // namespace fxvar
