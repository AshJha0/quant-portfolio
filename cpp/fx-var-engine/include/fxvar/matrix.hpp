// Dense matrix and robust Cholesky factorisation.
//
// A small linear-algebra layer: the engine needs dense
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

/// Symmetric eigen-decomposition by the cyclic Jacobi method. `values` are
/// ascending; column k of `vectors` is the unit eigenvector for values[k].
struct EigenResult {
  std::vector<double> values;
  Matrix vectors;
};
EigenResult symmetric_eigen(const Matrix& a);

/// PSD repair used when plain Cholesky plus the capped jitter ladder fails.
///   None   : no repair; a materially indefinite matrix throws.
///   Clip   : eigenvalues clipped to max(lam, 1e-10 mean(diag)), rebuilt,
///            rescaled D^{1/2} to keep the original variances.
///   Higham : Higham (2002) alternating projections with Dykstra correction
///            on the correlation matrix (unit diagonal + PSD projections),
///            relative Frobenius tolerance 1e-12, at most 1000 iterations,
///            rescaled by the original vols.
enum class Repair { kNone, kClip, kHigham };

/// Repaired copy of `cov` (Repair::kNone returns the validated input).
Matrix repair_covariance(const Matrix& cov, Repair repair);

/// Result of a Cholesky factorisation.
struct CholeskyResult {
  Matrix lower;           ///< L with L L^T = factored matrix (+ jitter I).
  double jitter = 0.0;    ///< Diagonal jitter actually added (0 if none).
  bool jittered = false;  ///< True if any jitter was required (a warning).
  bool repaired = false;  ///< True if a PSD repair changed the matrix.
  std::string warning;    ///< Human-readable diagnostic when jittered/repaired.
};

/// Lower-triangular Cholesky factor.
/// Symmetry |A - A'| <= 1e-10 max|A| (relative); NaN/Inf or a negative
/// diagonal throws std::invalid_argument. Plain Cholesky first; on failure
/// add eps I, eps = 1e-12 mean(diag), x10 per try, capped at 1e-6 mean(diag).
/// Past the cap: Repair::kNone throws std::runtime_error naming the most
/// negative eigenvalue; otherwise repair and factor the repaired matrix.
CholeskyResult cholesky(const Matrix& cov, Repair repair = Repair::kNone);

/// Old entry point. It used to escalate the jitter without a cap; it now
/// forwards to cholesky(cov, Repair::kNone) and `max_tries` is ignored.
[[deprecated("use cholesky(cov, Repair). Removed next release.")]]
CholeskyResult robust_cholesky(const Matrix& cov, int max_tries = 8);

}  // namespace fxvar
