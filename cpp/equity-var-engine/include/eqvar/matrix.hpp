// eqvar/matrix.hpp — minimal dense linear algebra for the VaR engine.
//
// A deliberately small surface: row-major dense Matrix over std::vector<double>,
// Cholesky with a diagonal-jitter fallback for near-singular PSD matrices,
// matrix-vector product / quadratic form, and covariance estimators (sample +
// RiskMetrics EWMA) from a (T x n) returns panel.  Semantics mirror the Python
// reference (eq_var.parametric_var.sample_covariance / ewma_covariance and
// eq_var.monte_carlo_var.safe_cholesky) so cross-language golden tests hold
// to ~1e-12.
//
// Conventions: covariances are daily, in the same units as the returns panel;
// invalid inputs throw std::invalid_argument with an informative message.

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace eqvar {

/// Row-major dense matrix of doubles.  Storage is a single contiguous
/// std::vector so hot paths (matvec, Cholesky, simulation) are cache-friendly
/// and allocation-free once constructed.
class Matrix {
public:
    Matrix() = default;

    /// rows x cols matrix, zero-initialised.
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

    /// rows x cols matrix with every entry set to `fill`.
    Matrix(std::size_t rows, std::size_t cols, double fill)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    /// rows x cols matrix from row-major data (size must be rows*cols).
    Matrix(std::size_t rows, std::size_t cols, std::vector<double> data);

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool square() const noexcept { return rows_ == cols_ && rows_ > 0; }

    [[nodiscard]] double& operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * cols_ + j];
    }
    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * cols_ + j];
    }

    /// Raw row-major contiguous storage (no-alloc hot paths in the MC engine).
    [[nodiscard]] double* data() noexcept { return data_.data(); }
    [[nodiscard]] const double* data() const noexcept { return data_.data(); }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }

    /// n x n identity.
    static Matrix identity(std::size_t n);

    /// max_ij |a_ij - b_ij|; throws std::invalid_argument on shape mismatch.
    static double max_abs_diff(const Matrix& a, const Matrix& b);

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

/// y = A x.  Throws std::invalid_argument on dimension mismatch.
[[nodiscard]] std::vector<double> matvec(const Matrix& a, std::span<const double> x);

/// Quadratic form w' S w for square S.  Throws on dimension mismatch.
[[nodiscard]] double quad_form(std::span<const double> w, const Matrix& s);

/// Covariance from per-asset vols and a correlation matrix:
/// cov_ij = vol_i * vol_j * corr_ij.  Convenience for tests / benchmarks.
[[nodiscard]] Matrix covariance_from_vols(std::span<const double> vols, const Matrix& corr);

/// Result of a (possibly jittered) Cholesky factorisation.
struct CholeskyResult {
    Matrix lower;              ///< L with L L^T ~= input (+ jitter * I).
    double jitter_added = 0.0; ///< diagonal perturbation actually used (0 if clean).
};

/// Lower-triangular Cholesky factor with a diagonal-jitter fallback.
///
/// For exactly or numerically singular PSD matrices (perfectly correlated
/// factors, zero-variance factors) plain Cholesky fails; we add
/// `jitter * mean(diag)` to the diagonal, escalating x10 up to `max_tries`
/// times — the perturbation is tiny relative to the variances, so simulated
/// moments are unchanged to within Monte Carlo noise.  Mirrors
/// eq_var.monte_carlo_var.safe_cholesky.
///
/// The escalation is capped at 1e-6 * mean(diag): a matrix that needs more
/// than that is materially indefinite (a genuine negative eigenvalue, not
/// rounding), and factoring it would silently simulate a *different*
/// covariance from the one supplied, understating or overstating risk with
/// no diagnostic.  Such matrices throw std::runtime_error; repair them
/// upstream (eigenvalue clipping / nearest-PSD projection) instead.
/// `jitter_added` reports the perturbation actually used so callers can log
/// or alert on it.
///
/// Throws std::invalid_argument if `cov` is not square, not symmetric, or
/// contains NaN/Inf entries, and std::runtime_error if the matrix is badly
/// indefinite even after the jitter escalation (a genuinely unusable
/// covariance, e.g. a correlation > 1 typo, rather than a bad argument).
[[nodiscard]] CholeskyResult cholesky(const Matrix& cov, double jitter = 1e-10,
                                      int max_tries = 12);

/// Unbiased (ddof = 1) sample covariance of a (T x n) returns panel.
/// Throws std::invalid_argument if T < 2.
[[nodiscard]] Matrix sample_covariance(const Matrix& returns);

/// RiskMetrics EWMA covariance forecast for the day after the sample:
/// Sigma <- lam * Sigma + (1 - lam) * r_t r_t' over all rows, seeded with the
/// sample covariance.  Zero-mean returns assumed (standard at daily horizon).
/// Mirrors eq_var.parametric_var.ewma_covariance exactly.
[[nodiscard]] Matrix ewma_covariance(const Matrix& returns, double lam = 0.94);

}  // namespace eqvar
