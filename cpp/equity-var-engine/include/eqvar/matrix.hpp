// eqvar/matrix.hpp: minimal dense linear algebra for the VaR engine.
//
// A small surface: row-major dense Matrix over std::vector<double>,
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
#include <string>
#include <vector>

#include "qp/matrix.hpp"

namespace eqvar {

/// Row-major dense matrix of doubles: qp::Matrix (cpp/core) plus the
/// eqvar accessors. Storage is a single contiguous std::vector so hot paths
/// (matvec, Cholesky, simulation) are cache-friendly and allocation-free
/// once constructed. Constructors: (), (rows, cols[, fill]) zero-filled by
/// default, and (rows, cols, row-major data) which throws
/// std::invalid_argument when data.size() != rows * cols.
class Matrix : public qp::Matrix {
public:
    using qp::Matrix::Matrix;

    /// Raw row-major contiguous storage (no-alloc hot paths in the MC engine).
    [[nodiscard]] double* data() noexcept { return data_.data(); }
    [[nodiscard]] const double* data() const noexcept { return data_.data(); }

    /// n x n identity.
    static Matrix identity(std::size_t n);

    /// max_ij |a_ij - b_ij|; throws std::invalid_argument on shape mismatch.
    static double max_abs_diff(const Matrix& a, const Matrix& b);
};

/// y = A x.  Throws std::invalid_argument on dimension mismatch.
[[nodiscard]] std::vector<double> matvec(const Matrix& a, std::span<const double> x);

/// Quadratic form w' S w for square S.  Throws on dimension mismatch.
[[nodiscard]] double quad_form(std::span<const double> w, const Matrix& s);

/// Covariance from per-asset vols and a correlation matrix:
/// cov_ij = vol_i * vol_j * corr_ij.  Convenience for tests / benchmarks.
[[nodiscard]] Matrix covariance_from_vols(std::span<const double> vols, const Matrix& corr);

/// Symmetric eigen-decomposition by the cyclic Jacobi method. `values` are
/// ascending; column k of `vectors` is the unit eigenvector for values[k].
/// Intended for the small factor matrices used here (O(n^3) per sweep).
struct EigenResult {
    std::vector<double> values;
    Matrix vectors;
};
[[nodiscard]] EigenResult symmetric_eigen(const Matrix& a);

/// PSD repair applied before factorisation when plain Cholesky plus the
/// capped jitter ladder fails.
///   None   : no repair; a materially indefinite matrix throws.
///   Clip   : eigenvalues clipped to max(lam, 1e-10 mean(diag)), rebuilt, then
///            rescaled D^{1/2} so the original variances are kept.
///   Higham : Higham (2002) alternating projections with Dykstra correction
///            on the correlation matrix (unit diagonal + PSD projections),
///            relative Frobenius tolerance 1e-12 on successive iterates,
///            at most 1000 iterations, rescaled by the original vols.
enum class Repair { None, Clip, Higham };

/// Repaired copy of `cov` (Repair::None returns the validated input). Same
/// input validation as cholesky(). Higham needs a strictly positive diagonal
/// and throws std::runtime_error if it does not converge.
[[nodiscard]] Matrix repair_covariance(const Matrix& cov, Repair repair);

/// Result of a Cholesky factorisation.
struct CholeskyResult {
    Matrix lower;               ///< L with L L^T = factored matrix (+ jitter_added * I).
    double jitter_added = 0.0;  ///< diagonal jitter used (0 if none). Non-zero is a warning.
    bool repaired = false;      ///< true if a PSD repair changed the matrix.
    std::string warning;        ///< human-readable note when jitter or repair was used.
};

/// Lower-triangular Cholesky factor.
///
/// Symmetry: |A - A'| <= 1e-10 max|A| (relative), else std::invalid_argument.
/// NaN/Inf entries or a negative diagonal: std::invalid_argument.
/// Plain Cholesky first. On failure add eps I with eps = 1e-12 mean(diag),
/// x10 per try, capped at 1e-6 mean(diag); any jitter is reported in
/// `jitter_added` and `warning`. Past the cap: with Repair::None throw
/// std::runtime_error naming the most negative eigenvalue; otherwise repair
/// (see Repair) and factor the repaired matrix with the same ladder.
[[nodiscard]] CholeskyResult cholesky(const Matrix& cov, Repair repair = Repair::None);

/// Unbiased (ddof = 1) sample covariance of a (T x n) returns panel.
/// Throws std::invalid_argument if T < 2.
[[nodiscard]] Matrix sample_covariance(const Matrix& returns);

/// RiskMetrics EWMA covariance forecast for the day after the sample:
/// Sigma <- lam * Sigma + (1 - lam) * r_t r_t' over all rows, seeded with the
/// zero-mean second moment mean(r r') of the first min(seed_window, T) rows
/// (convention S8), so the seed uses no later data.  Zero-mean returns assumed
/// (standard at daily horizon).  Throws std::invalid_argument if T < 2, the
/// panel is not finite, lam is outside (0, 1) or seed_window == 0.
/// Mirrors eq_var.parametric_var.ewma_covariance exactly.
[[nodiscard]] Matrix ewma_covariance(const Matrix& returns, double lam = 0.94,
                                     std::size_t seed_window = 30);

}  // namespace eqvar
