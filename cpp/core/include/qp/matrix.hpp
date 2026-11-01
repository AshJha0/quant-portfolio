// qp/matrix.hpp: dense row-major matrix, Cholesky, Jacobi eigen and PSD
// repair, shared by the two VaR engines.
//
// qp::Matrix holds the storage and the element access. Each engine derives
// its own Matrix from it (eqvar::Matrix, fxvar::Matrix) to keep its public
// API, so the algorithms below are templates over the matrix type M. M must
// be a qp::Matrix or derive from one and be constructible as M(rows, cols).
//
// Cholesky policy: plain factorisation first; on a non-positive pivot a
// diagonal jitter ladder of 1e-12 .. 1e-6 x mean(diag), x10 per try; past
// the cap the matrix is materially indefinite and the caller must ask for a
// repair (eigenvalue clip, or Higham's nearest correlation matrix).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qp/config.hpp"
#include "qp/simd.hpp"

namespace qp {

class Matrix {
public:
    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    /// Takes `data` (row-major, rows * cols values); throws
    /// std::invalid_argument when the size does not match.
    Matrix(std::size_t rows, std::size_t cols, std::vector<double> data)
        : rows_(rows), cols_(cols), data_(std::move(data)) {
        if (data_.size() != rows * cols) {
            throw std::invalid_argument("Matrix: data size " + std::to_string(data_.size()) +
                                        " does not match " + std::to_string(rows) + "x" +
                                        std::to_string(cols));
        }
    }

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool square() const noexcept { return rows_ == cols_ && rows_ > 0; }

    [[nodiscard]] double& operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * cols_ + j];
    }
    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * cols_ + j];
    }

    [[nodiscard]] const double* row(std::size_t i) const noexcept {
        return data_.data() + i * cols_;
    }
    [[nodiscard]] double* row(std::size_t i) noexcept { return data_.data() + i * cols_; }

    /// Row-major storage as a contiguous span.
    [[nodiscard]] std::span<const double> values() const noexcept { return data_; }
    [[nodiscard]] std::span<double> values() noexcept { return data_; }

protected:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

/// Plain Cholesky on raw row-major n x n storage: L L^T = A with L lower
/// triangular. Writes all n * n entries of `l`. Returns false at the first
/// non-positive (or NaN) pivot, leaving `l` partly written.
[[nodiscard]] inline bool cholesky_raw(const double* a, std::size_t n, double* l) noexcept
    QP_PRE(n == 0 || (a != nullptr && l != nullptr)) {
    std::fill(l, l + n * n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        const double* lj = l + j * n;
        double diag = a[j * n + j];
        for (std::size_t k = 0; k < j; ++k) diag -= lj[k] * lj[k];
        if (!(diag > 0.0)) return false;  // also catches NaN
        const double ljj = std::sqrt(diag);
        l[j * n + j] = ljj;
        for (std::size_t i = j + 1; i < n; ++i) {
            const double* li = l + i * n;
            double s = a[i * n + j];
            for (std::size_t k = 0; k < j; ++k) s -= li[k] * lj[k];
            l[i * n + j] = s / ljj;
        }
    }
    return true;
}

/// y = A x for row-major A (rows x cols). Rows are processed four at a
/// time with independent accumulators; each row is summed left to right,
/// so y[i] is the same double as the plain loop. In a C++26 build with
/// std::simd (qp/simd.hpp) the rows first go W at a time through the
/// vector path, which keeps the same per-row order and gives the same bits.
inline void matvec_rows(const double* d, std::size_t rows, std::size_t cols, const double* xv,
                        double* y) noexcept
    QP_PRE(rows == 0 || (d != nullptr && y != nullptr && (cols == 0 || xv != nullptr))) {
    std::size_t i = 0;
#if QP_HAS_SIMD
    i = simd_detail::matvec_rows_simd(d, rows, cols, xv, y);
#endif
    // Four rows at a time: four independent accumulators for instruction-
    // level parallelism, each summed in the original left-to-right order.
    for (; i + 4 <= rows; i += 4) {
        const double* r0 = d + i * cols;
        const double* r1 = r0 + cols;
        const double* r2 = r1 + cols;
        const double* r3 = r2 + cols;
        double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
        for (std::size_t j = 0; j < cols; ++j) {
            const double xj = xv[j];
            a0 += r0[j] * xj;
            a1 += r1[j] * xj;
            a2 += r2[j] * xj;
            a3 += r3[j] * xj;
        }
        y[i] = a0;
        y[i + 1] = a1;
        y[i + 2] = a2;
        y[i + 3] = a3;
    }
    for (; i < rows; ++i) {
        const double* row = d + i * cols;
        double acc = 0.0;
        for (std::size_t j = 0; j < cols; ++j) acc += row[j] * xv[j];
        y[i] = acc;
    }
}

namespace linalg {

/// Repair applied when the jitter ladder fails.
enum class RepairKind { None, Clip, Higham };

/// Output of cholesky(): the factor and what it took to get it.
template <class M>
struct CholeskyOut {
    M lower;                ///< L with L L^T = factored matrix (+ jitter * I).
    double jitter = 0.0;    ///< diagonal jitter added (0 if none).
    bool repaired = false;  ///< true if a PSD repair changed the matrix.
    std::string warning;    ///< note when jitter or a repair was used.
};

/// Plain Cholesky of `a` into a fresh `l`; false at a non-positive pivot.
template <class M>
[[nodiscard]] bool cholesky_plain(const M& a, M& l) {
    const std::size_t n = a.rows();
    l = M(n, n);
    return cholesky_raw(a.values().data(), n, l.values().data());
}

/// Validates shape, finiteness, symmetry and the diagonal; returns
/// mean(diag). Throws std::invalid_argument with `who` in the message.
template <class M>
double validate_cov(const M& cov, const char* who) {
    if (!(cov.rows() == cov.cols() && cov.rows() > 0)) {
        throw std::invalid_argument(std::string(who) + ": covariance must be square and non-empty");
    }
    const std::size_t n = cov.rows();
    double amax = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isfinite(cov(i, j))) {
                throw std::invalid_argument(std::string(who) +
                                            ": covariance contains NaN or infinite entries");
            }
            amax = std::max(amax, std::abs(cov(i, j)));
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (cov(i, i) < 0.0) {
            throw std::invalid_argument(std::string(who) + ": negative variance on the diagonal");
        }
        for (std::size_t j = i + 1; j < n; ++j) {
            if (std::abs(cov(i, j) - cov(j, i)) > 1e-10 * amax) {
                throw std::invalid_argument(std::string(who) +
                                            ": covariance matrix must be symmetric");
            }
        }
    }
    double scale = 0.0;
    for (std::size_t i = 0; i < n; ++i) scale += cov(i, i);
    return scale / static_cast<double>(n);
}

/// Plain Cholesky, then the capped jitter ladder. Returns false past the cap.
template <class M>
[[nodiscard]] bool factor_with_ladder(const M& a, double scale, CholeskyOut<M>& res) {
    constexpr double kJitterStart = 1e-12;
    constexpr double kMaxRelJitter = 1e-6;
    if (cholesky_plain(a, res.lower)) return true;
    const std::size_t n = a.rows();
    for (double eps = kJitterStart * scale; eps <= kMaxRelJitter * scale * (1.0 + 1e-9);
         eps *= 10.0) {
        M b = a;
        for (std::size_t i = 0; i < n; ++i) b(i, i) += eps;
        if (cholesky_plain(b, res.lower)) {
            res.jitter = eps;
            res.warning += "Cholesky needed diagonal jitter " + std::to_string(eps / scale) +
                           " x mean(diag). ";
            return true;
        }
    }
    return false;
}

/// Cyclic Jacobi eigen-decomposition of a symmetric matrix. Eigenvalues
/// ascending in `values`, eigenvectors in the columns of `vectors`.
template <class M>
void symmetric_eigen(const M& a_in, std::vector<double>& values, M& vectors) {
    if (!(a_in.rows() == a_in.cols() && a_in.rows() > 0)) {
        throw std::invalid_argument("symmetric_eigen: matrix must be square and non-empty");
    }
    const std::size_t n = a_in.rows();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isfinite(a_in(i, j))) {
                throw std::invalid_argument("symmetric_eigen: matrix contains NaN or inf");
            }
        }
    }
    M a = a_in;
    M v(n, n);
    for (std::size_t i = 0; i < n; ++i) v(i, i) = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        double diag = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            diag += a(i, i) * a(i, i);
            for (std::size_t j = i + 1; j < n; ++j) off += a(i, j) * a(i, j);
        }
        if (off == 0.0 || off <= 1e-32 * diag) break;
        for (std::size_t p = 0; p + 1 < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const double apq = a(p, q);
                if (apq == 0.0) continue;
                // Jacobi rotation that zeroes a(p, q) (Golub and Van Loan, 8.5.2).
                const double theta = (a(q, q) - a(p, p)) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (std::size_t k = 0; k < n; ++k) {
                    const double akp = a(k, p);
                    const double akq = a(k, q);
                    a(k, p) = c * akp - s * akq;
                    a(k, q) = s * akp + c * akq;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double apk = a(p, k);
                    const double aqk = a(q, k);
                    a(p, k) = c * apk - s * aqk;
                    a(q, k) = s * apk + c * aqk;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double vkp = v(k, p);
                    const double vkq = v(k, q);
                    v(k, p) = c * vkp - s * vkq;
                    v(k, q) = s * vkp + c * vkq;
                }
            }
        }
    }
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&a](std::size_t x, std::size_t y) { return a(x, x) < a(y, y); });
    values.resize(n);
    vectors = M(n, n);
    for (std::size_t k = 0; k < n; ++k) {
        values[k] = a(order[k], order[k]);
        for (std::size_t i = 0; i < n; ++i) vectors(i, k) = v(i, order[k]);
    }
}

/// V diag(lam) V^T.
template <class M>
[[nodiscard]] M rebuild(const M& v, const std::vector<double>& lam) {
    const std::size_t n = v.rows();
    M out(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) s += v(i, k) * lam[k] * v(j, k);
            out(i, j) = s;
            out(j, i) = s;
        }
    }
    return out;
}

/// Eigenvalues floored at `floor`, matrix rebuilt.
template <class M>
[[nodiscard]] M psd_project(const M& a, double floor) {
    std::vector<double> values;
    M vectors;
    symmetric_eigen(a, values, vectors);
    for (double& l : values) l = std::max(l, floor);
    return rebuild(vectors, values);
}

template <class M>
[[nodiscard]] double frobenius_diff(const M& a, const M& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            s += (a(i, j) - b(i, j)) * (a(i, j) - b(i, j));
        }
    }
    return std::sqrt(s);
}

template <class M>
[[nodiscard]] double frobenius(const M& a) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) s += a(i, j) * a(i, j);
    }
    return std::sqrt(s);
}

/// Nearest PSD covariance. Clip floors the eigenvalues at 1e-10 mean(diag)
/// and restores the variances; Higham (2002) finds the nearest correlation
/// matrix by alternating projections with Dykstra's correction and rescales.
template <class M>
[[nodiscard]] M repair_covariance(const M& cov, RepairKind repair) {
    const double scale = validate_cov(cov, "repair_covariance");
    const std::size_t n = cov.rows();
    if (repair == RepairKind::None) return cov;
    if (repair == RepairKind::Clip) {
        M b = psd_project(cov, 1e-10 * scale);
        // Restore the original variances: B_ij sqrt(A_ii / B_ii) sqrt(A_jj / B_jj).
        std::vector<double> d(n);
        for (std::size_t i = 0; i < n; ++i) d[i] = std::sqrt(cov(i, i) / b(i, i));
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) b(i, j) *= d[i] * d[j];
        }
        return b;
    }
    std::vector<double> vol(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!(cov(i, i) > 0.0)) {
            throw std::invalid_argument(
                "repair_covariance: Higham repair needs strictly positive variances");
        }
        vol[i] = std::sqrt(cov(i, i));
    }
    M y(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) y(i, j) = cov(i, j) / (vol[i] * vol[j]);
    }
    M ds(n, n);
    bool converged = false;
    for (int it = 0; it < 1000 && !converged; ++it) {
        M r = y;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) r(i, j) -= ds(i, j);
        }
        M x = psd_project(r, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) ds(i, j) = x(i, j) - r(i, j);
        }
        for (std::size_t i = 0; i < n; ++i) x(i, i) = 1.0;
        converged = frobenius_diff(x, y) <= 1e-12 * frobenius(x);
        y = std::move(x);
    }
    if (!converged) {
        throw std::runtime_error(
            "repair_covariance: Higham projection did not converge in 1000 iterations");
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) y(i, j) *= vol[i] * vol[j];
    }
    return y;
}

/// Validated Cholesky with the jitter ladder and an optional repair.
/// `repair_hint` names the engine's repair options in the error message.
/// Throws std::invalid_argument for bad input and std::runtime_error when
/// the matrix cannot be factored.
template <class M>
[[nodiscard]] CholeskyOut<M> cholesky(const M& cov, RepairKind repair, const char* repair_hint) {
    const double scale = validate_cov(cov, "cholesky");
    const std::size_t n = cov.rows();
    CholeskyOut<M> res;
    if (scale == 0.0) {
        // All variances zero: only the zero matrix is PSD, and L = 0 factors it.
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                if (cov(i, j) != 0.0) {
                    throw std::runtime_error("cholesky: zero variances with non-zero covariances");
                }
            }
        }
        res.lower = M(n, n);
        return res;
    }
    if (factor_with_ladder(cov, scale, res)) return res;
    if (repair == RepairKind::None) {
        std::vector<double> values;
        M vectors;
        symmetric_eigen(cov, values, vectors);
        throw std::runtime_error(
            "cholesky: covariance is materially indefinite (most negative eigenvalue " +
            std::to_string(values.front()) +
            "; jitter cap 1e-6 x mean(diag) exceeded). Repair it (" + repair_hint +
            ") before use.");
    }
    const M fixed = repair_covariance(cov, repair);
    res = CholeskyOut<M>{};
    res.repaired = true;
    res.warning = std::string("covariance repaired (") +
                  (repair == RepairKind::Clip ? "eigenvalue clip" : "Higham") + "). ";
    if (!factor_with_ladder(fixed, scale, res)) {
        throw std::runtime_error("cholesky: repaired covariance still does not factor");
    }
    return res;
}

}  // namespace linalg

}  // namespace qp
