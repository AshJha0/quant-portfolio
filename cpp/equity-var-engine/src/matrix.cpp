#include "eqvar/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace eqvar {

Matrix::Matrix(std::size_t rows, std::size_t cols, std::vector<double> data)
    : rows_(rows), cols_(cols), data_(std::move(data)) {
    if (data_.size() != rows * cols) {
        throw std::invalid_argument("Matrix: data size " + std::to_string(data_.size()) +
                                    " does not match " + std::to_string(rows) + "x" +
                                    std::to_string(cols));
    }
}

Matrix Matrix::identity(std::size_t n) {
    Matrix m(n, n);
    for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.0;
    return m;
}

double Matrix::max_abs_diff(const Matrix& a, const Matrix& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument("max_abs_diff: shape mismatch");
    }
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs(a.data()[i] - b.data()[i]));
    }
    return m;
}

std::vector<double> matvec(const Matrix& a, std::span<const double> x) {
    if (x.size() != a.cols()) {
        throw std::invalid_argument("matvec: vector size " + std::to_string(x.size()) +
                                    " does not match " + std::to_string(a.cols()) + " columns");
    }
    std::vector<double> y(a.rows(), 0.0);
    for (std::size_t i = 0; i < a.rows(); ++i) {
        const double* row = a.data() + i * a.cols();
        double acc = 0.0;
        for (std::size_t j = 0; j < a.cols(); ++j) acc += row[j] * x[j];
        y[i] = acc;
    }
    return y;
}

double quad_form(std::span<const double> w, const Matrix& s) {
    if (!s.square() || s.rows() != w.size()) {
        throw std::invalid_argument("quad_form: matrix shape does not match vector size " +
                                    std::to_string(w.size()));
    }
    double acc = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        const double* row = s.data() + i * s.cols();
        double inner = 0.0;
        for (std::size_t j = 0; j < w.size(); ++j) inner += row[j] * w[j];
        acc += w[i] * inner;
    }
    return acc;
}

Matrix covariance_from_vols(std::span<const double> vols, const Matrix& corr) {
    const std::size_t n = vols.size();
    if (!corr.square() || corr.rows() != n) {
        throw std::invalid_argument("covariance_from_vols: correlation shape does not match " +
                                    std::to_string(n) + " vols");
    }
    Matrix cov(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) cov(i, j) = vols[i] * vols[j] * corr(i, j);
    }
    return cov;
}

namespace {

/// Largest diagonal jitter, as a multiple of mean(diag), that still counts as
/// a rounding repair rather than a change of model.
constexpr double kMaxRelJitter = 1e-6;

/// Plain Cholesky attempt; returns false if a non-positive pivot is hit.
bool try_cholesky(const Matrix& a, Matrix& l) {
    const std::size_t n = a.rows();
    l = Matrix(n, n);
    for (std::size_t j = 0; j < n; ++j) {
        double diag = a(j, j);
        for (std::size_t k = 0; k < j; ++k) diag -= l(j, k) * l(j, k);
        if (!(diag > 0.0)) return false;  // also catches NaN
        const double ljj = std::sqrt(diag);
        l(j, j) = ljj;
        for (std::size_t i = j + 1; i < n; ++i) {
            double s = a(i, j);
            for (std::size_t k = 0; k < j; ++k) s -= l(i, k) * l(j, k);
            l(i, j) = s / ljj;
        }
    }
    return true;
}

}  // namespace

CholeskyResult cholesky(const Matrix& cov, double jitter, int max_tries) {
    const std::size_t n = cov.rows();
    if (!cov.square()) {
        throw std::invalid_argument("cholesky: covariance must be square and non-empty");
    }
    double amax = 0.0;
    for (std::size_t i = 0; i < cov.size(); ++i) {
        // A NaN/Inf entry would otherwise survive every jitter attempt (the
        // pivot test rejects NaN) and surface as a misleading "badly
        // indefinite" runtime_error; reject it here as the input error it is.
        if (!std::isfinite(cov.data()[i])) {
            throw std::invalid_argument(
                "cholesky: covariance contains NaN or infinite entries");
        }
        amax = std::max(amax, std::abs(cov.data()[i]));
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (std::abs(cov(i, j) - cov(j, i)) > 1e-12 * std::max(1.0, amax)) {
                throw std::invalid_argument("cholesky: covariance matrix must be symmetric");
            }
        }
    }
    CholeskyResult res;
    if (try_cholesky(cov, res.lower)) return res;

    // Diagonal-jitter fallback for singular / near-singular PSD matrices.
    double scale = 0.0;
    for (std::size_t i = 0; i < n; ++i) scale += cov(i, i);
    scale /= static_cast<double>(n);
    if (scale <= 0.0) scale = 1.0;
    double eps = jitter * scale;
    for (int t = 0; t < max_tries; ++t, eps *= 10.0) {
        Matrix a = cov;
        for (std::size_t i = 0; i < n; ++i) a(i, i) += eps;
        if (try_cholesky(a, res.lower)) {
            // The jitter is only legitimate while it is negligible against
            // the variances themselves.  A matrix that needs more than
            // kMaxRelJitter * mean(diag) to become positive definite is not
            // "numerically singular PSD" but materially indefinite: factoring
            // it would silently simulate a covariance that is not the one the
            // caller passed in.  Reject instead, and say so.
            if (eps > kMaxRelJitter * scale) {
                throw std::runtime_error(
                    "cholesky: covariance needed a diagonal jitter of " +
                    std::to_string(eps / scale) +
                    " x mean(diag) to factor, far above the " +
                    std::to_string(kMaxRelJitter) +
                    " tolerance; the matrix is materially indefinite. Repair it "
                    "(eigenvalue clipping / nearest-PSD projection) before use.");
            }
            res.jitter_added = eps;
            return res;
        }
    }
    throw std::runtime_error(
        "cholesky: failed even with jitter; covariance matrix is badly indefinite");
}

Matrix sample_covariance(const Matrix& returns) {
    const std::size_t t = returns.rows(), n = returns.cols();
    if (t < 2) {
        throw std::invalid_argument("sample_covariance: need at least 2 observations, got " +
                                    std::to_string(t));
    }
    std::vector<double> mu(n, 0.0);
    for (std::size_t r = 0; r < t; ++r)
        for (std::size_t c = 0; c < n; ++c) mu[c] += returns(r, c);
    for (double& m : mu) m /= static_cast<double>(t);

    Matrix cov(n, n);
    for (std::size_t r = 0; r < t; ++r) {
        for (std::size_t i = 0; i < n; ++i) {
            const double di = returns(r, i) - mu[i];
            for (std::size_t j = i; j < n; ++j) {
                cov(i, j) += di * (returns(r, j) - mu[j]);
            }
        }
    }
    const double denom = static_cast<double>(t - 1);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            cov(i, j) /= denom;
            cov(j, i) = cov(i, j);
        }
    }
    return cov;
}

Matrix ewma_covariance(const Matrix& returns, double lam) {
    if (!(lam > 0.0 && lam < 1.0)) {
        throw std::invalid_argument("ewma_covariance: decay lam must be in (0, 1), got " +
                                    std::to_string(lam));
    }
    Matrix cov = sample_covariance(returns);  // validates T >= 2
    const std::size_t t = returns.rows(), n = returns.cols();
    for (std::size_t r = 0; r < t; ++r) {
        for (std::size_t i = 0; i < n; ++i) {
            const double ri = returns(r, i);
            for (std::size_t j = 0; j < n; ++j) {
                cov(i, j) = lam * cov(i, j) + (1.0 - lam) * ri * returns(r, j);
            }
        }
    }
    return cov;
}

}  // namespace eqvar
