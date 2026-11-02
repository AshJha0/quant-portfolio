#include "eqvar/matrix.hpp"

#include "eqvar/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace eqvar {

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

namespace {

void require_finite_span(std::span<const double> x, const char* who) {
    for (double v : x) {
        if (!std::isfinite(v))
            throw std::invalid_argument(std::string(who) + ": input contains NaN or inf");
    }
}

void require_finite_matrix(const Matrix& a, const char* who) {
    require_finite_span(std::span<const double>(a.data(), a.size()), who);
}

}  // namespace

std::vector<double> matvec(const Matrix& a, std::span<const double> x) {
    std::vector<double> y(x.size() == a.cols() ? a.rows() : 0);
    value_or_throw(try_matvec(a, x, y));
    return y;
}

double quad_form(std::span<const double> w, const Matrix& s) {
    return value_or_throw(try_quad_form(w, s));
}

Matrix covariance_from_vols(std::span<const double> vols, const Matrix& corr) {
    const std::size_t n = vols.size();
    if (!corr.square() || corr.rows() != n) {
        throw std::invalid_argument("covariance_from_vols: correlation shape does not match " +
                                    std::to_string(n) + " vols");
    }
    require_finite_span(vols, "covariance_from_vols");
    require_finite_matrix(corr, "covariance_from_vols");
    for (double v : vols) {
        if (v < 0.0) throw std::invalid_argument("covariance_from_vols: vols must be >= 0");
    }
    Matrix cov(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) cov(i, j) = vols[i] * vols[j] * corr(i, j);
    }
    return cov;
}

namespace {

qp::linalg::RepairKind kind(Repair r) noexcept {
    switch (r) {
        case Repair::Clip:
            return qp::linalg::RepairKind::Clip;
        case Repair::Higham:
            return qp::linalg::RepairKind::Higham;
        case Repair::None:
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
    auto out = qp::linalg::cholesky(cov, kind(repair), "Repair::Clip or Repair::Higham");
    CholeskyResult res;
    res.lower = std::move(out.lower);
    res.jitter_added = out.jitter;
    res.repaired = out.repaired;
    res.warning = std::move(out.warning);
    return res;
}

CholeskyResult cholesky(const Matrix& cov, double /*jitter*/, int /*max_tries*/) {
    return cholesky(cov, Repair::None);
}

Matrix sample_covariance(const Matrix& returns) {
    const std::size_t t = returns.rows(), n = returns.cols();
    if (t < 2) {
        throw std::invalid_argument("sample_covariance: need at least 2 observations, got " +
                                    std::to_string(t));
    }
    for (std::size_t i = 0; i < returns.size(); ++i) {
        if (!std::isfinite(returns.data()[i])) {
            throw std::invalid_argument("sample_covariance: returns contain NaN or inf");
        }
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
    const auto denom = static_cast<double>(t - 1);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            cov(i, j) /= denom;
            cov(j, i) = cov(i, j);
        }
    }
    return cov;
}

Matrix ewma_covariance(const Matrix& returns, double lam, std::size_t seed_window) {
    if (!(lam > 0.0 && lam < 1.0)) {
        throw std::invalid_argument("ewma_covariance: decay lam must be in (0, 1), got " +
                                    std::to_string(lam));
    }
    if (seed_window == 0) {
        throw std::invalid_argument("ewma_covariance: seed_window must be a positive integer");
    }
    (void)sample_covariance(returns);  // validates T >= 2 and finiteness
    const std::size_t t = returns.rows(), n = returns.cols();
    // Seed: zero-mean second moment over the first min(seed_window, T) rows
    // (convention S8), same summation order as head.T @ head / m in Python.
    const std::size_t m = seed_window < t ? seed_window : t;
    Matrix cov(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (std::size_t r = 0; r < m; ++r) acc += returns(r, i) * returns(r, j);
            cov(i, j) = acc / static_cast<double>(m);
        }
    }
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
