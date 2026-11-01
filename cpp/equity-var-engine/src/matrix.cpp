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

namespace {

void require_finite_span(std::span<const double> x, const char* who) {
    for (double v : x) {
        if (!std::isfinite(v)) throw std::invalid_argument(std::string(who) + ": input contains NaN or inf");
    }
}

void require_finite_matrix(const Matrix& a, const char* who) {
    require_finite_span(std::span<const double>(a.data(), a.size()), who);
}

}  // namespace

std::vector<double> matvec(const Matrix& a, std::span<const double> x) {
    if (x.size() != a.cols()) {
        throw std::invalid_argument("matvec: vector size " + std::to_string(x.size()) +
                                    " does not match " + std::to_string(a.cols()) + " columns");
    }
    require_finite_span(x, "matvec");
    require_finite_matrix(a, "matvec");
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
    require_finite_span(w, "quad_form");
    require_finite_matrix(s, "quad_form");
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

/// Jitter ladder: 1e-12 .. 1e-6 x mean(diag), x10 per try.
constexpr double kJitterStart = 1e-12;
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

/// Validates shape, finiteness, symmetry and diagonal; returns mean(diag).
double validate_cov(const Matrix& cov, const char* who) {
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
bool factor_with_ladder(const Matrix& a, double scale, CholeskyResult& res) {
    if (try_cholesky(a, res.lower)) return true;
    const std::size_t n = a.rows();
    for (double eps = kJitterStart * scale; eps <= kMaxRelJitter * scale * (1.0 + 1e-9);
         eps *= 10.0) {
        Matrix b = a;
        for (std::size_t i = 0; i < n; ++i) b(i, i) += eps;
        if (try_cholesky(b, res.lower)) {
            res.jitter_added = eps;
            res.warning += "Cholesky needed diagonal jitter " + std::to_string(eps / scale) +
                           " x mean(diag). ";
            return true;
        }
    }
    return false;
}

/// V diag(lam) V^T.
Matrix rebuild(const Matrix& v, const std::vector<double>& lam) {
    const std::size_t n = v.rows();
    Matrix out(n, n);
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

Matrix psd_project(const Matrix& a, double floor) {
    EigenResult e = symmetric_eigen(a);
    for (double& l : e.values) l = std::max(l, floor);
    return rebuild(e.vectors, e.values);
}

double frobenius_diff(const Matrix& a, const Matrix& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j) s += (a(i, j) - b(i, j)) * (a(i, j) - b(i, j));
    return std::sqrt(s);
}

double frobenius(const Matrix& a) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j) s += a(i, j) * a(i, j);
    return std::sqrt(s);
}

}  // namespace

EigenResult symmetric_eigen(const Matrix& a_in) {
    if (!(a_in.rows() == a_in.cols() && a_in.rows() > 0)) {
        throw std::invalid_argument("symmetric_eigen: matrix must be square and non-empty");
    }
    const std::size_t n = a_in.rows();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (!std::isfinite(a_in(i, j))) {
                throw std::invalid_argument("symmetric_eigen: matrix contains NaN or inf");
            }
    Matrix a = a_in;
    Matrix v = Matrix::identity(n);
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0, diag = 0.0;
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
                    const double akp = a(k, p), akq = a(k, q);
                    a(k, p) = c * akp - s * akq;
                    a(k, q) = s * akp + c * akq;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double apk = a(p, k), aqk = a(q, k);
                    a(p, k) = c * apk - s * aqk;
                    a(q, k) = s * apk + c * aqk;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double vkp = v(k, p), vkq = v(k, q);
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
    EigenResult res;
    res.values.resize(n);
    res.vectors = Matrix(n, n);
    for (std::size_t k = 0; k < n; ++k) {
        res.values[k] = a(order[k], order[k]);
        for (std::size_t i = 0; i < n; ++i) res.vectors(i, k) = v(i, order[k]);
    }
    return res;
}

Matrix repair_covariance(const Matrix& cov, Repair repair) {
    const double scale = validate_cov(cov, "repair_covariance");
    const std::size_t n = cov.rows();
    if (repair == Repair::None) return cov;
    if (repair == Repair::Clip) {
        Matrix b = psd_project(cov, 1e-10 * scale);
        // Restore the original variances: B_ij sqrt(A_ii / B_ii) sqrt(A_jj / B_jj).
        std::vector<double> d(n);
        for (std::size_t i = 0; i < n; ++i) d[i] = std::sqrt(cov(i, i) / b(i, i));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) b(i, j) *= d[i] * d[j];
        return b;
    }
    // Higham (2002) nearest correlation matrix with Dykstra correction.
    std::vector<double> vol(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!(cov(i, i) > 0.0)) {
            throw std::invalid_argument(
                "repair_covariance: Higham repair needs strictly positive variances");
        }
        vol[i] = std::sqrt(cov(i, i));
    }
    Matrix y(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) y(i, j) = cov(i, j) / (vol[i] * vol[j]);
    Matrix ds(n, n);
    bool converged = false;
    for (int it = 0; it < 1000 && !converged; ++it) {
        Matrix r = y;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) r(i, j) -= ds(i, j);
        Matrix x = psd_project(r, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) ds(i, j) = x(i, j) - r(i, j);
        for (std::size_t i = 0; i < n; ++i) x(i, i) = 1.0;
        converged = frobenius_diff(x, y) <= 1e-12 * frobenius(x);
        y = std::move(x);
    }
    if (!converged) {
        throw std::runtime_error(
            "repair_covariance: Higham projection did not converge in 1000 iterations");
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) y(i, j) *= vol[i] * vol[j];
    return y;
}

CholeskyResult cholesky(const Matrix& cov, Repair repair) {
    const double scale = validate_cov(cov, "cholesky");
    const std::size_t n = cov.rows();
    CholeskyResult res;
    if (scale == 0.0) {
        // All variances zero: only the zero matrix is PSD, and L = 0 factors it.
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                if (cov(i, j) != 0.0) {
                    throw std::runtime_error(
                        "cholesky: zero variances with non-zero covariances");
                }
        res.lower = Matrix(n, n);
        return res;
    }
    if (factor_with_ladder(cov, scale, res)) return res;
    if (repair == Repair::None) {
        const double lam_min = symmetric_eigen(cov).values.front();
        throw std::runtime_error(
            "cholesky: covariance is materially indefinite (most negative eigenvalue " +
            std::to_string(lam_min) +
            "; jitter cap 1e-6 x mean(diag) exceeded). Repair it (Repair::Clip or "
            "Repair::Higham) before use.");
    }
    const Matrix fixed = repair_covariance(cov, repair);
    res = CholeskyResult{};
    res.repaired = true;
    res.warning = std::string("covariance repaired (") +
                  (repair == Repair::Clip ? "eigenvalue clip" : "Higham") + "). ";
    if (!factor_with_ladder(fixed, scale, res)) {
        throw std::runtime_error("cholesky: repaired covariance still does not factor");
    }
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
