// Risk kernels: quote-to-node Jacobian algebra, par-rate DV01, quote gamma
// with the bootstrap-curvature term, vega bucketing on vol grids and the
// P&L-explain Taylor attribution (plan W14, C++ parity D1).
//
// C++ port of python/core/src/qp_core/risk/{jacobian,vega_grid,pnl_explain}.py
// and of the algebra in python/equity/04-fixed-income/src/fi_rates/risk/par.py;
// the Rust port is rust/core/src/risk.rs. Plain idea, formal statement,
// worked examples and sources are in the Python docstrings. In short:
//
//   * quote risk g_q = J^-T g_z with J = dQ/dz (Andersen and Piterbarg,
//     Interest Rate Modeling vol. 3, 2010, ch. 26); par DV01 = 1e-4 g_q;
//   * quote gamma J^-T (Gamma_z - sum_i g_q,i H_i) J^-1. The shortcut
//     without the sum ignores the curvature of the bootstrap map and was
//     measured about 200% off on the Python lane's test book;
//   * multilinear vol-grid weights (flat extrapolation, partition of unity)
//     and bucket vegas by central differences;
//   * per-factor P&L delta_i dx_i + 0.5 dx_i (S dx)_i, S = (G + G^T)/2.
//
// Matrices are row-major std::vector<double>. tests/test_risk.cpp checks the
// Python reference values (rust/core/tests/data/risk_parity.txt, shared with
// Rust) to 1e-12 relative to the largest entry of each output.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace qp::risk {

// Largest 1-norm condition number accepted (the Python uses the 2-norm
// condition, within a factor n of this one).
inline constexpr double kMaxCondition = 1e12;

namespace detail {

inline void check_square(std::span<const double> m, std::size_t n, const char* name) {
    if (n == 0 || m.size() != n * n)
        throw std::invalid_argument(std::string(name) + " must be a non-empty n x n row-major matrix");
    for (double v : m)
        if (!std::isfinite(v)) throw std::invalid_argument(std::string(name) + " contains NaN or infinite values");
}

struct Lu {
    std::size_t n;
    std::vector<double> a;
    std::vector<std::size_t> piv;

    Lu(std::span<const double> m, std::size_t n_) : n(n_), a(m.begin(), m.end()), piv(n_) {
        for (std::size_t i = 0; i < n; ++i) piv[i] = i;
        for (std::size_t k = 0; k < n; ++k) {
            std::size_t p = k;
            for (std::size_t i = k + 1; i < n; ++i)
                if (std::fabs(a[i * n + k]) > std::fabs(a[p * n + k])) p = i;
            if (a[p * n + k] == 0.0)
                throw std::invalid_argument("Jacobian is numerically singular (zero pivot)");
            if (p != k) {
                for (std::size_t j = 0; j < n; ++j) std::swap(a[k * n + j], a[p * n + j]);
                std::swap(piv[k], piv[p]);
            }
            for (std::size_t i = k + 1; i < n; ++i) {
                const double f = a[i * n + k] / a[k * n + k];
                a[i * n + k] = f;
                for (std::size_t j = k + 1; j < n; ++j) a[i * n + j] -= f * a[k * n + j];
            }
        }
    }

    std::vector<double> solve(std::span<const double> b) const {
        std::vector<double> x(n);
        for (std::size_t i = 0; i < n; ++i) x[i] = b[piv[i]];
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < i; ++j) x[i] -= a[i * n + j] * x[j];
        for (std::size_t i = n; i-- > 0;) {
            for (std::size_t j = i + 1; j < n; ++j) x[i] -= a[i * n + j] * x[j];
            x[i] /= a[i * n + i];
        }
        return x;
    }

    std::vector<double> inverse() const {
        std::vector<double> inv(n * n), e(n);
        for (std::size_t j = 0; j < n; ++j) {
            std::fill(e.begin(), e.end(), 0.0);
            e[j] = 1.0;
            const auto col = solve(e);
            for (std::size_t i = 0; i < n; ++i) inv[i * n + j] = col[i];
        }
        return inv;
    }
};

inline std::vector<double> transpose(std::span<const double> m, std::size_t n) {
    std::vector<double> t(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) t[j * n + i] = m[i * n + j];
    return t;
}

inline double norm1(std::span<const double> m, std::size_t n) {
    double best = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += std::fabs(m[i * n + j]);
        best = std::max(best, s);
    }
    return best;
}

inline std::vector<double> matmul(std::span<const double> a, std::span<const double> b, std::size_t n) {
    std::vector<double> c(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < n; ++k) {
            const double aik = a[i * n + k];
            for (std::size_t j = 0; j < n; ++j) c[i * n + j] += aik * b[k * n + j];
        }
    return c;
}

inline void check_step(double h) {
    if (!std::isfinite(h) || h <= 0.0) throw std::invalid_argument("finite-difference step must be > 0");
}

inline void check_point(std::span<const double> x) {
    if (x.empty()) throw std::invalid_argument("x must be a non-empty finite vector");
    for (double v : x)
        if (!std::isfinite(v)) throw std::invalid_argument("x must be a non-empty finite vector");
}

}  // namespace detail

// Inverse of a square Jacobian; throws std::invalid_argument when it is
// numerically singular (1-norm condition above kMaxCondition).
inline std::vector<double> checked_inverse(std::span<const double> jac, std::size_t n) {
    detail::check_square(jac, n, "Jacobian");
    auto inv = detail::Lu(jac, n).inverse();
    const double cond = detail::norm1(jac, n) * detail::norm1(inv, n);
    if (!std::isfinite(cond) || cond > kMaxCondition)
        throw std::invalid_argument("Jacobian is numerically singular (condition number above 1e12)");
    return inv;
}

// Quote risk dP/dq from node risk dP/dz: solves J^T g_q = g_z.
inline std::vector<double> node_to_quote_risk(std::span<const double> jac, std::size_t n,
                                              std::span<const double> node_risk) {
    detail::check_square(jac, n, "Jacobian");
    if (node_risk.size() != n) throw std::invalid_argument("node risk length must equal the node count");
    (void)checked_inverse(jac, n);
    return detail::Lu(detail::transpose(jac, n), n).solve(node_risk);
}

// Inverse map: dP/dz = J^T dP/dq.
inline std::vector<double> quote_to_node_risk(std::span<const double> jac, std::size_t n,
                                              std::span<const double> quote_risk) {
    detail::check_square(jac, n, "Jacobian");
    if (quote_risk.size() != n) throw std::invalid_argument("quote risk length must equal n");
    std::vector<double> out(n, 0.0);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i) out[j] += jac[i * n + j] * quote_risk[i];
    return out;
}

// Par-rate DV01 per basis point from quote risk per unit rate.
inline std::vector<double> par_dv01(std::span<const double> quote_risk) {
    std::vector<double> out(quote_risk.begin(), quote_risk.end());
    for (double& v : out) v *= 1e-4;
    return out;
}

// Quote-space cross-gamma J^-T (Gamma_z - sum_i g_q,i H_i) J^-1.
// quote_hessians holds the n node Hessians H_i one after another (n^3 values).
inline std::vector<double> quote_gamma(std::span<const double> jac, std::size_t n,
                                       std::span<const double> gamma_z,
                                       std::span<const double> quote_risk,
                                       std::span<const double> quote_hessians) {
    const auto inv = checked_inverse(jac, n);
    detail::check_square(gamma_z, n, "node gamma");
    if (quote_risk.size() != n || quote_hessians.size() != n * n * n)
        throw std::invalid_argument("need n quote risks and n^3 Hessian entries");
    std::vector<double> core(gamma_z.begin(), gamma_z.end());
    for (std::size_t jk = 0; jk < n * n; ++jk) {
        double corr = 0.0;
        for (std::size_t i = 0; i < n; ++i) corr += quote_risk[i] * quote_hessians[i * n * n + jk];
        core[jk] -= corr;
    }
    return detail::matmul(detail::matmul(detail::transpose(inv, n), core, n), inv, n);
}

// Central-difference Jacobian J[i][j] = d f_i / d x_j, row-major (m x n).
inline std::vector<double> fd_jacobian(const std::function<std::vector<double>(std::span<const double>)>& f,
                                       std::span<const double> x, double h) {
    detail::check_step(h);
    detail::check_point(x);
    const std::size_t n = x.size();
    std::vector<double> xp(x.begin(), x.end());
    std::vector<std::vector<double>> cols;
    for (std::size_t j = 0; j < n; ++j) {
        xp[j] = x[j] + h;
        const auto up = f(xp);
        xp[j] = x[j] - h;
        const auto dn = f(xp);
        xp[j] = x[j];
        std::vector<double> c(up.size());
        for (std::size_t i = 0; i < up.size(); ++i) c[i] = (up[i] - dn[i]) / (2.0 * h);
        cols.push_back(std::move(c));
    }
    const std::size_t m = cols[0].size();
    std::vector<double> out(m * n);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < m; ++i) out[i * n + j] = cols[j][i];
    return out;
}

// Full Hessian by central differences (symmetric by construction).
inline std::vector<double> cross_gamma(const std::function<double(std::span<const double>)>& f,
                                       std::span<const double> x, double h) {
    detail::check_step(h);
    detail::check_point(x);
    const std::size_t n = x.size();
    const double f0 = f(x);
    std::vector<double> out(n * n);
    auto at = [&](std::size_t i, double si, std::size_t j, double sj) {
        std::vector<double> v(x.begin(), x.end());
        v[i] += si * h;
        v[j] += sj * h;
        return f(v);
    };
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> v(x.begin(), x.end());
        v[i] = x[i] + h;
        const double up = f(v);
        v[i] = x[i] - h;
        const double dn = f(v);
        out[i * n + i] = (up - 2.0 * f0 + dn) / (h * h);
        for (std::size_t j = i + 1; j < n; ++j) {
            const double g = (at(i, 1, j, 1) - at(i, 1, j, -1) - at(i, -1, j, 1) + at(i, -1, j, -1)) / (4.0 * h * h);
            out[i * n + j] = g;
            out[j * n + i] = g;
        }
    }
    return out;
}

// Node vols on a rectangular 2-D or 3-D grid (row-major vols).
class VolGrid {
public:
    VolGrid(std::vector<std::vector<double>> axes, std::vector<double> vols)
        : axes_(std::move(axes)), vols_(std::move(vols)) {
        if (axes_.size() < 2 || axes_.size() > 3) throw std::invalid_argument("a vol grid has 2 or 3 axes");
        std::size_t size = 1;
        for (const auto& a : axes_) {
            if (a.empty()) throw std::invalid_argument("axes must be non-empty");
            for (std::size_t k = 0; k < a.size(); ++k)
                if (!std::isfinite(a[k]) || (k > 0 && a[k] <= a[k - 1]))
                    throw std::invalid_argument("axes must be finite and strictly increasing");
            size *= a.size();
        }
        if (vols_.size() != size) throw std::invalid_argument("vols size does not match the axes");
        for (double v : vols_)
            if (!std::isfinite(v) || v < 0.0) throw std::invalid_argument("node vols must be finite and >= 0");
    }

    const std::vector<std::vector<double>>& axes() const { return axes_; }
    const std::vector<double>& vols() const { return vols_; }

    // Multilinear weights at coords (row-major, flat extrapolation, sum to 1).
    std::vector<double> weights(std::span<const double> coords) const {
        if (coords.size() != axes_.size()) throw std::invalid_argument("wrong number of coordinates");
        std::vector<double> w{1.0};
        for (std::size_t d = 0; d < axes_.size(); ++d) {
            const auto& a = axes_[d];
            const double x = coords[d];
            std::vector<double> wa(a.size(), 0.0);
            if (a.size() == 1 || x <= a.front()) {
                wa.front() = 1.0;
            } else if (x >= a.back()) {
                wa.back() = 1.0;
            } else {
                const auto i = static_cast<std::size_t>(std::upper_bound(a.begin(), a.end(), x) - a.begin()) - 1;
                const double t = (x - a[i]) / (a[i + 1] - a[i]);
                wa[i] = 1.0 - t;
                wa[i + 1] = t;
            }
            std::vector<double> next;
            next.reserve(w.size() * wa.size());
            for (double p : w)
                for (double q : wa) next.push_back(p * q);
            w = std::move(next);
        }
        return w;
    }

    double vol(std::span<const double> coords) const {
        const auto w = weights(coords);
        double s = 0.0;
        for (std::size_t k = 0; k < w.size(); ++k) s += w[k] * vols_[k];
        return s;
    }

    VolGrid bumped(std::size_t index, double h) const {
        VolGrid g = *this;
        g.vols_.at(index) += h;
        return g;
    }

private:
    std::vector<std::vector<double>> axes_;
    std::vector<double> vols_;
};

// Bucket vegas (value per unit vol), row-major; one-sided where vol < h.
inline std::vector<double> vega_grid(const std::function<double(const VolGrid&)>& pv, const VolGrid& grid,
                                     double h) {
    if (!std::isfinite(h) || h <= 0.0) throw std::invalid_argument("bump must be > 0");
    std::optional<double> base;
    std::vector<double> out;
    out.reserve(grid.vols().size());
    for (std::size_t idx = 0; idx < grid.vols().size(); ++idx) {
        const double up = pv(grid.bumped(idx, h));
        if (grid.vols()[idx] >= h) {
            out.push_back((up - pv(grid.bumped(idx, -h))) / (2.0 * h));
        } else {
            if (!base) base = pv(grid);
            out.push_back((up - *base) / h);
        }
    }
    return out;
}

// Per-factor second-order P&L; gamma is n x n row-major or empty.
inline std::vector<double> taylor_attribution(std::span<const double> delta, std::span<const double> moves,
                                              std::span<const double> gamma = {}) {
    const std::size_t n = delta.size();
    if (n == 0 || moves.size() != n)
        throw std::invalid_argument("delta and move must be 1-D vectors of equal length");
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = delta[i] * moves[i];
    if (!gamma.empty()) {
        if (gamma.size() != n * n) throw std::invalid_argument("gamma must be n x n");
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < n; ++j) s += 0.5 * (gamma[i * n + j] + gamma[j * n + i]) * moves[j];
            out[i] += 0.5 * moves[i] * s;
        }
    }
    return out;
}

// One period's P&L explain (reporting currency).
struct PnlExplain {
    double hpl;
    std::vector<double> time;
    std::vector<double> risk;
    double new_trades;

    double theta() const {
        double s = 0.0;
        for (double v : time) s += v;
        return s;
    }
    double rtpl() const {
        double s = 0.0;
        for (double v : risk) s += v;
        return theta() + s;
    }
    double unexplained() const { return hpl - rtpl(); }
    double total() const { return hpl + new_trades; }
};

// Assembles an explain; throws on non-finite inputs (NaN policy: refuse).
inline PnlExplain explain(double hpl, std::span<const double> time, std::span<const double> delta,
                          std::span<const double> moves, std::span<const double> gamma = {},
                          double new_trades = 0.0) {
    auto risk = taylor_attribution(delta, moves, gamma);
    bool finite = std::isfinite(hpl) && std::isfinite(new_trades);
    for (double v : time) finite = finite && std::isfinite(v);
    for (double v : risk) finite = finite && std::isfinite(v);
    if (!finite) throw std::invalid_argument("explain inputs must be finite (NaN policy: refuse)");
    return PnlExplain{hpl, std::vector<double>(time.begin(), time.end()), std::move(risk), new_trades};
}

}  // namespace qp::risk
