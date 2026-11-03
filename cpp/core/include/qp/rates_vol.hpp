// Rates volatility: SABR implied vol and the Hull-White trinomial Bermudan.
//
// C++ port of python/core/src/qp_core/vol/sabr.py (sabr_vol) and of
// bermudan_tree_core in python/equity/04-fixed-income/src/fi_rates/bermudan.py;
// the Rust port is rust/core/src/rates_vol.rs. Formulas, sources and worked
// examples are in the Python docstrings. In short:
//
//   * SABR: Hagan, Kumar, Lesniewski and Woodward (2002) eq. (2.17a)
//     (Variant::hagan) with the Obloj (2008) leading term (Variant::obloj);
//     z/x(z) by its series near z = 0 (West 2005); shifted on F+s, K+s.
//   * Bermudan: Hull-White trinomial tree, spacing sqrt(3 V_i) per step,
//     nearest-node branching, alpha_i fitted to the discount factors by
//     forward induction (Brigo and Mercurio 2006, section 3.3.3), coupon bond
//     rolled back beside the option.
//
// Expressions are in the Python order; tests/test_rates_vol.cpp checks the
// Python reference values (the Rust fixture) to a few ulps.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace qp::rates_vol {

enum class Variant { hagan, obloj };

namespace detail {
inline constexpr double kSqrt3 = 1.7320508075688772;

inline double z_over_x(double z, double rho) {
    if (std::fabs(z) < 1e-6) return 1.0 - 0.5 * rho * z + (2.0 - 3.0 * rho * rho) * z * z / 12.0;
    const double root = std::sqrt(1.0 - 2.0 * rho * z + z * z);
    const double x = std::log1p((z + (z * z - 2.0 * rho * z) / (root + 1.0)) / (1.0 - rho));
    return z / x;
}
}  // namespace detail

// Black (lognormal) implied vol of SABR at strike k. Throws std::invalid_argument
// on the inputs the Python rejects.
inline double sabr_vol(double f, double k, double t, double alpha, double beta, double rho,
                       double nu, double shift = 0.0, Variant variant = Variant::obloj) {
    for (double v : {f, k, t, alpha, beta, rho, nu, shift})
        if (!std::isfinite(v)) throw std::invalid_argument("sabr_vol: inputs must be finite");
    if (f + shift <= 0.0 || k + shift <= 0.0)
        throw std::invalid_argument("sabr_vol: F + shift and K + shift must be > 0");
    if (t < 0.0 || alpha <= 0.0 || nu < 0.0 || beta < 0.0 || beta > 1.0 ||
        !(-1.0 < rho && rho < 1.0))
        throw std::invalid_argument(
            "sabr_vol: need T >= 0, alpha > 0, nu >= 0, 0 <= beta <= 1, -1 < rho < 1");
    const double fs = f + shift, ks = k + shift;
    const double omb = 1.0 - beta;
    const double log_fk = std::log(fs / ks);
    const double fk_mid = std::pow(fs * ks, 0.5 * omb);
    const double correction = 1.0 + t * (omb * omb * alpha * alpha / (24.0 * fk_mid * fk_mid) +
                                         rho * beta * nu * alpha / (4.0 * fk_mid) +
                                         (2.0 - 3.0 * rho * rho) * nu * nu / 24.0);
    if (variant == Variant::hagan) {
        const double z = nu / alpha * fk_mid * log_fk;
        const double l2 = log_fk * log_fk;
        const double denom =
            1.0 + (omb * omb) / 24.0 * l2 + (omb * omb * omb * omb) / 1920.0 * (l2 * l2);
        return alpha / (fk_mid * denom) * detail::z_over_x(z, rho) * correction;
    }
    double z = 0.0, lead = 0.0;
    if (omb == 0.0) {
        z = nu * log_fk / alpha;
        lead = alpha;
    } else {
        const double diff = std::pow(ks, omb) * std::expm1(omb * log_fk);
        z = nu * diff / (alpha * omb);
        lead = log_fk == 0.0 ? alpha / std::pow(ks, omb) : alpha * omb * log_fk / diff;
    }
    return lead * detail::z_over_x(z, rho) * correction;
}

// Bermudan on a Hull-White trinomial tree; see bermudan_tree_core (Python).
inline double bermudan_tree(double a, double sigma, std::span<const double> times,
                            std::span<const double> dfs, std::span<const double> cash,
                            const std::vector<bool>& exercise, bool payer) {
    if (times.size() < 2 || dfs.size() != times.size() || cash.size() != times.size() ||
        exercise.size() != times.size())
        throw std::invalid_argument(
            "bermudan_tree: times, dfs, cash and exercise must have equal length >= 2");
    const std::size_t n = times.size() - 1;
    if (times[0] != 0.0) throw std::invalid_argument("bermudan_tree: times must start at 0");
    for (std::size_t i = 0; i < n; ++i)
        if (times[i + 1] <= times[i])
            throw std::invalid_argument("bermudan_tree: times must increase strictly");
    if (!(a > 0.0 && sigma > 0.0))
        throw std::invalid_argument("bermudan_tree: need a > 0 and sigma > 0");

    struct Probs {
        double p0, p1, p2;
    };
    std::vector<double> dx(n + 1, 0.0);
    std::vector<std::int64_t> jmin(n + 1, 0);
    std::vector<std::size_t> width(n + 1, 1);
    std::vector<std::vector<std::int64_t>> kk(n);
    std::vector<std::vector<Probs>> pp(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double dt = times[i + 1] - times[i];
        const double v2 = sigma * sigma * (1.0 - std::exp(-2.0 * a * dt)) / (2.0 * a);
        const double v = std::sqrt(v2);
        dx[i + 1] = v * detail::kSqrt3;
        const double decay = std::exp(-a * dt);
        kk[i].resize(width[i]);
        pp[i].resize(width[i]);
        for (std::size_t jj = 0; jj < width[i]; ++jj) {
            const double x = static_cast<double>(jmin[i] + static_cast<std::int64_t>(jj)) * dx[i];
            const double m = x * decay;
            const auto k = static_cast<std::int64_t>(std::floor(m / dx[i + 1] + 0.5));
            const double e = m - static_cast<double>(k) * dx[i + 1];
            const double e2 = e * e / v2;
            const double e3 = e * detail::kSqrt3 / v;
            kk[i][jj] = k;
            pp[i][jj] = {(1.0 + e2 - e3) / 6.0, (2.0 - e2) / 3.0, (1.0 + e2 + e3) / 6.0};
        }
        const auto [lo, hi] = std::minmax_element(kk[i].begin(), kk[i].end());
        jmin[i + 1] = *lo - 1;
        width[i + 1] = static_cast<std::size_t>(*hi + 1 - jmin[i + 1] + 1);
    }
    std::vector<double> alpha(n, 0.0);
    std::vector<double> q{1.0};
    for (std::size_t i = 0; i < n; ++i) {
        const double dt = times[i + 1] - times[i];
        double s = 0.0;
        for (std::size_t jj = 0; jj < width[i]; ++jj) {
            const double x = static_cast<double>(jmin[i] + static_cast<std::int64_t>(jj)) * dx[i];
            s += q[jj] * std::exp(-x * dt);
        }
        alpha[i] = std::log(s / dfs[i + 1]) / dt;
        std::vector<double> d(width[i]);
        for (std::size_t jj = 0; jj < width[i]; ++jj) {
            const double x = static_cast<double>(jmin[i] + static_cast<std::int64_t>(jj)) * dx[i];
            d[jj] = q[jj] * std::exp(-(x + alpha[i]) * dt);
        }
        std::vector<double> qn(width[i + 1], 0.0);
        // Same order as the Python: all p0 terms, then p1, then p2.
        for (std::size_t jj = 0; jj < width[i]; ++jj)
            qn[static_cast<std::size_t>(kk[i][jj] - jmin[i + 1]) - 1] += d[jj] * pp[i][jj].p0;
        for (std::size_t jj = 0; jj < width[i]; ++jj)
            qn[static_cast<std::size_t>(kk[i][jj] - jmin[i + 1])] += d[jj] * pp[i][jj].p1;
        for (std::size_t jj = 0; jj < width[i]; ++jj)
            qn[static_cast<std::size_t>(kk[i][jj] - jmin[i + 1]) + 1] += d[jj] * pp[i][jj].p2;
        q = std::move(qn);
    }
    std::vector<double> bond(width[n], cash[n]);
    std::vector<double> opt(width[n], 0.0);
    for (std::size_t ii = n; ii-- > 0;) {
        const double dt = times[ii + 1] - times[ii];
        std::vector<double> nb(width[ii]), no(width[ii]);
        for (std::size_t jj = 0; jj < width[ii]; ++jj) {
            const double x = static_cast<double>(jmin[ii] + static_cast<std::int64_t>(jj)) * dx[ii];
            const double d = std::exp(-(x + alpha[ii]) * dt);
            const auto c = static_cast<std::size_t>(kk[ii][jj] - jmin[ii + 1]);
            const Probs& p = pp[ii][jj];
            const double b = d * (p.p0 * bond[c - 1] + p.p1 * bond[c] + p.p2 * bond[c + 1]);
            double o = d * (p.p0 * opt[c - 1] + p.p1 * opt[c] + p.p2 * opt[c + 1]);
            if (exercise[ii]) {
                const double pay = payer ? 1.0 - b : b - 1.0;
                if (pay > o) o = pay;
            }
            nb[jj] = b + cash[ii];
            no[jj] = o;
        }
        bond = std::move(nb);
        opt = std::move(no);
    }
    return opt[0];
}

}  // namespace qp::rates_vol
