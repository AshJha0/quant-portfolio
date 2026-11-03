// ISDA CDS Standard Model legs, the numeric kernel under the credit indices.
//
// C++ port of isda_legs_kernel in python/core/src/qp_core/credit_index/isda.py;
// the Rust port is rust/core/src/credit_index.rs. Conventions, sources and the
// worked example (Markit's published upfronts for 21 May 2009) are in the
// Python docstring: piecewise-flat hazard and forwards, exact integration on
// the union of the knots (O'Kane 2008 eqs. 6.6, 6.11), accrued-on-default with
// the ISDA half-day bias 1/730, Taylor expansion when |k D| < 1e-8.
// Same order of operations as the Python; tests/test_credit_bondfut.cpp checks
// the Python reference values.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace qp::credit_index {

namespace detail {
inline constexpr double kBias = 1.0 / 730.0;

inline double cum_hazard(std::span<const double> t, std::span<const double> r, double x) {
    double out = 0.0, prev = 0.0;
    const std::size_t n = t.size();
    for (std::size_t k = 0; k < n; ++k) {
        const double hi = k + 1 < n ? t[k] : std::numeric_limits<double>::infinity();
        if (x <= prev) break;
        out += r[k] * (std::min(x, hi) - prev);
        prev = hi;
    }
    return out;
}

inline double df(std::span<const double> t, std::span<const double> z, double x) {
    if (x <= 0.0) return 1.0;
    const std::size_t n = t.size();
    if (x <= t[0]) return std::exp(-z[0] * x);
    for (std::size_t k = 1; k < n; ++k) {
        if (x <= t[k]) {
            const double a = z[k - 1] * t[k - 1], b = z[k] * t[k];
            const double w = (x - t[k - 1]) / (t[k] - t[k - 1]);
            return std::exp(-(a + w * (b - a)));
        }
    }
    if (n == 1) return std::exp(-z[0] * x);
    const double f = (z[n - 1] * t[n - 1] - z[n - 2] * t[n - 2]) / (t[n - 1] - t[n - 2]);
    return std::exp(-(z[n - 1] * t[n - 1] + f * (x - t[n - 1])));
}

inline std::vector<double> grid(double a, double b, std::span<const double> x,
                                std::span<const double> y) {
    std::vector<double> p{a, b};
    for (double v : x)
        if (a < v && v < b) p.push_back(v);
    for (double v : y)
        if (a < v && v < b) p.push_back(v);
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
    return p;
}

inline std::pair<double, double> segment(double lam, double f, double d) {
    const double k = lam + f, x = k * d;
    if (std::fabs(x) < 1e-8) return {d * (1.0 - x / 2.0), d * d * (0.5 - x / 3.0)};
    const double e = std::exp(-x);
    return {(1.0 - e) / k, (1.0 - e * (1.0 + x)) / (k * k)};
}

inline void check_curve(std::span<const double> t, std::span<const double> v,
                        const std::string& what) {
    if (t.empty() || t.size() != v.size())
        throw std::invalid_argument(what + ": times and values must be non-empty and equal length");
    for (std::size_t i = 0; i < t.size(); ++i) {
        if ((i == 0 && t[0] <= 0.0) || (i > 0 && t[i] <= t[i - 1]))
            throw std::invalid_argument(what + ": times must be positive and strictly increasing");
        if (!std::isfinite(t[i]) || !std::isfinite(v[i]))
            throw std::invalid_argument(what + ": values must be finite");
    }
}
}  // namespace detail

struct Schedule {
    std::span<const double> acc_start;  // hazard time (from the start of the step-in day)
    std::span<const double> acc_end;    // hazard time
    std::span<const double> pay_time;   // discount time (from the trade date)
    std::span<const double> acc_frac;   // ACT/360
};

// (protection PV, risky annuity incl. accrued-on-default, df to cash settlement).
inline std::tuple<double, double, double> isda_legs(
    std::span<const double> hz_t, std::span<const double> hz_r, std::span<const double> df_t,
    std::span<const double> df_z, double recovery, double t_end, const Schedule& s, double t_cash) {
    detail::check_curve(hz_t, hz_r, "hazard");
    detail::check_curve(df_t, df_z, "discount");
    if (!(recovery >= 0.0 && recovery < 1.0))
        throw std::invalid_argument("recovery must lie in [0, 1)");
    const std::size_t n = s.acc_start.size();
    if (s.acc_end.size() != n || s.pay_time.size() != n || s.acc_frac.size() != n)
        throw std::invalid_argument("schedule arrays must have equal length");
    auto disc = [&](double u) { return detail::df(df_t, df_z, u); };
    auto qp = [&](double u) {
        return std::exp(-detail::cum_hazard(hz_t, hz_r, u)) * disc(std::max(u, 0.0));
    };
    auto lam_f = [&](double u0, double u1) {
        const double lam =
            (detail::cum_hazard(hz_t, hz_r, u1) - detail::cum_hazard(hz_t, hz_r, u0)) / (u1 - u0);
        const double f = std::log(disc(std::max(u0, 0.0)) / disc(u1)) / (u1 - u0);
        return std::pair{lam, f};
    };
    double prot = 0.0;
    const auto g = detail::grid(0.0, t_end, hz_t, df_t);
    for (std::size_t j = 0; j + 1 < g.size(); ++j) {
        const auto [lam, f] = lam_f(g[j], g[j + 1]);
        const double i0 = detail::segment(lam, f, g[j + 1] - g[j]).first;
        prot += lam * qp(g[j]) * i0;
    }
    prot *= 1.0 - recovery;
    double prem = 0.0;
    for (std::size_t p = 0; p < n; ++p) {
        const double st = s.acc_start[p], e = s.acc_end[p];
        prem += s.acc_frac[p] * std::exp(-detail::cum_hazard(hz_t, hz_r, e)) * disc(s.pay_time[p]);
        const auto gg = detail::grid(std::max(st, 0.0), e, hz_t, df_t);
        double aod = 0.0;
        for (std::size_t j = 0; j + 1 < gg.size(); ++j) {
            const double u0 = gg[j], u1 = gg[j + 1];
            const auto [lam, f] = lam_f(u0, u1);
            const auto [i0, i1] = detail::segment(lam, f, u1 - u0);
            aod += lam * qp(u0) * ((u0 - st + detail::kBias) * i0 + i1);
        }
        prem += 365.0 / 360.0 * aod;
    }
    return {prot, prem, disc(t_cash)};
}

}  // namespace qp::credit_index
