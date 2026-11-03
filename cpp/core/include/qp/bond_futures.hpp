// Bond futures kernels: CME conversion factor, implied repo, net basis and the
// basic delivery (switch) option.
//
// C++ port of the *_kernel functions in
// python/equity/04-fixed-income/src/fi_rates/bond_futures.py; the Rust port is
// rust/core/src/bond_futures.rs. Formulas and sources (CME IR232; Burghardt,
// The Treasury Bond Basis; Hull s. 6.2) are in the Python docstring. Same order
// of operations as the Python; tests/test_credit_bondfut.cpp checks the Python
// reference values.
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qp::bond_futures {

// Unrounded CME conversion factor; coupon already rounded to 1/8%.
inline double cme_cf(double coupon, unsigned n, unsigned z, bool quarterly) {
    unsigned v;
    if (quarterly) {
        z = 3 * (z / 3);
        v = z < 7 ? z : 3;
    } else {
        v = z < 7 ? z : z - 6;
    }
    const double vd = static_cast<double>(v);
    const double a = 1.0 / std::pow(1.03, vd / 6.0);
    const double b = coupon / 2.0 * (6.0 - vd) / 6.0;
    const double k = z < 7 ? 1.0 / std::pow(1.03, static_cast<double>(2 * n))
                           : 1.0 / std::pow(1.03, static_cast<double>(2 * n + 1));
    const double d = coupon / 0.06 * (1.0 - k);
    return a * (coupon / 2.0 + k + d) - b;
}

inline double implied_repo(double dirty_now, double invoice, double coupon_cash, double tau,
                           double tau_c) {
    const double den = dirty_now * tau - coupon_cash * tau_c;
    if (!(den > 0.0)) throw std::invalid_argument("delivery must be after settlement");
    return (invoice + coupon_cash - dirty_now) / den;
}

inline double net_basis(double dirty_now, double coupon_cash, double tau, double tau_c, double repo,
                        double accrued_delivery, double futures_x_cf) {
    const double fwd_dirty = dirty_now * (1.0 + repo * tau) - coupon_cash * (1.0 + repo * tau_c);
    return fwd_dirty - accrued_delivery - futures_x_cf;
}

struct Bond {
    std::span<const double> exps;   // street-convention discount exponents
    std::span<const double> flows;  // cash flows per 100
    double accrued;                 // accrued at delivery
    unsigned freq;
    double fwd_yield;
    double cf;
};

namespace detail {
inline double ratio(const Bond& b, double y) {
    const double m = static_cast<double>(b.freq);
    double p = 0.0;
    for (std::size_t i = 0; i < b.exps.size(); ++i)
        p += b.flows[i] * std::pow(1.0 + y / m, -b.exps[i]);
    return (p - b.accrued) / b.cf;
}
}  // namespace detail

// (fair futures price, switch-option value); 801-point trapezoid on +-8 sd.
inline std::pair<double, double> delivery_option(std::span<const Bond> bonds, double sd) {
    if (!(sd >= 0.0)) throw std::invalid_argument("sd must be non-negative");
    if (bonds.empty()) throw std::invalid_argument("empty basket");
    constexpr std::size_t kNodes = 801;
    constexpr double kWidth = 8.0;
    if (sd == 0.0) {
        double best = std::numeric_limits<double>::infinity();
        for (const auto& b : bonds) best = std::fmin(best, detail::ratio(b, b.fwd_yield));
        return {best, 0.0};
    }
    const double h = 2.0 * kWidth / static_cast<double>(kNodes - 1);
    double den = 0.0, num = 0.0;
    std::vector<double> each(bonds.size(), 0.0);
    for (std::size_t j = 0; j < kNodes; ++j) {
        const double x = -kWidth + static_cast<double>(j) * h;
        const double w = std::exp(-0.5 * x * x) * ((j == 0 || j == kNodes - 1) ? 0.5 : 1.0);
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < bonds.size(); ++i) {
            const double r = detail::ratio(bonds[i], bonds[i].fwd_yield + sd * x);
            each[i] += w * r;
            best = std::fmin(best, r);
        }
        num += w * best;
        den += w;
    }
    const double fair = num / den;
    double min_each = std::numeric_limits<double>::infinity();
    for (double e : each) min_each = std::fmin(min_each, e / den);
    return {fair, min_each - fair};
}

}  // namespace qp::bond_futures
