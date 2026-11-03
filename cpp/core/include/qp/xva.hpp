// Capital and xVA kernels: SA-CCR (BCBS 279 / CRE52), Hull-White bond and
// swaption closed forms, and the CVA / DVA / FVA / effective-EPE sums.
//
// C++ port of qp_core.capital.saccr and qp_core.xva (Python holds the
// derivations and sources) and twin of rust/core/src/xva.rs. Every function
// takes plain numbers (discount factors, not a curve object) so the three
// languages are checked on one fixture,
// rust/core/tests/data/xva_capital_parity.txt. Invalid input throws
// std::invalid_argument; a failed root bracket throws std::runtime_error.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "qp/normal.hpp"

namespace qp::xva {

inline constexpr double kBdPerYear = 250.0;

// ------------------------------------------------------------------ SA-CCR

[[nodiscard]] inline double saccr_supervisory_duration(double start, double end) {
    if (!std::isfinite(start) || !std::isfinite(end)) throw std::invalid_argument("start and end must be finite");
    const double s = start > 0.0 ? std::max(start, 10.0 / kBdPerYear) : 0.0;
    const double e = std::max(end, 10.0 / kBdPerYear);
    if (e < s) throw std::invalid_argument("end must not precede start");
    return (std::exp(-0.05 * s) - std::exp(-0.05 * e)) / 0.05;
}

[[nodiscard]] inline double saccr_supervisory_delta(double price, double strike, double vol, double expiry,
                                                    bool call, bool bought, double shift = 0.0) {
    const double p = price + shift, k = strike + shift;
    if (!(p > 0.0 && k > 0.0)) throw std::invalid_argument("shifted price and strike must be positive");
    if (!(vol > 0.0 && expiry > 0.0)) throw std::invalid_argument("vol and expiry must be positive");
    const double srt = vol * std::sqrt(expiry);
    const double d1 = (std::log(p / k) + 0.5 * vol * vol * expiry) / srt;
    const double d = call ? qp::norm_cdf(d1) : -qp::norm_cdf(-d1);
    return bought ? d : -d;
}

[[nodiscard]] inline double saccr_maturity_factor(double maturity, std::optional<double> mpor_days) {
    if (mpor_days) {
        if (!(*mpor_days > 0.0)) throw std::invalid_argument("mpor must be positive");
        return 1.5 * std::sqrt(*mpor_days / kBdPerYear);
    }
    if (!std::isfinite(maturity)) throw std::invalid_argument("maturity must be finite");
    return std::sqrt(std::clamp(maturity, 10.0 / kBdPerYear, 1.0));
}

[[nodiscard]] inline double saccr_multiplier(double v, double c, double addon) {
    if (addon < 0.0) throw std::invalid_argument("addon must be non-negative");
    if (addon == 0.0) return 1.0;
    constexpr double f = 0.05;
    return std::min(1.0, f + (1.0 - f) * std::exp((v - c) / (2.0 * (1.0 - f) * addon)));
}

[[nodiscard]] inline double saccr_ir_effective_notional(double d1, double d2, double d3) noexcept {
    const double q = d1 * d1 + d2 * d2 + d3 * d3 + 1.4 * d1 * d2 + 1.4 * d2 * d3 + 0.6 * d1 * d3;
    return std::sqrt(std::max(q, 0.0));
}

[[nodiscard]] inline double saccr_single_factor_addon(std::span<const double> addons, std::span<const double> rhos) {
    if (addons.size() != rhos.size()) throw std::invalid_argument("addons and rhos must have equal length");
    double sys = 0.0, idio = 0.0;
    for (std::size_t i = 0; i < addons.size(); ++i) {
        sys += rhos[i] * addons[i];
        idio += (1.0 - rhos[i] * rhos[i]) * addons[i] * addons[i];
    }
    return std::sqrt(sys * sys + idio);
}

// -------------------------------------------------------------- Hull-White

struct HullWhite {
    double a;
    double sigma;

    HullWhite(double a_, double sigma_) : a(a_), sigma(sigma_) {
        if (!(a > 0.0 && std::isfinite(a) && sigma > 0.0 && std::isfinite(sigma)))
            throw std::invalid_argument("a and sigma must be positive and finite");
    }

    [[nodiscard]] double b(double t, double T) const noexcept { return (1.0 - std::exp(-a * (T - t))) / a; }

    [[nodiscard]] double v(double t, double T) const noexcept {
        const double tau = T - t;
        return (sigma * sigma / (a * a)) *
               (tau + 2.0 * std::exp(-a * tau) / a - std::exp(-2.0 * a * tau) / (2.0 * a) - 1.5 / a);
    }

    [[nodiscard]] double log_a(double t, double T, double p_t, double p_T) const noexcept {
        return std::log(p_T / p_t) + 0.5 * (v(t, T) - v(0.0, T) + v(0.0, t));
    }

    [[nodiscard]] double zcb_option(double T, double S, double p_T, double p_S, double strike, bool call) const {
        if (!(0.0 < T && T < S && strike > 0.0)) throw std::invalid_argument("need 0 < T < S and strike > 0");
        const double sp = sigma * std::sqrt((1.0 - std::exp(-2.0 * a * T)) / (2.0 * a)) * b(T, S);
        const double h = std::log(p_S / (p_T * strike)) / sp + 0.5 * sp;
        return call ? p_S * qp::norm_cdf(h) - strike * p_T * qp::norm_cdf(h - sp)
                    : strike * p_T * qp::norm_cdf(-h + sp) - p_S * qp::norm_cdf(-h);
    }

    // European swaption by Jamshidian; p_pay[i] = P(0, pay_times[i]).
    [[nodiscard]] double swaption(double expiry, double p_exp, std::span<const double> pay_times,
                                  std::span<const double> p_pay, std::span<const double> accruals, double strike,
                                  bool payer) const {
        const std::size_t n = pay_times.size();
        if (n == 0 || p_pay.size() != n || accruals.size() != n)
            throw std::invalid_argument("pay_times, p_pay and accruals must be non-empty and equal length");
        if (!(expiry > 0.0) || std::any_of(pay_times.begin(), pay_times.end(), [&](double t) { return t <= expiry; }))
            throw std::invalid_argument("pay times must follow expiry > 0");
        std::vector<double> c(n), la(n), bb(n);
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = strike * accruals[i];
            la[i] = log_a(expiry, pay_times[i], p_exp, p_pay[i]);
            bb[i] = b(expiry, pay_times[i]);
        }
        c[n - 1] += 1.0;
        for (double ci : c)
            if (ci <= 0.0) throw std::invalid_argument("Jamshidian needs positive coupons");
        auto f = [&](double x) {
            double acc = 0.0;
            for (std::size_t i = 0; i < n; ++i) acc += c[i] * std::exp(la[i] - bb[i] * x);
            return acc - 1.0;
        };
        double lo = -1.0, hi = 1.0;
        int k = 0;
        while (f(lo) < 0.0) {
            lo *= 2.0;
            if (++k > 60) throw std::runtime_error("Jamshidian bracket failed");
        }
        while (f(hi) > 0.0) {
            hi *= 2.0;
            if (++k > 60) throw std::runtime_error("Jamshidian bracket failed");
        }
        for (int it = 0; it < 200; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (mid == lo || mid == hi) break;
            (f(mid) > 0.0 ? lo : hi) = mid;
        }
        const double xstar = 0.5 * (lo + hi);
        double total = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            total += c[i] * zcb_option(expiry, pay_times[i], p_exp, p_pay[i], std::exp(la[i] - bb[i] * xstar), !payer);
        return total;
    }
};

// --------------------------------------------------------------------- xVA

struct Hazard {
    std::span<const double> times;
    std::span<const double> hazards;

    [[nodiscard]] double survival(double t) const noexcept {
        if (t <= 0.0) return 1.0;
        double acc = 0.0, prev = 0.0;
        for (std::size_t i = 0; i < times.size(); ++i) {
            if (t <= times[i]) return std::exp(-(acc + hazards[i] * (t - prev)));
            acc += hazards[i] * (times[i] - prev);
            prev = times[i];
        }
        return std::exp(-(acc + hazards.back() * (t - prev)));
    }
};

namespace detail {
inline void check_grid(std::span<const double> t, std::span<const double> p) {
    if (t.size() != p.size() || t.size() < 2 || t[0] != 0.0)
        throw std::invalid_argument("times must start at 0 and match the profile");
    for (std::size_t k = 1; k < t.size(); ++k)
        if (t[k] <= t[k - 1]) throw std::invalid_argument("times must increase strictly");
}
}  // namespace detail

// (1 - R) sum expo(t_k) [S(t_{k-1}) - S(t_k)] w(t_k); pass -ENE* for DVA.
[[nodiscard]] inline double default_leg(std::span<const double> times, std::span<const double> expo, const Hazard& who,
                                        double recovery, const Hazard* other = nullptr) {
    if (!(recovery >= 0.0 && recovery <= 1.0)) throw std::invalid_argument("recovery must be in [0, 1]");
    detail::check_grid(times, expo);
    double acc = 0.0;
    for (std::size_t k = 1; k < times.size(); ++k) {
        const double dpd = who.survival(times[k - 1]) - who.survival(times[k]);
        const double w = other ? other->survival(times[k]) : 1.0;
        acc += expo[k] * dpd * w;
    }
    return (1.0 - recovery) * acc;
}

[[nodiscard]] inline std::pair<double, double> fva(std::span<const double> times, std::span<const double> ee_disc,
                                                   std::span<const double> ene_disc, double borrow, double lend,
                                                   const Hazard& cpty, const Hazard& bank) {
    detail::check_grid(times, ee_disc);
    detail::check_grid(times, ene_disc);
    double fca = 0.0, fba = 0.0;
    for (std::size_t k = 1; k < times.size(); ++k) {
        const double w = (times[k] - times[k - 1]) * cpty.survival(times[k]) * bank.survival(times[k]);
        fca += borrow * ee_disc[k] * w;
        fba += lend * -ene_disc[k] * w;
    }
    return {fca, fba};
}

[[nodiscard]] inline double eepe(std::span<const double> times, std::span<const double> ee, double maturity) {
    detail::check_grid(times, ee);
    const double h = std::min(maturity, 1.0);
    if (!(h > 0.0)) throw std::invalid_argument("maturity must be positive");
    double eee = 0.0, acc = 0.0;
    for (std::size_t k = 1; k < times.size(); ++k) {
        eee = std::max(eee, ee[k]);
        if (times[k] <= h + 1e-12) acc += eee * (times[k] - times[k - 1]);
    }
    return acc / h;
}

[[nodiscard]] inline std::size_t pfe_index(std::size_t n, double q) {
    if (n < 1 || !(q > 0.0 && q < 1.0)) throw std::invalid_argument("need n >= 1 and 0 < q < 1");
    const auto c = static_cast<std::size_t>(std::ceil(q * static_cast<double>(n)));
    return c == 0 ? 0 : c - 1;
}

}  // namespace qp::xva
