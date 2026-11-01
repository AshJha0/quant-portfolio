// qp/iv_core.hpp: implied volatility from normalised Black, solved on the
// OTM side. Used by both options engines (eqopt::iv_core and
// fxopt::iv_core are using-declarations of these names).
//
// The same algorithm, constants and branch thresholds are implemented in
// the Python (_iv_core.py) and Rust (iv_core.rs) engines. Inputs to
// total_vol are the normalised OTM time value beta, its complement
// u = b_max - beta and x = -|ln(F/K)| <= 0; the result is the total
// volatility s = sigma sqrt(T). One evaluation is one pass of the loop.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "qp/constants.hpp"
#include "qp/normal.hpp"

namespace qp::iv_core {

inline constexpr int kMaxIter = 100;

// b(x,s) = e^{x/2} N(x/s+s/2) - e^{-x/2} N(x/s-s/2) for x <= 0, s > 0,
// written as cosh(x/2) A + sinh(x/2) (N(d1)+N(d2)), A = N(d1) - N(d2).
inline double normalised_otm_call(double x, double s) noexcept {
    const double h = x / s;
    const double t = 0.5 * s;
    const double d1 = h + t;
    const double d2 = h - t;
    double a;
    if (-x < 0.1 && s < 0.2) {
        // A = 2 n(h) sum_k He_{2k}(h) t^{2k+1} / (2k+1)!
        double he_prev = 1.0, he_cur = h;  // He_0, He_1
        double n = 1.0;
        const double tt = t * t;
        double term = t;
        double acc = t;
        int small = 0;
        for (int k = 1; k < 40; ++k) {
            double he_next = h * he_cur - n * he_prev;
            n += 1.0;
            he_prev = he_cur;
            he_cur = he_next;
            he_next = h * he_cur - n * he_prev;
            n += 1.0;
            he_prev = he_cur;
            he_cur = he_next;
            // he_prev is now He_{2k}
            const double kf = static_cast<double>(k);
            term *= tt / ((2.0 * kf) * (2.0 * kf + 1.0));
            const double c = he_prev * term;
            acc += c;
            // two consecutive negligible terms: He_{2k}(h) can vanish at isolated h
            small = std::fabs(c) <= 1e-17 * std::fabs(acc) ? small + 1 : 0;
            if (small == 2) break;
        }
        a = 2.0 * norm_pdf(h) * acc;
    } else if (std::fabs(d1) < 1.0 && std::fabs(d2) < 1.0) {
        a = 0.5 * (std::erf(d1 / kSqrt2) - std::erf(d2 / kSqrt2));
    } else {
        a = norm_cdf(d1) - norm_cdf(d2);
    }
    return std::cosh(0.5 * x) * a + std::sinh(0.5 * x) * (norm_cdf(d1) + norm_cdf(d2));
}

// b_max - b = e^{x/2} N(-d1) + e^{-x/2} N(d2): a sum, no cancellation.
inline double normalised_otm_call_complement(double x, double s) noexcept {
    const double d1 = x / s + 0.5 * s;
    const double d2 = x / s - 0.5 * s;
    return std::exp(0.5 * x) * norm_cdf(-d1) + std::exp(-0.5 * x) * norm_cdf(d2);
}

// Acklam inverse normal, no refinement: used for the initial guess only.
[[nodiscard]] inline double acklam_guess(double p) noexcept { return acklam_ppf(p); }

// Outcome of the noexcept solver entry points.
enum class IvStatus { Ok, AtBound, NoConvergence };

// Solve b(x, s) = beta for s. Requires beta > 0, u > 0, x <= 0. `evals` is
// incremented once per iteration. noexcept: writes the root to `out` and
// returns IvStatus::Ok, or returns IvStatus::NoConvergence.
inline IvStatus total_vol_nx(double beta, double u, double x, int& evals,
                             double& out) noexcept {
    const double b_max = std::exp(0.5 * x);
    const double s_c = std::sqrt(-2.0 * x);  // inflection point of b in s
    const double b_c = s_c > 0.0 ? normalised_otm_call(x, s_c) : 0.0;
    int mode;
    double ln_target, lo, hi, s;
    if (beta < b_c) {
        mode = 0;  // lower: f = 1/ln(beta) - 1/ln b(s)
        ln_target = std::log(beta);
        lo = 0.0;
        hi = s_c;
        // Mills-ratio asymptotic, y = s^2:
        // ln b ~ -x^2/(2y) - y/8 - ln sqrt(2pi) + 1.5 ln y - ln(x^2 - y^2/4)
        double y = x * x / (-2.0 * ln_target);
        for (int i = 0; i < 3; ++i) {
            if (!(y < -2.0 * x)) break;
            const double rhs = -ln_target - 0.125 * y - kLnSqrt2Pi + 1.5 * std::log(y) -
                               std::log(x * x - 0.25 * y * y);
            if (!(rhs > 0.0)) break;
            y = x * x / (2.0 * rhs);
        }
        s = std::min(std::sqrt(y), s_c);
    } else {
        lo = s_c;
        hi = std::numeric_limits<double>::infinity();
        if (beta <= 0.5 * b_max) {
            mode = 1;  // middle: f = ln b(s) - ln(beta)
            ln_target = std::log(beta);
        } else {
            mode = 2;  // upper: f = ln(u) - ln(b_max - b(s))
            ln_target = std::log(u);
        }
        const double p = u / (b_max + 1.0 / b_max);  // < 0.5 always
        s = std::max(-2.0 * acklam_guess(p), s_c);
        if (!(s > 0.0)) s = kSqrt2Pi * beta;
    }
    // The bracket ends start as the a-priori bounds (0 or s_c, s_c or inf);
    // an end is "probed" once the objective has been evaluated there.
    bool lo_probed = false;
    bool hi_probed = false;
    for (int it = 0; it < kMaxIter; ++it) {
        ++evals;
        const double d1 = x / s + 0.5 * s;
        const double bp = b_max * norm_pdf(d1);         // db/ds
        const double r2 = x * x / (s * s * s) - 0.25 * s;  // b''/b'
        double f, fp, fpp;
        if (mode == 0) {
            const double b = normalised_otm_call(x, s);
            if (!(b > 0.0)) {  // underflow: s is far too small
                lo = s;
                s = 0.5 * (lo + hi);
                continue;
            }
            const double l = std::log(b);
            const double lp = bp / b;
            const double lpp = lp * r2 - lp * lp;
            f = 1.0 / ln_target - 1.0 / l;
            fp = lp / (l * l);
            fpp = lpp / (l * l) - 2.0 * lp * lp / (l * l * l);
        } else if (mode == 1) {
            const double b = normalised_otm_call(x, s);
            const double lp = bp / b;
            f = std::log(b) - ln_target;
            fp = lp;
            fpp = lp * r2 - lp * lp;
        } else {
            const double w = normalised_otm_call_complement(x, s);
            const double gp = bp / w;
            f = ln_target - std::log(w);
            fp = gp;
            fpp = gp * r2 + gp * gp;
        }
        if (f > 0.0) {
            hi = std::min(hi, s);
            hi_probed = hi == s;
        } else if (f < 0.0) {
            lo = std::max(lo, s);
            lo_probed = lo == s;
        } else {
            out = s;
            return IvStatus::Ok;
        }
        const double newton = f / fp;
        const double den = 1.0 - 0.5 * newton * fpp / fp;
        double s_new = den > 0.5 ? s - newton / den : std::numeric_limits<double>::quiet_NaN();
        double tol;
        if (lo < s_new && s_new < hi) {
            tol = 1e-7;  // Halley: next error ~ step^3
        } else {
            s_new = s - newton;
            if (lo < s_new && s_new < hi) {
                tol = 1e-10;  // Newton: next error ~ step^2
            } else if (s_new >= hi && std::isfinite(hi) && !hi_probed) {
                s_new = hi;  // root lies next to s_c: probe it
                tol = 0.0;
            } else if (s_new <= lo && lo > 0.0 && !lo_probed) {
                s_new = lo;
                tol = 0.0;
            } else {
                s_new = std::isfinite(hi) ? 0.5 * (lo + hi) : 2.0 * s;
                tol = 0.0;
            }
        }
        if (std::fabs(s_new - s) <= std::max(tol, 4.0 * kEps) * s || (hi - lo) <= 4.0 * kEps * s) {
            out = s_new;
            return IvStatus::Ok;
        }
        s = s_new;
    }
    return IvStatus::NoConvergence;
}

// Throwing form: std::runtime_error on non-convergence.
inline double total_vol(double beta, double u, double x, int& evals) {
    double s = 0.0;
    if (total_vol_nx(beta, u, x, evals, s) != IvStatus::Ok) {
        throw std::runtime_error("implied vol: no convergence in 100 iterations");
    }
    return s;
}

// Black implied vol from a validated quote with lower < price < upper.
// noexcept: IvStatus::AtBound when the quote is within rounding of a bound.
inline IvStatus black_implied_vol_nx(double price, double F, double K, double T, double df,
                                     bool is_call, int& evals, double& out) noexcept {
    const double lower = df * std::max(is_call ? F - K : K - F, 0.0);
    const double upper = is_call ? df * F : df * K;
    const double norm = df * std::sqrt(F * K);
    const double beta = (price - lower) / norm;
    const double u = (upper - price) / norm;
    if (!(beta > 0.0 && u > 0.0)) [[unlikely]] return IvStatus::AtBound;
    const double x = -std::fabs(std::log1p((F - K) / K));  // F-K exact near the money
    double s = 0.0;
    const IvStatus st = total_vol_nx(beta, u, x, evals, s);
    if (st != IvStatus::Ok) [[unlikely]] return st;
    out = s / std::sqrt(T);
    return IvStatus::Ok;
}

// Throwing form: std::invalid_argument when the quote is within rounding
// of a bound, std::runtime_error on non-convergence.
inline double black_implied_vol(double price, double F, double K, double T, double df,
                                bool is_call, int& evals) {
    double out = 0.0;
    switch (black_implied_vol_nx(price, F, K, T, df, is_call, evals, out)) {
        case IvStatus::Ok:
            return out;
        case IvStatus::AtBound:
            throw std::invalid_argument("price is within rounding of a no-arbitrage bound");
        case IvStatus::NoConvergence:
            break;
    }
    throw std::runtime_error("implied vol: no convergence in 100 iterations");
}

}  // namespace qp::iv_core
