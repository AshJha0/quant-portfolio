// Dated OIS, multi-curve and cross-currency curve bootstrap (W13b).
//
// The C++ port of fi_rates.bootstrap.bootstrap_dated,
// fi_rates.multicurve.bootstrap_projection_dated and
// fx_rates.xccy.bootstrap_xccy_discount (and of rust/rates-curves). Same
// algorithm and loop order:
//
//   * curve time t = days / 365 (ACT/365F); ln P linear in t between pillars,
//     anchored at P(0) = 1; beyond the last pillar the last forward is flat;
//   * forward quote: (P(s)/P(e) - 1)/tau = R (deposits, FRAs, futures with
//     R = (100 - price)/100 - convexity);
//   * swap: sum_j D(p_j)(P(s_j)/P(e_j) - 1) = K sum_i tau_i D(p_i);
//   * cross-currency basis (constant notional, spread on the foreign leg):
//     V_f / X(s0) = V_d / D(s0);
//   * each pillar's ln P by bisection on [ln 1e-8, ln 20] until the bracket
//     cannot shrink.
//
// Oracle: QuantLib 1.44 through the shared fixture
// python/equity/04-fixed-income/tests/data/dated_bootstrap_usd.txt
// (tests/test_rates_curves.cpp). Header-only, std only.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace qp::rates {

inline constexpr double kDaysPerYear = 365.0;

class DfSource {
public:
    virtual ~DfSource() = default;
    [[nodiscard]] virtual double df(double day) const = 0;
};

// Log-linear discount curve on integer day pillars, flat forward beyond.
class DatedCurve final : public DfSource {
public:
    DatedCurve(std::vector<std::int64_t> days, std::vector<double> dfs)
        : days_(std::move(days)), dfs_(std::move(dfs)) {
        if (days_.empty() || days_.size() != dfs_.size()) {
            throw std::invalid_argument("days and dfs must be non-empty and of equal length");
        }
        std::int64_t prev = 0;
        for (auto d : days_) {
            if (d <= prev)
                throw std::invalid_argument("pillar days must be positive and strictly increasing");
            prev = d;
        }
        for (double x : dfs_) {
            if (!(std::isfinite(x) && x > 0.0))
                throw std::invalid_argument("discount factors must be finite and positive");
        }
        t_.reserve(days_.size() + 1);
        x_.reserve(days_.size() + 1);
        t_.push_back(0.0);
        x_.push_back(0.0);
        for (std::size_t i = 0; i < days_.size(); ++i) {
            t_.push_back(static_cast<double>(days_[i]) / kDaysPerYear);
            x_.push_back(std::log(dfs_[i]));
        }
    }

    [[nodiscard]] const std::vector<std::int64_t>& days() const noexcept { return days_; }
    [[nodiscard]] const std::vector<double>& dfs() const noexcept { return dfs_; }

    // ln P(day); NaN for a negative day.
    [[nodiscard]] double log_df(double day) const noexcept {
        if (!(day >= 0.0)) return std::numeric_limits<double>::quiet_NaN();
        const double t = day / kDaysPerYear;
        const std::size_t n = t_.size() - 1;
        if (t >= t_[n]) {
            const double slope = (x_[n] - x_[n - 1]) / (t_[n] - t_[n - 1]);
            return x_[n] + slope * (t - t_[n]);
        }
        std::size_t lo = 0;
        std::size_t hi = n;
        while (hi - lo > 1) {
            const std::size_t mid = (lo + hi) / 2;
            if (t_[mid] <= t) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        const double w = (t - t_[lo]) / (t_[hi] - t_[lo]);
        return x_[lo] + w * (x_[hi] - x_[lo]);
    }

    [[nodiscard]] double df(double day) const override { return std::exp(log_df(day)); }

    // Instantaneous forward just after `day` (piecewise flat).
    [[nodiscard]] double forward_rate(double day) const noexcept {
        const double t = day / kDaysPerYear;
        std::size_t k = t_.size() - 1;
        for (std::size_t i = 1; i < t_.size(); ++i) {
            if (t < t_[i]) {
                k = i;
                break;
            }
        }
        return -(x_[k] - x_[k - 1]) / (t_[k] - t_[k - 1]);
    }

private:
    std::vector<std::int64_t> days_;
    std::vector<double> dfs_;
    std::vector<double> t_;
    std::vector<double> x_;
};

struct FixedPeriod {
    std::int64_t pay;
    double accrual;
};

struct FloatPeriod {
    std::int64_t start;
    std::int64_t end;
    std::int64_t pay;
    double accrual;
};

struct ForwardQuote {
    std::int64_t start;
    std::int64_t end;
    double accrual;
    double rate;
};

struct SwapQuote {
    double rate;
    std::vector<FixedPeriod> fixed;
    std::vector<FloatPeriod> floating;
};

using Quote = std::variant<ForwardQuote, SwapQuote>;

[[nodiscard]] inline double futures_rate(double price, double convexity = 0.0) noexcept {
    return (100.0 - price) / 100.0 - convexity;
}

// Hull-White futures convexity bias (QuantLib HullWhite::convexityBias).
[[nodiscard]] inline double hull_white_convexity(double price, double t1, double t2, double sigma,
                                                 double a) {
    if (!(t2 > t1 && t1 >= 0.0) || sigma < 0.0 || a < 0.0) {
        throw std::invalid_argument("need 0 <= t1 < t2, sigma >= 0 and a >= 0");
    }
    const double dt = t2 - t1;
    double b_dt = dt;
    double b_t1 = t1;
    double var = 2.0 * t1;
    if (a >= 1e-12) {
        b_dt = (1.0 - std::exp(-a * dt)) / a;
        b_t1 = (1.0 - std::exp(-a * t1)) / a;
        var = (1.0 - std::exp(-2.0 * a * t1)) / a;
    }
    const double half_s2 = 0.5 * sigma * sigma;
    const double z = half_s2 * var * b_dt * b_dt + half_s2 * b_dt * b_t1 * b_t1;
    return (1.0 - std::exp(-z)) * ((100.0 - price) / 100.0 + 1.0 / dt);
}

// Model minus quoted rate.
[[nodiscard]] inline double quote_residual(const Quote& q, const DfSource& disc,
                                           const DfSource& proj) {
    if (const auto* f = std::get_if<ForwardQuote>(&q)) {
        return (proj.df(static_cast<double>(f->start)) / proj.df(static_cast<double>(f->end)) -
                1.0) /
                   f->accrual -
               f->rate;
    }
    const auto& s = std::get<SwapQuote>(q);
    double flt = 0.0;
    for (const auto& p : s.floating) {
        flt += disc.df(static_cast<double>(p.pay)) *
               (proj.df(static_cast<double>(p.start)) / proj.df(static_cast<double>(p.end)) - 1.0);
    }
    double ann = 0.0;
    for (const auto& p : s.fixed) ann += p.accrual * disc.df(static_cast<double>(p.pay));
    return flt / ann - s.rate;
}

// Bisection on [ln 1e-8, ln 20] to floating-point resolution: (root, iterations).
[[nodiscard]] inline std::pair<double, int> bisect_log_df(const std::function<double(double)>& f) {
    double lo = std::log(1e-8);
    double hi = std::log(20.0);
    double flo = f(lo);
    double fhi = f(hi);
    if (!(std::isfinite(flo) && std::isfinite(fhi)) || (flo > 0.0) == (fhi > 0.0)) {
        throw std::runtime_error("no sign change on the bracket");
    }
    int it = 0;
    while (it < 200) {
        const double mid = 0.5 * (lo + hi);
        if (mid <= lo || mid >= hi) break;
        ++it;
        const double fm = f(mid);
        if (fm == 0.0) return {mid, it};
        if ((fm > 0.0) == (flo > 0.0)) {
            lo = mid;
            flo = fm;
        } else {
            hi = mid;
            fhi = fm;
        }
    }
    return {std::abs(flo) <= std::abs(fhi) ? lo : hi, it};
}

struct Diagnostic {
    std::int64_t pillar;
    double df;
    double residual;
    int iterations;
};

struct BootstrapResult {
    DatedCurve curve;
    std::vector<Diagnostic> diagnostics;

    [[nodiscard]] double max_abs_residual() const noexcept {
        double m = 0.0;
        for (const auto& d : diagnostics) m = std::max(m, std::abs(d.residual));
        return m;
    }
};

namespace detail {

inline void validate(const Quote& q) {
    if (const auto* f = std::get_if<ForwardQuote>(&q)) {
        if (!(0 <= f->start && f->start < f->end) || !(f->accrual > 0.0) ||
            !std::isfinite(f->rate)) {
            throw std::invalid_argument("bad forward quote");
        }
        return;
    }
    const auto& s = std::get<SwapQuote>(q);
    if (!std::isfinite(s.rate) || s.fixed.empty() || s.floating.empty()) {
        throw std::invalid_argument("bad swap quote: need a finite rate and both legs");
    }
    for (const auto& p : s.fixed) {
        if (p.pay < 0 || !(p.accrual > 0.0)) throw std::invalid_argument("bad fixed period");
    }
    for (const auto& p : s.floating) {
        if (!(0 <= p.start && p.start < p.end) || p.pay < 0 || !(p.accrual > 0.0)) {
            throw std::invalid_argument("bad float period");
        }
    }
}

inline std::int64_t pillar(const Quote& q, bool self_proj, bool self_disc) {
    if (const auto* f = std::get_if<ForwardQuote>(&q)) return f->end;
    const auto& s = std::get<SwapQuote>(q);
    std::int64_t m = std::numeric_limits<std::int64_t>::min();
    if (self_proj) {
        for (const auto& p : s.floating) m = std::max(m, p.end);
    }
    if (self_disc) {
        for (const auto& p : s.floating) m = std::max(m, p.pay);
        for (const auto& p : s.fixed) m = std::max(m, p.pay);
    }
    return m;
}

// Sequential solve shared by both bootstraps. residual(trial, quote index).
template <class Residual>
BootstrapResult run(const std::vector<std::pair<std::int64_t, std::size_t>>& keyed,
                    Residual residual) {
    for (std::size_t k = 1; k < keyed.size(); ++k) {
        if (keyed[k - 1].first == keyed[k].first) {
            throw std::invalid_argument("two quotes share pillar day " +
                                        std::to_string(keyed[k].first));
        }
    }
    if (keyed.front().first <= 0)
        throw std::invalid_argument("pillars must be after the valuation date");
    std::vector<std::int64_t> days;
    std::vector<double> dfs;
    std::vector<int> iters;
    for (const auto& [p, i] : keyed) {
        auto f = [&, p = p, i = i](double x) {
            auto d = days;
            d.push_back(p);
            auto v = dfs;
            v.push_back(std::exp(x));
            const DatedCurve trial(std::move(d), std::move(v));
            return residual(trial, i);
        };
        std::pair<double, int> r;
        try {
            r = bisect_log_df(f);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("bootstrap failed at pillar day " + std::to_string(p) + ": " +
                                     e.what());
        }
        days.push_back(p);
        dfs.push_back(std::exp(r.first));
        iters.push_back(r.second);
    }
    DatedCurve curve(days, dfs);
    std::vector<Diagnostic> diags;
    diags.reserve(keyed.size());
    for (std::size_t k = 0; k < keyed.size(); ++k) {
        diags.push_back(
            {keyed[k].first, curve.dfs()[k], residual(curve, keyed[k].second), iters[k]});
    }
    return {std::move(curve), std::move(diags)};
}

}  // namespace detail

// Bootstrap a dated curve. nullptr for discount/projection means the curve
// being built (an OIS curve passes both as nullptr).
[[nodiscard]] inline BootstrapResult bootstrap_dated(const std::vector<Quote>& quotes,
                                                     const DfSource* discount = nullptr,
                                                     const DfSource* projection = nullptr) {
    if (quotes.empty()) throw std::invalid_argument("no quotes supplied to bootstrap");
    if (discount != nullptr && projection != nullptr) {
        throw std::invalid_argument(
            "at least one of discount/projection must be the curve being built");
    }
    std::vector<std::pair<std::int64_t, std::size_t>> keyed;
    for (std::size_t i = 0; i < quotes.size(); ++i) {
        detail::validate(quotes[i]);
        keyed.emplace_back(detail::pillar(quotes[i], projection == nullptr, discount == nullptr),
                           i);
    }
    std::sort(keyed.begin(), keyed.end());
    return detail::run(keyed, [&](const DatedCurve& c, std::size_t i) {
        return quote_residual(quotes[i], discount != nullptr ? *discount : c,
                              projection != nullptr ? *projection : c);
    });
}

struct XccyBasisQuote {
    double basis;
    std::int64_t start;
    std::int64_t end;
    std::vector<FloatPeriod> foreign;
    std::vector<FloatPeriod> domestic;

    [[nodiscard]] std::int64_t pillar() const noexcept {
        std::int64_t m = end;
        for (const auto& p : foreign) m = std::max(m, p.pay);
        return m;
    }
};

namespace detail {
inline std::pair<double, double> leg_value(const std::vector<FloatPeriod>& periods, double spread,
                                           std::int64_t start, std::int64_t end,
                                           const DfSource& disc, const DfSource& proj) {
    double v = -disc.df(static_cast<double>(start)) + disc.df(static_cast<double>(end));
    double ann = 0.0;
    for (const auto& p : periods) {
        const double d = disc.df(static_cast<double>(p.pay));
        v += d * (proj.df(static_cast<double>(p.start)) / proj.df(static_cast<double>(p.end)) -
                  1.0 + spread * p.accrual);
        ann += p.accrual * d;
    }
    return {v, ann};
}
}  // namespace detail

// Par error in spread units: (V_f/X(s0) - V_d/D(s0)) / A_f.
[[nodiscard]] inline double xccy_basis_residual(const XccyBasisQuote& q, const DfSource& fdisc,
                                                const DfSource& fproj, const DfSource& dom) {
    const auto [vf, af] = detail::leg_value(q.foreign, q.basis, q.start, q.end, fdisc, fproj);
    const auto [vd, ad] = detail::leg_value(q.domestic, 0.0, q.start, q.end, dom, dom);
    static_cast<void>(ad);
    const double x0 = fdisc.df(static_cast<double>(q.start));
    const double d0 = dom.df(static_cast<double>(q.start));
    return (vf / x0 - vd / d0) / (af / x0);
}

// Foreign discount curve under domestic collateral from basis swaps.
[[nodiscard]] inline BootstrapResult bootstrap_xccy_discount(
    const std::vector<XccyBasisQuote>& quotes, const DfSource& foreign_projection,
    const DfSource& domestic) {
    if (quotes.empty()) throw std::invalid_argument("no basis quotes supplied");
    std::vector<std::pair<std::int64_t, std::size_t>> keyed;
    for (std::size_t i = 0; i < quotes.size(); ++i) {
        const auto& q = quotes[i];
        if (!std::isfinite(q.basis) || !(0 <= q.start && q.start < q.end) || q.foreign.empty() ||
            q.domestic.empty()) {
            throw std::invalid_argument("bad basis quote");
        }
        keyed.emplace_back(q.pillar(), i);
    }
    std::sort(keyed.begin(), keyed.end());
    return detail::run(keyed, [&](const DatedCurve& c, std::size_t i) {
        return xccy_basis_residual(quotes[i], c, foreign_projection, domestic);
    });
}

}  // namespace qp::rates
