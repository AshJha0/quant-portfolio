// Collateralised exposure under a CSA: the margin balance, the margin period of
// risk lag and pathwise exposure.
//
// C++ port of python/core/src/qp_core/xva/collateral.py; twin of
// rust/core/src/collateral.rs. Plain idea, formal statement, worked examples and
// sources (Gregory, The xVA Challenge, ch. 8; BCBS CRE53) are in the Python
// docstring. Matrices are row-major paths x times.
//
// Parity: cpp/core/tests/test_frtb_collateral.cpp reproduces the Python
// reference values (rust/core/tests/data/frtb_collateral_parity.txt, shared with
// Rust) to 1e-12 relative.
//
// Differences from the Python, none of which changes a value on valid input:
// NaN is rejected here, as is an infinite value, time or initial; infinite
// thresholds, MTA, MPOR and margin stay legal, because H = infinity is the
// documented uncollateralised limit. Csa is checked when it is used, not when it
// is built. Python's broadcastable margin arrays are the explicit Margin shapes.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qp::collateral {

inline constexpr double kEps = 1e-12;

// Initial margin, in the array shapes Python's broadcast is used with.
struct Margin {
    enum class Shape { Scalar, ByTime, ByPath, Full };
    Shape shape = Shape::Scalar;
    std::vector<double> values{0.0};

    [[nodiscard]] static Margin scalar(double x) { return {Shape::Scalar, {x}}; }
    [[nodiscard]] static Margin by_time(std::vector<double> x) {
        return {Shape::ByTime, std::move(x)};
    }
    [[nodiscard]] static Margin by_path(std::vector<double> x) {
        return {Shape::ByPath, std::move(x)};
    }
    [[nodiscard]] static Margin full(std::vector<double> x) { return {Shape::Full, std::move(x)}; }

    void check(std::size_t paths, std::size_t times) const {
        std::size_t want = 1;
        if (shape == Shape::ByTime) want = times;
        if (shape == Shape::ByPath) want = paths;
        if (shape == Shape::Full) want = paths * times;
        if (values.size() != want)
            throw std::invalid_argument("initial margin does not match (paths, times)");
        for (double x : values)
            if (!(x >= 0.0))  // written so that NaN is refused too
                throw std::invalid_argument("initial margin must be non-negative");
    }

    [[nodiscard]] double at(std::size_t path, std::size_t time, std::size_t times) const {
        switch (shape) {
            case Shape::ByTime:
                return values[time];
            case Shape::ByPath:
                return values[path];
            case Shape::Full:
                return values[path * times + time];
            case Shape::Scalar:
                break;
        }
        return values[0];
    }
};

// CSA terms. mpor is in the grid's time unit (years: 10 business days is about
// 10/250). initial is the balance at t_0.
struct Csa {
    double threshold_cpty = 0.0;
    double threshold_bank = 0.0;
    double mta = 0.0;
    double mpor = 0.0;
    Margin im_received{};
    Margin im_posted{};
    double initial = 0.0;

    void validate() const {
        for (double x : {threshold_cpty, threshold_bank, mta, mpor})
            if (!(x >= 0.0))  // written so that NaN is refused too
                throw std::invalid_argument("thresholds, MTA and MPOR must be non-negative");
        if (!std::isfinite(initial)) throw std::invalid_argument("initial must be finite");
    }

    // Two-way target collateral T(V) (positive: the bank holds it).
    [[nodiscard]] double target_vm(double v) const noexcept {
        return std::max(v - threshold_cpty, 0.0) - std::max(-v - threshold_bank, 0.0);
    }
};

namespace detail {

inline void check_finite_times(std::span<const double> times) {
    if (times.empty()) throw std::invalid_argument("times must not be empty");
    for (double t : times)
        if (!std::isfinite(t)) throw std::invalid_argument("times must be finite");
}

inline void check_times(std::span<const double> times) {
    check_finite_times(times);
    for (std::size_t i = 1; i < times.size(); ++i)
        if (times[i] <= times[i - 1])
            throw std::invalid_argument("times must be strictly increasing");
}

}  // namespace detail

// Balance B(t_k) per path after each margin call, with the MTA rule.
[[nodiscard]] inline std::vector<double> collateral_balance(std::span<const double> values,
                                                            std::size_t paths, std::size_t times,
                                                            const Csa& csa) {
    csa.validate();
    if (values.size() != paths * times)
        throw std::invalid_argument("values must be (paths, len(times))");
    for (double v : values)
        if (!std::isfinite(v)) throw std::invalid_argument("values must be finite");
    std::vector<double> out(values.size());
    for (std::size_t p = 0; p < paths; ++p) {
        double cur = csa.initial;
        for (std::size_t k = 0; k < times; ++k) {
            const double tgt = csa.target_vm(values[p * times + k]);
            if (csa.mta <= 0.0 || std::fabs(tgt - cur) >= csa.mta) cur = tgt;
            out[p * times + k] = cur;
        }
    }
    return out;
}

// times with every t - mpor >= t_0 added, rounded to 12 decimals, sorted and
// de-duplicated, so the MPOR lag lands exactly on grid dates.
[[nodiscard]] inline std::vector<double> lagged_grid(std::span<const double> times, double mpor) {
    detail::check_finite_times(times);
    if (std::isnan(mpor)) throw std::invalid_argument("mpor must not be NaN");
    std::vector<double> all(times.begin(), times.end());
    for (double t : times) {
        const double lag = t - mpor;
        if (lag >= times[0] - kEps) all.push_back(lag);
    }
    // numpy's round(x, 12): scale, round half to even (the default rounding
    // mode), unscale.
    for (double& x : all) x = std::nearbyint(x * 1e12) / 1e12;
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    return all;
}

// For each date, the index of the latest grid date <= t - mpor, or 0 if none.
[[nodiscard]] inline std::vector<std::size_t> lag_index(std::span<const double> times,
                                                        double mpor) {
    detail::check_times(times);
    if (std::isnan(mpor)) throw std::invalid_argument("mpor must not be NaN");
    std::vector<std::size_t> out;
    out.reserve(times.size());
    for (double t : times) {
        const auto hi = std::upper_bound(times.begin(), times.end(), t - mpor + kEps);
        const auto n = static_cast<std::size_t>(hi - times.begin());
        out.push_back(n == 0 ? 0 : n - 1);
    }
    return out;
}

// Pathwise collateral C(t), exposure E(t) and negative exposure N(t), row-major
// paths x times.
struct CollateralisedExposure {
    std::vector<double> collateral;
    std::vector<double> exposure;
    std::vector<double> negative_exposure;
};

[[nodiscard]] inline CollateralisedExposure collateralised_exposure(std::span<const double> times,
                                                                    std::span<const double> values,
                                                                    std::size_t paths,
                                                                    const Csa& csa) {
    detail::check_times(times);
    const std::size_t n = times.size();
    const auto bal = collateral_balance(values, paths, n, csa);
    csa.im_received.check(paths, n);
    csa.im_posted.check(paths, n);
    auto c = bal;
    if (csa.mpor > 0.0) {
        const auto lagged = lag_index(times, csa.mpor);
        for (std::size_t p = 0; p < paths; ++p)
            for (std::size_t k = 0; k < n; ++k)
                c[p * n + k] =
                    times[k] - csa.mpor < times[0] - kEps ? csa.initial : bal[p * n + lagged[k]];
    }
    std::vector<double> e(values.size()), neg(values.size());
    for (std::size_t p = 0; p < paths; ++p)
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t i = p * n + k;
            e[i] = std::max(values[i] - c[i] - csa.im_received.at(p, k, n), 0.0);
            neg[i] = std::min(values[i] - c[i] + csa.im_posted.at(p, k, n), 0.0);
        }
    return {std::move(c), std::move(e), std::move(neg)};
}

}  // namespace qp::collateral
