#include "fxopt/implied_vol.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "fxopt/iv_core.hpp"

namespace fxopt {

Result<double> try_implied_vol(double price, double S, double K, double T, double r_d, double r_f,
                               OptionType type) noexcept {
    const double p = phi(type);
    if (auto e = try_validate_inputs(S, K, T, r_d, r_f, 0.0); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    if (T <= 0.0) [[unlikely]]
        return std::unexpected(Error{Errc::IvExpired});
    if (auto e = try_require_finite(price, "price"); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }

    const auto df_d = try_finite_result(std::exp(-r_d * T), "implied_vol");
    if (!df_d) [[unlikely]]
        return df_d;
    const auto df_f = try_finite_result(std::exp(-r_f * T), "implied_vol");
    if (!df_f) [[unlikely]]
        return df_f;
    const auto fwd = try_finite_result(S * *df_f / *df_d, "forward");
    if (!fwd) [[unlikely]]
        return fwd;
    const double forward = *fwd;
    const double lower = *df_d * std::max(p * (forward - K), 0.0);  // sigma->0
    const double upper = p > 0.0 ? S * *df_f : K * *df_d;           // sigma->inf
    constexpr double eps = std::numeric_limits<double>::epsilon();
    // Rounding of the sigma -> 0 bound df (F - K): a few eps of df max(F, K)
    // when in the money; exactly zero otherwise. Same rule in all engines.
    const double tau_lo = lower > 0.0 ? 4.0 * eps * *df_d * std::max(forward, K) : 0.0;
    const double tau = 4.0 * eps * upper;
    if (price < lower - tau_lo || price > upper + tau) [[unlikely]] {
        return std::unexpected(Error{Errc::IvOutsideBounds, "", price, lower, upper});
    }
    if (price - lower <= tau_lo) {
        // Time value below the rounding of the bound: the sigma -> 0 limit.
        return 0.0;
    }
    if (upper - price <= tau) [[unlikely]] {
        return std::unexpected(Error{Errc::IvNearUpperBound, "", price, tau, upper});
    }
    int evals = 0;
    double sigma = 0.0;
    switch (iv_core::black_implied_vol_nx(price, forward, K, T, *df_d, p > 0.0, evals, sigma)) {
        case iv_core::IvStatus::Ok:
            return sigma;
        case iv_core::IvStatus::AtBound:
            return std::unexpected(Error{Errc::AtBound});
        case iv_core::IvStatus::NoConvergence:
            break;
    }
    return std::unexpected(Error{Errc::NoConvergence});
}

double implied_vol(double price, double S, double K, double T, double r_d, double r_f,
                   OptionType type) {
    return value_or_throw(try_implied_vol(price, S, K, T, r_d, r_f, type));
}

}  // namespace fxopt
