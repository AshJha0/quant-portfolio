#include "eqopt/implied_vol.hpp"

#include <cmath>

#include "eqopt/black_scholes.hpp"
#include "eqopt/iv_core.hpp"

namespace eqopt {

Result<double> try_implied_vol(double price, double S, double K, double T, double r,
                               double q, OptionType type) noexcept {
    if (auto e = try_validate_inputs(S, K, T, 0.0); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    if (auto e = try_validate_rates(r, q); !e) [[unlikely]] return std::unexpected(e.error());
    if (!std::isfinite(price)) [[unlikely]] return std::unexpected(Error{Errc::PriceNotFinite});
    if (T <= 0.0) [[unlikely]] return std::unexpected(Error{Errc::Expired});
    if (S <= 0.0 || K <= 0.0) [[unlikely]] {
        return std::unexpected(Error{Errc::NonPositiveSpotStrike});
    }

    // No-arbitrage bounds: sigma -> 0 (discounted forward intrinsic) and
    // sigma -> inf.
    const auto lower = try_bs_price(S, K, T, r, 0.0, q, type);
    if (!lower) [[unlikely]] return lower;
    const auto df = try_finite(std::exp(-r * T), "implied_vol");
    if (!df) [[unlikely]] return df;
    double upper = K * *df;
    if (type == OptionType::Call) {
        const auto u = try_finite(S * std::exp(-q * T), "implied_vol");
        if (!u) [[unlikely]] return u;
        upper = *u;
    }
    if (price <= *lower) [[unlikely]] {
        return std::unexpected(Error{Errc::BelowLowerBound, "", price, *lower});
    }
    if (price >= upper) [[unlikely]] {
        return std::unexpected(Error{Errc::AboveUpperBound, "", price, upper});
    }
    const auto forward = try_finite(S * std::exp((r - q) * T), "implied_vol");
    if (!forward) [[unlikely]] return forward;
    int evals = 0;
    double sigma = 0.0;
    switch (iv_core::black_implied_vol_nx(price, *forward, K, T, *df,
                                          type == OptionType::Call, evals, sigma)) {
        case iv_core::IvStatus::Ok:
            return sigma;
        case iv_core::IvStatus::AtBound:
            return std::unexpected(Error{Errc::AtBound});
        case iv_core::IvStatus::NoConvergence:
            break;
    }
    return std::unexpected(Error{Errc::NoConvergence});
}

double implied_vol(double price, double S, double K, double T, double r,
                   double q, OptionType type) {
    return value_or_throw(try_implied_vol(price, S, K, T, r, q, type));
}

double implied_vol(double price, double S, double K, double T, double r,
                   double q, OptionType type, double /*tol*/, int /*max_iter*/,
                   double /*sigma_lo*/, double /*sigma_hi*/) {
    return implied_vol(price, S, K, T, r, q, type);
}

}  // namespace eqopt
