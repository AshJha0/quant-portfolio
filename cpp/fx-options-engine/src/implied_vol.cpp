#include "fxopt/implied_vol.hpp"

#include <algorithm>
#include <cmath>

#include "fxopt/garman_kohlhagen.hpp"
#include "brent.hpp"

namespace fxopt {

namespace {

// GK vega (dV/dsigma), same for calls and puts.
double gk_vega(double S, double K, double T, double r_d, double r_f,
               double sigma) {
    const double d1v = d1(S, K, T, r_d, r_f, sigma);
    return S * std::exp(-r_f * T) * norm_pdf(d1v) * std::sqrt(T);
}

}  // namespace

double implied_vol(double price, double S, double K, double T, double r_d,
                   double r_f, OptionType type, double tol, int max_iter) {
    const double p = phi(type);
    validate_inputs(S, K, T, r_d, r_f, 0.0);
    if (T <= 0.0) {
        throw std::invalid_argument("implied_vol requires T > 0");
    }
    detail::require_finite(price, "price");

    const double df_d = std::exp(-r_d * T);
    const double df_f = std::exp(-r_f * T);
    const double forward = S * df_f / df_d;
    const double lower = df_d * std::max(p * (forward - K), 0.0);  // sigma->0
    const double upper = p > 0.0 ? S * df_f : K * df_d;            // sigma->inf
    if (price < lower - 1e-14 || price > upper + 1e-14) {
        throw std::invalid_argument(
            "price " + std::to_string(price) +
            " outside no-arbitrage bounds [" + std::to_string(lower) + ", " +
            std::to_string(upper) + "]");
    }
    if (price - lower <= 1e-16 * std::max(1.0, lower)) {
        // Time value below double-precision resolution: vol unrecoverable,
        // return the sigma -> 0 limit (documented in docs/VALIDATION.md).
        return 0.0;
    }

    const auto objective = [&](double sig) {
        return gk_price(S, K, T, r_d, r_f, sig, type) - price;
    };

    // Newton with a moneyness-aware start (Brenner-Subrahmanyam flavoured).
    double sigma =
        std::max(0.05, std::sqrt(2.0 * std::abs(std::log(forward / K)) / T));
    const double lo = 1e-10;
    double hi = 10.0;
    for (int i = 0; i < max_iter; ++i) {
        const double diff = objective(sigma);
        if (std::abs(diff) < 1e-14) {
            return sigma;
        }
        const double vega = gk_vega(S, K, T, r_d, r_f, sigma);
        if (vega < 1e-12) {
            break;  // flat objective; Newton unreliable -> Brent
        }
        const double new_sigma = sigma - diff / vega;
        if (!(new_sigma > lo && new_sigma < hi)) {
            break;
        }
        if (std::abs(new_sigma - sigma) < tol) {
            return new_sigma;
        }
        sigma = new_sigma;
    }

    // Brent fallback on an expanding bracket.
    hi = 1.0;
    double f_hi = objective(hi);
    while (f_hi < 0.0 && hi < 50.0) {
        hi *= 2.0;
        f_hi = objective(hi);
    }
    if (f_hi < 0.0) {
        throw std::invalid_argument("implied vol > " + std::to_string(hi) +
                                    ": price " + std::to_string(price) +
                                    " unattainably high");
    }
    if (f_hi == 0.0) {
        // objective(hi) landed exactly on zero without ever going strictly
        // positive during expansion: deep ITM + long-dated + high vol drives
        // |d1|, |d2| large enough that N(d1)/N(d2) saturate to 0 or 1 in
        // double precision, so gk_price(sigma) is bit-identical to the
        // sigma -> inf bound for every sigma from the true root up to `hi`
        // (and beyond -- this is not a bracket, it is a flat plateau). Any
        // point in that plateau is an equally "valid" root of the floating
        // point objective, so accepting `hi` (an arbitrary artifact of the
        // doubling schedule) would silently return a vol that can be wrong
        // by whole vol points or more with no signal to the caller. This is
        // the upper-bound mirror of the near-`lower` short-circuit above;
        // unlike that case there is no finite limiting sigma to fall back
        // to (sigma -> infinity is not representable), so the honest
        // answer is that the vol is unrecoverable at this precision.
        throw std::invalid_argument(
            "price " + std::to_string(price) +
            " is within double-precision resolution of the sigma->inf "
            "bound " + std::to_string(upper) +
            "; implied volatility is unrecoverably large (vega has "
            "underflowed to zero in this regime -- see docs/VALIDATION.md)");
    }
    return detail::brentq(objective, lo, hi, tol, 200);
}

}  // namespace fxopt
