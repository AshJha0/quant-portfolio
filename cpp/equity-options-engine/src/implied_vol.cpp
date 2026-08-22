#include "eqopt/implied_vol.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "eqopt/black_scholes.hpp"

namespace eqopt {

namespace {

/// Analytic vega dV/dsigma used inside the Newton iteration.
double bs_vega(double S, double K, double T, double r, double sigma,
               double q) {
    const auto [d1, d2] = d1_d2(S, K, T, r, sigma, q);
    (void)d2;
    return S * std::exp(-q * T) * norm_pdf(d1) * std::sqrt(T);
}

}  // namespace

double implied_vol(double price, double S, double K, double T, double r,
                   double q, OptionType type, double tol, int max_iter,
                   double sigma_lo, double sigma_hi) {
    validate_inputs(S, K, T, 0.0);
    validate_rates(r, q);
    if (std::isnan(price)) {
        throw std::invalid_argument("price must not be NaN");
    }
    if (T <= 0.0) {
        throw std::invalid_argument(
            "implied_vol requires T > 0 (option already expired)");
    }
    if (S <= 0.0 || K <= 0.0) {
        throw std::invalid_argument("implied_vol requires S > 0 and K > 0");
    }

    // No-arbitrage bounds: sigma -> 0 (discounted forward intrinsic) and
    // sigma -> inf.
    const double lower = bs_price(S, K, T, r, 0.0, q, type);
    const double upper = type == OptionType::Call ? S * std::exp(-q * T)
                                                  : K * std::exp(-r * T);
    if (price <= lower) {
        std::ostringstream oss;
        oss << "price " << price
            << " is at or below the sigma->0 arbitrage bound " << lower
            << "; implied vol is undefined";
        throw std::invalid_argument(oss.str());
    }
    if (price >= upper) {
        std::ostringstream oss;
        oss << "price " << price << " is at or above the sigma->inf bound "
            << upper << "; implied vol is undefined";
        throw std::invalid_argument(oss.str());
    }

    const auto objective = [&](double sig) {
        return bs_price(S, K, T, r, sig, q, type) - price;
    };

    double lo = sigma_lo;
    double hi = sigma_hi;
    const double f_lo = objective(lo);
    double f_hi = objective(hi);
    // Expand the top of the bracket if needed (extremely high premiums).
    while (f_hi < 0.0 && hi < 1e3) {
        hi *= 2.0;
        f_hi = objective(hi);
    }
    if (f_lo > 0.0 || f_hi < 0.0) {
        throw std::invalid_argument(
            "failed to bracket implied volatility (solver cap ~1e3 vol)");
    }

    // Bracketed Newton: start from the midpoint, never leave [lo, hi];
    // fall back to bisection whenever Newton stalls or steps outside.
    double sigma = 0.5 * (lo + hi);
    for (int i = 0; i < max_iter; ++i) {
        const double diff = objective(sigma);
        if (std::fabs(diff) < tol) {
            break;
        }
        if (diff > 0.0) {
            hi = sigma;
        } else {
            lo = sigma;
        }
        double candidate;
        const double vega = bs_vega(S, K, T, r, sigma, q);
        if (vega > 1e-14) {
            candidate = sigma - diff / vega;
        } else {
            candidate = std::nan("");
        }
        if (!(lo < candidate && candidate < hi)) {
            candidate = 0.5 * (lo + hi);  // bisection fallback
        }
        if (std::fabs(candidate - sigma) < 1e-16) {
            break;
        }
        sigma = candidate;
    }

    // Final safeguard: pure bisection on the maintained bracket, run to
    // full bracket precision rather than stopping as soon as the price
    // residual satisfies `tol`. `tol` alone is not a reliable stopping
    // rule: in a flat-vega region (very long-dated + very high vol, where
    // d1/d2 blow up and vega ~ S sqrt(T) phi(d1) underflows towards zero
    // near the bracket's arbitrage bound) a tiny price residual can map
    // through that near-zero vega to a sigma residual of whole vol points,
    // so exiting the moment `|f_mid| < tol` risks returning an
    // insufficiently refined sigma. Bisecting the bracket itself down to
    // double-precision width is the only stopping rule that is safe in
    // that regime too, and costs at most ~50 extra evaluations elsewhere.
    for (int i = 0; i < 200 && hi - lo > 1e-15 * std::fmax(hi, 1.0); ++i) {
        const double mid = 0.5 * (lo + hi);
        const double f_mid = objective(mid);
        if (f_mid > 0.0) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return 0.5 * (lo + hi);
}

}  // namespace eqopt
