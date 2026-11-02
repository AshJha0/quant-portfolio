// Bachelier model; see include/eqopt/bachelier.hpp.
#include "eqopt/bachelier.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace eqopt {
namespace {

void check(double F, double K, double T, double sigma_n, double r) {
    if (!std::isfinite(F) || !std::isfinite(K) || !std::isfinite(T) || !std::isfinite(sigma_n) ||
        !std::isfinite(r))
        throw std::invalid_argument("bachelier: inputs must be finite");
    if (T < 0.0 || sigma_n < 0.0)
        throw std::invalid_argument("bachelier: T and sigma_n must be >= 0");
}

}  // namespace

double bachelier_price(double F, double K, double T, double sigma_n, double r, OptionType type) {
    check(F, K, T, sigma_n, r);
    const double df = std::exp(-r * T);
    const double sign = type == OptionType::Call ? 1.0 : -1.0;
    const double s = sigma_n * std::sqrt(T);
    if (s == 0.0) return df * std::fmax(sign * (F - K), 0.0);
    const double d = (F - K) / s;
    return df * (sign * (F - K) * norm_cdf(sign * d) + s * norm_pdf(d));
}

BachelierGreeks bachelier_greeks(double F, double K, double T, double sigma_n, double r,
                                 OptionType type) {
    check(F, K, T, sigma_n, r);
    if (T <= 0.0 || sigma_n <= 0.0)
        throw std::invalid_argument("bachelier_greeks: need T > 0 and sigma_n > 0");
    const double df = std::exp(-r * T);
    const double sq = std::sqrt(T);
    const double s = sigma_n * sq;
    const double d = (F - K) / s;
    const double pdf = norm_pdf(d);
    BachelierGreeks g{};
    g.price = bachelier_price(F, K, T, sigma_n, r, type);
    g.delta = type == OptionType::Call ? df * norm_cdf(d) : -df * norm_cdf(-d);
    g.gamma = df * pdf / s;
    g.vega = df * sq * pdf;
    g.theta = r * g.price - df * sigma_n * pdf / (2.0 * sq);
    return g;
}

double bachelier_implied_vol(double price, double F, double K, double T, double r,
                             OptionType type) {
    check(F, K, T, 0.0, r);
    if (!std::isfinite(price) || T <= 0.0)
        throw std::invalid_argument("bachelier_implied_vol: need a finite price and T > 0");
    const double df = std::exp(-r * T);
    double sign = type == OptionType::Call ? 1.0 : -1.0;
    if (!(price > df * std::fmax(sign * (F - K), 0.0)))
        throw std::invalid_argument("bachelier_implied_vol: price at or below intrinsic");
    if (sign * (F - K) > 0.0) {  // invert the out-of-the-money option via parity
        price -= df * sign * (F - K);
        sign = -sign;
        type = sign > 0.0 ? OptionType::Call : OptionType::Put;
    }
    double lo = 0.0;
    double hi = std::fmax(std::fabs(F - K), 1.0) / std::sqrt(T);
    while (bachelier_price(F, K, T, hi, r, type) < price) {
        hi *= 2.0;
        if (hi > 1e300) throw std::invalid_argument("bachelier_implied_vol: price too high");
    }
    double sig = std::fmin(price / df * std::sqrt(2.0 * std::numbers::pi / T), hi);
    for (int it = 0; it < 200; ++it) {
        const double diff = bachelier_price(F, K, T, sig, r, type) - price;
        if (std::fabs(diff) <= 4.0 * std::numeric_limits<double>::epsilon() * price) return sig;
        if (diff > 0.0)
            hi = sig;
        else
            lo = sig;
        const double d =
            sig > 0.0 ? (F - K) / (sig * std::sqrt(T)) : std::numeric_limits<double>::infinity();
        const double vega = df * std::sqrt(T) * norm_pdf(d);
        const double step =
            vega > 0.0 ? sig - diff / vega : std::numeric_limits<double>::quiet_NaN();
        const double next = (step >= lo && step <= hi) ? step : 0.5 * (lo + hi);
        if (std::fabs(next - sig) <= 1e-14 * next) return next;
        sig = next;
    }
    return sig;
}

}  // namespace eqopt
