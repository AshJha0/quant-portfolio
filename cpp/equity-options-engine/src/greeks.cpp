#include "eqopt/greeks.hpp"

#include <cmath>

#include "eqopt/black_scholes.hpp"

namespace eqopt {

BSGreeks bs_greeks(double S, double K, double T, double r, double sigma,
                   double q, OptionType type) {
    const auto [d1, d2] = d1_d2(S, K, T, r, sigma, q);
    const double sqrt_t = std::sqrt(T);
    const double df_q = std::exp(-q * T);
    const double df_r = std::exp(-r * T);
    const double pdf_d1 = norm_pdf(d1);

    BSGreeks g{};
    g.gamma = df_q * pdf_d1 / (S * sigma * sqrt_t);
    g.vega = S * df_q * pdf_d1 * sqrt_t;
    g.vanna = -df_q * pdf_d1 * d2 / sigma;
    g.volga = g.vega * d1 * d2 / sigma;
    const double common_theta = -S * df_q * pdf_d1 * sigma / (2.0 * sqrt_t);

    if (type == OptionType::Call) {
        const double nd1 = norm_cdf(d1);
        const double nd2 = norm_cdf(d2);
        g.price = S * df_q * nd1 - K * df_r * nd2;
        g.delta = df_q * nd1;
        g.theta = common_theta + q * S * df_q * nd1 - r * K * df_r * nd2;
        g.rho = K * T * df_r * nd2;
    } else {
        const double nmd1 = norm_cdf(-d1);
        const double nmd2 = norm_cdf(-d2);
        g.price = K * df_r * nmd2 - S * df_q * nmd1;
        g.delta = -df_q * nmd1;
        g.theta = common_theta - q * S * df_q * nmd1 + r * K * df_r * nmd2;
        g.rho = -K * T * df_r * nmd2;
    }
    return g;
}

}  // namespace eqopt
