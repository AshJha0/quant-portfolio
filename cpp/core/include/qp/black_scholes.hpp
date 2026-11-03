// qp/black_scholes.hpp: Black-Scholes in cost-of-carry form, shared by the
// equity (Black-Scholes-Merton) and FX (Garman-Kohlhagen) options engines.
//
// The plain idea. A stock paying a continuous dividend yield q and a
// currency earning the foreign rate r_f are the same underlying to a
// pricer: both grow at the cost of carry b (b = r - q for a stock,
// b = r_d - r_f for a currency) and the payoff is discounted at r (r_d).
// Garman and Kohlhagen (1983) made exactly this observation, so one kernel
// prices both, and eqopt and fxopt keep only their own names, validation
// and boundary rules.
//
// The formal statement. With phi = +1 for a call, -1 for a put and the
// carry yield q = r - b:
//
//   d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T),  d2 = d1 - sigma sqrt T
//   V  = phi [S e^{-qT} N(phi d1) - K e^{-rT} N(phi d2)]
//   delta = phi e^{-qT} N(phi d1)            delta_forward = phi N(phi d1)
//   gamma = e^{-qT} n(d1) / (S sigma sqrt T) vega  = S e^{-qT} n(d1) sqrt T
//   theta = -S e^{-qT} n(d1) sigma / (2 sqrt T) + phi [q S e^{-qT} N(phi d1) - r K e^{-rT} N(phi d2)]
//   rho   = phi K T e^{-rT} N(phi d2)        rho_q = -phi S T e^{-qT} N(phi d1)
//   vanna = -e^{-qT} n(d1) d2 / sigma        volga = vega d1 d2 / sigma
//
// The functions take r and q rather than b so the two discount factors are
// computed from the caller's own numbers (recovering q as r - b would round
// and move the engines' golden values in the last bit); d1 uses r - q = b.
//
// Worked example. S = K = 100, T = 1, r = 5 %, q = 2 %, sigma = 20 %:
// d1 = (0 + 0.03 + 0.02) / 0.2 = 0.25, d2 = 0.05, and the call is
// 100 e^{-0.02} N(0.25) - 100 e^{-0.05} N(0.05) = 9.2270 to four decimals.
// The same number prices a currency call with S = K = 100, r_d = 5 %,
// r_f = 2 % (tests/test_core.cpp, "carry").
//
// Every function here is noexcept and unchecked: S, K, T, sigma > 0 and
// finite rates. Validation and the boundary branches (T == 0, sigma == 0,
// and for equity S == 0 or K == 0) stay in the engines, whose domains differ.
//
// Floating point. The expressions are evaluated in a fixed order and both
// engines return the values they returned before sharing this header, bit
// for bit. The engines have always differed in one place: the equity engine
// evaluates N(x) as 0.5 erfc(-x * (1/sqrt 2)) (norm_cdf_rcp), the FX engine
// as 0.5 erfc(-x / sqrt 2) (norm_cdf); the two can differ in the last bit,
// so the normal cdf is a template parameter. The fast variants in
// qp/black_scholes_vm.hpp use the branch-free vmath kernels instead.

#pragma once

#include <cmath>

#include "qp/normal.hpp"

namespace qp {

/// N(x) as 0.5 erfc(-x / sqrt 2): the FX engines' form.
struct NormCdfDiv {
    [[nodiscard]] static double eval(double x) noexcept { return norm_cdf(x); }
};

/// N(x) as 0.5 erfc(-x * (1 / sqrt 2)): the equity engines' form.
struct NormCdfRcp {
    [[nodiscard]] static double eval(double x) noexcept { return norm_cdf_rcp(x); }
};

/// d1 and d2 of the carry form.
struct CarryD {
    double d1;
    double d2;
};

/// d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T), d2 = d1 - sigma sqrt T.
[[nodiscard]] inline CarryD carry_d1_d2(double S, double K, double T, double r, double q,
                                        double sigma) noexcept {
    const double vol_sqrt_t = sigma * std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t;
    return {d1, d1 - vol_sqrt_t};
}

/// phi [disc_s N(phi d1) - disc_k N(phi d2)] from precomputed terms.
template <class Cdf = NormCdfDiv>
[[nodiscard]] inline double carry_value(double disc_s, double disc_k, double d1, double d2,
                                        double phi) noexcept {
    return phi * (disc_s * Cdf::eval(phi * d1) - disc_k * Cdf::eval(phi * d2));
}

/// Price for S, K, T, sigma > 0. phi is +1 (call) or -1 (put). May return a
/// non-finite value when a discount factor overflows; the caller checks.
template <class Cdf = NormCdfDiv>
[[nodiscard]] inline double carry_price(double S, double K, double T, double r, double q,
                                        double sigma, double phi) noexcept {
    const double disc_s = S * std::exp(-q * T);
    const double disc_k = K * std::exp(-r * T);
    const CarryD d = carry_d1_d2(S, K, T, r, q, sigma);
    return carry_value<Cdf>(disc_s, disc_k, d.d1, d.d2, phi);
}

/// Price, analytic Greeks and the terms they are built from.
struct CarryGreeks {
    double price;
    double delta;
    double delta_forward;
    double gamma;
    double vega;
    double theta;  ///< phi form (see the file header); per year
    double rho;    ///< dV/dr
    double rho_q;  ///< dV/dq (fxopt: rho_foreign)
    double vanna;
    double volga;
    // Terms, for an engine that adds them in its own order.
    double d1, d2, sqrt_t, df_q, df_r, pdf, n1, n2;  ///< n1 = N(phi d1), n2 = N(phi d2)
    double common_theta;                             ///< -S e^{-qT} n(d1) sigma / (2 sqrt T)
};

/// Analytic Greeks for S, K, T, sigma > 0.
template <class Cdf = NormCdfDiv>
[[nodiscard]] inline CarryGreeks carry_greeks(double S, double K, double T, double r, double q,
                                              double sigma, double phi) noexcept {
    const CarryD d = carry_d1_d2(S, K, T, r, q, sigma);
    const double sqrt_t = std::sqrt(T);
    const double df_q = std::exp(-q * T);
    const double df_r = std::exp(-r * T);
    const double pdf = norm_pdf(d.d1);
    const double n1 = Cdf::eval(phi * d.d1);
    const double n2 = Cdf::eval(phi * d.d2);
    const double vega = S * df_q * pdf * sqrt_t;
    const double common_theta = -S * df_q * pdf * sigma / (2.0 * sqrt_t);
    CarryGreeks g{};
    g.price = phi * (S * df_q * n1 - K * df_r * n2);
    g.delta = phi * df_q * n1;
    g.delta_forward = phi * n1;
    g.gamma = df_q * pdf / (S * sigma * sqrt_t);
    g.vega = vega;
    g.theta = common_theta + phi * (q * S * df_q * n1 - r * K * df_r * n2);
    g.rho = phi * K * T * df_r * n2;
    g.rho_q = -phi * S * T * df_q * n1;
    g.vanna = -df_q * pdf * d.d2 / sigma;
    g.volga = vega * d.d1 * d.d2 / sigma;
    g.d1 = d.d1;
    g.d2 = d.d2;
    g.sqrt_t = sqrt_t;
    g.df_q = df_q;
    g.df_r = df_r;
    g.pdf = pdf;
    g.n1 = n1;
    g.n2 = n2;
    g.common_theta = common_theta;
    return g;
}

}  // namespace qp
