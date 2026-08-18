#include "fxopt/greeks.hpp"

#include <cmath>

namespace fxopt {

namespace {

struct Core {
    double d1v;
    double d2v;
    double df_f;
    double df_d;
};

Core core(double S, double K, double T, double r_d, double r_f, double sigma) {
    const double d1v = d1(S, K, T, r_d, r_f, sigma);
    return Core{d1v, d1v - sigma * std::sqrt(T), std::exp(-r_f * T),
                std::exp(-r_d * T)};
}

}  // namespace

double gamma(double S, double K, double T, double r_d, double r_f,
             double sigma) {
    const Core c = core(S, K, T, r_d, r_f, sigma);
    return c.df_f * norm_pdf(c.d1v) / (S * sigma * std::sqrt(T));
}

double vega(double S, double K, double T, double r_d, double r_f,
            double sigma) {
    const Core c = core(S, K, T, r_d, r_f, sigma);
    return S * c.df_f * norm_pdf(c.d1v) * std::sqrt(T);
}

double vanna(double S, double K, double T, double r_d, double r_f,
             double sigma) {
    const Core c = core(S, K, T, r_d, r_f, sigma);
    return -c.df_f * norm_pdf(c.d1v) * c.d2v / sigma;
}

double volga(double S, double K, double T, double r_d, double r_f,
             double sigma) {
    const Core c = core(S, K, T, r_d, r_f, sigma);
    return S * c.df_f * norm_pdf(c.d1v) * std::sqrt(T) * c.d1v * c.d2v / sigma;
}

GreeksResult analytic_greeks(double S, double K, double T, double r_d,
                             double r_f, double sigma, OptionType type) {
    const double p = phi(type);
    validate_inputs(S, K, T, r_d, r_f, sigma);
    if (T <= 0.0 || sigma <= 0.0) {
        throw std::invalid_argument(
            "analytic_greeks requires T > 0 and sigma > 0");
    }
    const Core c = core(S, K, T, r_d, r_f, sigma);
    const double sqrt_t = std::sqrt(T);
    const double n_d1 = norm_pdf(c.d1v);
    const double N_pd1 = norm_cdf(p * c.d1v);
    const double N_pd2 = norm_cdf(p * c.d2v);

    const double price = p * (S * c.df_f * N_pd1 - K * c.df_d * N_pd2);
    const double delta_spot = p * c.df_f * N_pd1;
    const double theta =
        -S * c.df_f * n_d1 * sigma / (2.0 * sqrt_t) +
        p * (r_f * S * c.df_f * N_pd1 - r_d * K * c.df_d * N_pd2);

    return GreeksResult{
        .price = price,
        .delta_spot = delta_spot,
        .delta_forward = p * N_pd1,
        .gamma = c.df_f * n_d1 / (S * sigma * sqrt_t),
        .vega = S * c.df_f * n_d1 * sqrt_t,
        .theta = theta,
        .rho_domestic = p * K * T * c.df_d * N_pd2,
        .rho_foreign = -p * S * T * c.df_f * N_pd1,
        .vanna = -c.df_f * n_d1 * c.d2v / sigma,
        .volga = S * c.df_f * n_d1 * sqrt_t * c.d1v * c.d2v / sigma,
    };
}

FDGreeks finite_difference_greeks(double S, double K, double T, double r_d,
                                  double r_f, double sigma, OptionType type,
                                  double rel_bump) {
    return finite_difference_greeks(
        [type](double s, double k, double t, double rd, double rf,
               double sig) { return gk_price(s, k, t, rd, rf, sig, type); },
        S, K, T, r_d, r_f, sigma, rel_bump);
}

}  // namespace fxopt
