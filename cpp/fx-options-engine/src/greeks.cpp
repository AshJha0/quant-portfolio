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
    return value_or_throw(try_analytic_greeks(S, K, T, r_d, r_f, sigma, type));
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
