// Analytic Garman-Kohlhagen Greeks, including both rhos and vanna/volga.
//
// FX specifics:
//   * Two rhos.  An FX option has rate sensitivity to *both* legs:
//     rho_d = dV/dr_d (positive for calls -- higher domestic rate lifts the
//     forward) and rho_f = dV/dr_f (negative for calls -- higher foreign
//     rate is a larger 'dividend' on the base currency).
//   * Vanna and volga.  FX desks mark smiles with risk reversals and
//     butterflies, whose P&L maps directly onto vanna (dDelta/dVol) and
//     volga (dVega/dVol), so they are first-class here.
//
// All Greeks are per unit foreign notional, prices in domestic currency.
// Theta is per year (divide by 365 for a daily theta); vega is per unit of
// vol (divide by 100 for 'per vol point').  Unit conventions match the
// Python reference exactly (cross-validated by the golden vectors).

#pragma once

#include <algorithm>
#include <cmath>

#include "fxopt/common.hpp"
#include "fxopt/garman_kohlhagen.hpp"

namespace fxopt {

/// Full GK Greek set for one option.  delta_spot/delta_forward are
/// unadjusted deltas (see fxopt/deltas.hpp for premium-adjusted variants).
struct GreeksResult {
    double price;
    double delta_spot;
    double delta_forward;
    double gamma;
    double vega;
    double theta;          ///< per year; divide by 365 for daily theta
    double rho_domestic;   ///< dV/dr_d
    double rho_foreign;    ///< dV/dr_f
    double vanna;          ///< d2V/(dS dsigma)
    double volga;          ///< d2V/dsigma^2
};

/// Spot gamma d2V/dS2 = e^{-r_f T} n(d1) / (S sigma sqrt(T)).
double gamma(double S, double K, double T, double r_d, double r_f,
             double sigma);

/// Vega dV/dsigma = S e^{-r_f T} n(d1) sqrt(T) (call = put).
double vega(double S, double K, double T, double r_d, double r_f,
            double sigma);

/// Vanna d2V/(dS dsigma) = -e^{-r_f T} n(d1) d2 / sigma.
/// The sensitivity a 25-delta risk reversal position monetises.
double vanna(double S, double K, double T, double r_d, double r_f,
             double sigma);

/// Volga d2V/dsigma2 = vega * d1 * d2 / sigma.
/// The sensitivity a 25-delta butterfly position monetises.
double volga(double S, double K, double T, double r_d, double r_f,
             double sigma);

/// Closed-form GK Greeks.  Requires T > 0 and sigma > 0 (throws otherwise).
GreeksResult analytic_greeks(double S, double K, double T, double r_d,
                             double r_f, double sigma, OptionType type);

/// Finite-difference Greek set (no price / delta_forward: those are either
/// the pricer itself or a trivial rescale).
struct FDGreeks {
    double delta_spot;
    double gamma;
    double vega;
    double theta;
    double rho_domestic;
    double rho_foreign;
    double vanna;
    double volga;
};

/// Central finite-difference Greeks for validating any pricer with the GK
/// signature.  `price_fn(S, K, T, r_d, r_f, sigma) -> double` is the
/// pricing function under test (templated so tree/MC/analytic pricers can
/// all be compared against the closed forms).
///
/// Uses relative bumps of size rel_bump on S and sigma, absolute bumps on
/// rates, and a central difference in calendar time for theta
/// (theta = -dV/dT).  Second-order Greeks (gamma, vanna, volga) use the
/// standard central stencils; the sigma second difference uses a larger
/// bump (sigma*1e-3) to balance round-off against truncation error.
template <typename PriceFn>
FDGreeks finite_difference_greeks(PriceFn&& price_fn, double S, double K,
                                  double T, double r_d, double r_f,
                                  double sigma, double rel_bump = 1e-5) {
    validate_inputs(S, K, T, r_d, r_f, sigma);
    const double h_s = S * rel_bump;
    const double h_v = std::max(sigma * rel_bump, 1e-7);
    const double h_r = 1e-6;
    const double h_t = std::min(1e-6, T / 4.0);
    const double h_v2 = std::max(sigma * 1e-3, 1e-5);

    const auto p = [&](double s, double sig, double rd, double rf, double t) {
        return price_fn(s, K, t, rd, rf, sig);
    };
    const double base = p(S, sigma, r_d, r_f, T);
    const double up_s = p(S + h_s, sigma, r_d, r_f, T);
    const double dn_s = p(S - h_s, sigma, r_d, r_f, T);
    const double up_v = p(S, sigma + h_v, r_d, r_f, T);
    const double dn_v = p(S, sigma - h_v, r_d, r_f, T);
    const double up_v2 = p(S, sigma + h_v2, r_d, r_f, T);
    const double dn_v2 = p(S, sigma - h_v2, r_d, r_f, T);

    return FDGreeks{
        .delta_spot = (up_s - dn_s) / (2.0 * h_s),
        .gamma = (up_s - 2.0 * base + dn_s) / (h_s * h_s),
        .vega = (up_v - dn_v) / (2.0 * h_v),
        .theta = -(p(S, sigma, r_d, r_f, T + h_t) -
                   p(S, sigma, r_d, r_f, T - h_t)) /
                 (2.0 * h_t),
        .rho_domestic = (p(S, sigma, r_d + h_r, r_f, T) -
                         p(S, sigma, r_d - h_r, r_f, T)) /
                        (2.0 * h_r),
        .rho_foreign = (p(S, sigma, r_d, r_f + h_r, T) -
                        p(S, sigma, r_d, r_f - h_r, T)) /
                       (2.0 * h_r),
        .vanna = (p(S + h_s, sigma + h_v, r_d, r_f, T) -
                  p(S + h_s, sigma - h_v, r_d, r_f, T) -
                  p(S - h_s, sigma + h_v, r_d, r_f, T) +
                  p(S - h_s, sigma - h_v, r_d, r_f, T)) /
                 (4.0 * h_s * h_v),
        .volga = (up_v2 - 2.0 * base + dn_v2) / (h_v2 * h_v2),
    };
}

/// Convenience overload: finite-difference Greeks of the GK pricer itself.
FDGreeks finite_difference_greeks(double S, double K, double T, double r_d,
                                  double r_f, double sigma, OptionType type,
                                  double rel_bump = 1e-5);

}  // namespace fxopt
