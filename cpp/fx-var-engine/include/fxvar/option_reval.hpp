// fxvar/option_reval.hpp: option revaluation under VaR scenarios.
//
// Black-Scholes-Merton with a continuous carry yield (foreign rate r_f, which makes this Garman-Kohlhagen), closed-form Greeks, full
// revaluation P&L per scenario and the delta-gamma approximation. Mirrors
// fx_var.revaluation; golden values in tests/test_option_reval.cpp come from
// that module.
//
// Scenario layout: row-major n_scenarios x n_factors spot log returns; vol
// shocks (absolute, decimal) in the same layout or empty for none.
// Self-contained (std::erfc normal CDF) so it does not depend on the kernel
// API that Round 3 lane B is changing.

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace fxvar {

enum class OptionKind { Call, Put };

struct OptionLeg {
  double quantity;
  OptionKind kind;
  double spot;
  double strike;
  double tau;    // years, > 0
  double rate;   // continuously compounded
  double carry;  // dividend yield or foreign rate
  double vol;    // annualised, > 0
  std::size_t factor = 0;
};

struct BsmGreeks {
  double delta, gamma, vega, vanna, volga, theta;  // vega per 1.00 vol; theta = -dV/dtau
};

/// BSM price; intrinsic value when tau <= 0. Throws std::invalid_argument on
/// non-finite or out-of-domain input.
[[nodiscard]] double bsm_price(double spot, double strike, double tau, double rate, double carry,
                               double vol, OptionKind kind);

/// Closed-form Greeks per unit quantity (Hull, OFOD 11th ed., ch. 19).
[[nodiscard]] BsmGreeks bsm_greeks(const OptionLeg& leg);

/// Exact book P&L per scenario by repricing every leg; dt (years) ages the
/// options, floored at expiry.
[[nodiscard]] std::vector<double> full_revaluation_pnl(std::span<const OptionLeg> legs,
                                                       std::span<const double> spot_log_returns,
                                                       std::span<const double> vol_shocks,
                                                       std::size_t n_factors, double dt = 0.0);

/// Delta-gamma P&L per scenario: Delta dS + Gamma dS^2 / 2, dS = S (e^x - 1).
[[nodiscard]] std::vector<double> delta_gamma_pnl(std::span<const OptionLeg> legs,
                                                  std::span<const double> spot_log_returns,
                                                  std::size_t n_factors);

}  // namespace fxvar
