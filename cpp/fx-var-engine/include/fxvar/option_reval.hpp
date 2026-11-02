// fxvar/option_reval.hpp: option revaluation under VaR scenarios.
//
// Black-Scholes-Merton with a continuous carry yield (foreign rate r_f, which makes this Garman-Kohlhagen), closed-form Greeks, full
// revaluation P&L per scenario and the delta-gamma approximation. Mirrors
// fx_var.revaluation; golden values in tests/test_option_reval.cpp come from
// that module.
//
// Scenario layout: row-major n_scenarios x n_factors spot log returns; vol
// shocks (absolute, decimal) in the same layout or empty for none.
// The pricer is self-contained (std::erfc normal CDF). The layout and P&L
// helpers at the bottom let historical_var and monte_carlo_var (overloads in
// historical.hpp / monte_carlo.hpp) reprice every option leg in full in every
// scenario (Jorion, Value at Risk, 3rd ed., ch. 10). Options are on CCYUSD:
// spot in USD per unit of CCY, domestic rate r_USD, carry r_CCY, P&L in USD;
// a book holding options must therefore report in USD.

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fxvar/book.hpp"
#include "fxvar/returns.hpp"

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

/// A linear book plus option legs on one factor axis. `factors` lists the
/// book's own factors first (CompiledBook order), then any option factor the
/// book does not carry; `legs` have `factor` re-indexed into `factors`.
struct OptionBookLayout {
  std::vector<std::string> factors;
  std::optional<CompiledBook> linear;  ///< empty when the linear book is empty
  std::vector<OptionLeg> legs;
};

/// Build the layout. `options[k].factor` indexes `option_factors`, whose
/// names must be "FX:CCY" (CCY != USD). Throws std::invalid_argument for no
/// options, a bad factor index or name, invalid legs, or a non-USD base.
[[nodiscard]] OptionBookLayout layout_option_book(const Book& book, const Market& market,
                                                  std::span<const OptionLeg> options,
                                                  const std::vector<std::string>& option_factors);

/// Scenario P&L (USD): CompiledBook::pnl for the linear legs plus
/// full_revaluation_pnl for the options. `scenarios` must carry every factor
/// in layout.factors (it is re-selected into that order; extras ignored).
[[nodiscard]] std::vector<double> option_book_pnl(const OptionBookLayout& layout,
                                                  const ReturnsMatrix& scenarios, double dt = 0.0);

}  // namespace fxvar
