// eqvar/option_reval.hpp: option revaluation under VaR scenarios.
//
// Black-Scholes-Merton with a continuous carry yield (dividend yield for
// equity, foreign rate for FX = Garman-Kohlhagen), closed-form Greeks, full
// revaluation P&L per scenario and the delta-gamma approximation. Mirrors
// eq_var.revaluation; golden values in tests/test_option_reval.cpp come from
// that module.
//
// Scenario layout: row-major n_scenarios x n_factors spot log returns; vol
// shocks (absolute, decimal) in the same layout or empty for none.
// The pricer is self-contained (std::erfc normal CDF). The two VaR drivers at
// the bottom (historical_option_var_es, monte_carlo_option_var) wire the
// pricer into historical simulation and Monte Carlo: every scenario reprices
// every option leg in full (Hull, OFOD 11th ed., ch. 22 "full revaluation";
// Jorion, Value at Risk, 3rd ed., ch. 10). A book with no option legs goes
// down the unchanged linear path.

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "eqvar/historical.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/monte_carlo.hpp"
#include "eqvar/parametric.hpp"

namespace eqvar {

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

/// How the scenario vol of an option leg is formed (mirrors the Python
/// reference's `vol_mode`; Derman 1999, "Regimes of volatility", for the
/// first two):
///   StickyStrike   sigma_1 = sigma_0 + dv              (strike's vol fixed)
///   StickyDelta    sigma_1 = sigma_0 - skew * x + dv   (smile moves with spot;
///                  skew = d sigma / d ln(K/S), x = the factor log return)
///   HistoricalAtm  sigma_1 = sigma_0 * exp(dv)         (dv = historical log
///                  change of the ATM vol, applied to every strike)
/// Every mode floors the vol at 1e-12.
enum class VolMode { StickyStrike, StickyDelta, HistoricalAtm };

/// Full revaluation P&L with an explicit vol mode. `skews` is empty (all
/// zero) or has one entry per leg; it is read only by StickyDelta. With
/// StickyStrike and empty skews this is exactly the overload above.
[[nodiscard]] std::vector<double> full_revaluation_pnl(std::span<const OptionLeg> legs,
                                                       std::span<const double> spot_log_returns,
                                                       std::span<const double> vol_shocks,
                                                       std::size_t n_factors, double dt,
                                                       VolMode mode, std::span<const double> skews);

/// Delta-gamma P&L per scenario: Delta dS + Gamma dS^2 / 2, dS = S (e^x - 1).
[[nodiscard]] std::vector<double> delta_gamma_pnl(std::span<const OptionLeg> legs,
                                                  std::span<const double> spot_log_returns,
                                                  std::size_t n_factors);

/// Scenario P&L of a mixed book: exposures . r (the engine's linear
/// convention, unchanged) plus full_revaluation_pnl of the option legs on the
/// same rows. `factor_returns` is n x J spot log returns; `exposures` is empty
/// (options only) or has J entries; legs may be empty. No vol shocks.
[[nodiscard]] std::vector<double> option_book_pnl(std::span<const OptionLeg> legs,
                                                  std::span<const double> exposures,
                                                  const Matrix& factor_returns, double dt = 0.0);

/// Historical-simulation VaR / ES with full option revaluation: each of the
/// T >= kMinHistObs historical rows of `factor_returns` is a scenario, the
/// book P&L is option_book_pnl, and VaR / ES come from
/// var_es_order_statistic. With no option legs this is exactly
/// var_es_order_statistic(exposures . r), the linear path.
[[nodiscard]] TailRisk historical_option_var_es(std::span<const OptionLeg> legs,
                                                std::span<const double> exposures,
                                                const Matrix& factor_returns, double alpha = 0.01,
                                                double dt = 0.0);

/// Monte Carlo VaR / ES with full option revaluation: simulate_factor_returns
/// (same model, seed and repair as monte_carlo_var), option_book_pnl on every
/// path, then mc_tail_metrics. With no option legs it calls monte_carlo_var
/// unchanged. `cov` is J x J; `exposures` is empty or has J entries.
[[nodiscard]] MonteCarloResult monte_carlo_option_var(
    std::span<const OptionLeg> legs, std::span<const double> exposures, const Matrix& cov,
    double alpha = 0.01, std::size_t n_paths = 100'000, Dist dist = Dist::Normal, double df = 6.0,
    std::uint64_t seed = 0, std::size_t n_boot = 200, Repair repair = Repair::None,
    double dt = 0.0);

}  // namespace eqvar
