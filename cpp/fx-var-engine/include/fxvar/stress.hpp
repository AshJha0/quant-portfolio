// FX stress testing: historical replays, hypothetical scenarios, peg
// breaks, and reverse stress (mirrors fx_var.stress_testing, spot/forward
// factor set - no vol factors in the C++ engine).
//
// Stress is the complement to VaR, not a substitute: HS and var-covar are
// *blind* to risks absent from the estimation window (a pegged currency
// has no history of breaking - until it does).  Every scenario is a joint
// factor-shock map {per-ccy spot shocks (log returns), rate shifts
// (absolute), jumps} applied through full revaluation, so forwards feel
// their rate legs.
//
// Historical replay calibrations (close-to-close one-day moves vs USD,
// sources in docs/METHODOLOGY.md):
//   * GBP flash - Brexit referendum, 24 Jun 2016: GBPUSD -8.1%
//     (1.4877 -> 1.3679), EURUSD -2.4%, JPY +3.9% (safe haven), and a
//     BoE-easing rate shift.
//   * CHF depeg - SNB floor removal, 15 Jan 2015: CHFUSD +14.9%
//     close-to-close (intraday +30%+), EURUSD -1.4%.  The canonical
//     peg break: the prior 250 days of USDCHF had no daily move over
//     1.9%.
//   * JPY carry unwind, 7-8 Oct 1998: JPYUSD +11.5% over two days as
//     USDJPY fell 131 -> 117 (LTCM deleveraging); AUD -4%.
//
// Reverse stress: for a linearised book with exposures w and factor
// covariance Sigma, the most damaging shock at Mahalanobis radius k is
//   dx* = -k Sigma w / sqrt(w' Sigma w),   loss = k sqrt(w' Sigma w)
// - closed form, verified against an independent numerical search in
// tests (tolerance 1e-6).

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "fxvar/book.hpp"
#include "fxvar/matrix.hpp"

namespace fxvar {

/// A named stress scenario: factor shocks + description.  `shocks` maps
/// factor names to shocks in engine units ("FX:*" log returns, "IR:*"
/// absolute).  Shocks for factors the book does not carry are ignored at
/// run time, so one scenario library serves every book.
struct Scenario {
  std::string name;
  std::map<std::string, double> shocks;
  std::string description;
};

/// Convert a simple percentage move to a log return: ln(1 + pct).
/// Throws std::invalid_argument unless `pct` is finite and > -100% (a
/// -100% move is an infinite log return, which would silently turn a whole
/// stress report into -inf/NaN).
double simple_to_log(double pct);

/// Library of calibrated historical FX replay scenarios, keyed
/// "brexit_2016", "chf_depeg_2015", "jpy_1998".
std::map<std::string, Scenario> historical_scenarios();

/// Hypothetical broad USD move: USD strengthens by `pct` vs every listed
/// currency (pct = +0.10 means every CCYUSD falls 10% in simple terms;
/// negative pct weakens USD).  USD itself is skipped.  Throws for
/// pct <= -100%.
Scenario usd_broad_move(const std::vector<std::string>& ccys, double pct);

/// Peg-break stress add-on for a pegged/managed currency - the mandatory
/// companion to any HS/parametric VaR on a book holding pegged currencies
/// (the engine's peg-blindness warnings point here).  `jump` is the
/// simple revaluation vs USD (-0.30 = 30% devaluation; positive models a
/// CHF-2015-style upward break); `contagion` adds simple-percentage
/// co-moves for other currencies.  Throws for jump <= -100%.
Scenario peg_break_scenario(const std::string& ccy, double jump = -0.30,
                            const std::map<std::string, double>& contagion = {});

/// One row of a stress report.
struct StressRow {
  std::string key;
  std::string name;
  double pnl = 0.0;  ///< base-ccy P&L (loss negative)
  std::string description;
};

/// Full-revaluation P&L of `book` under each scenario, sorted
/// worst-first.  Shocks on factors the book does not carry are dropped.
std::vector<StressRow> run_stress(
    const Book& book, const Market& market,
    const std::map<std::string, Scenario>& scenarios);

/// Reverse-stress outcome: the worst-case factor shock and its loss.
struct ReverseStress {
  std::vector<double> shocks;  ///< aligned with the exposure vector
  double loss = 0.0;           ///< positive loss at the optimum
};

/// Closed-form reverse stress for a linear book: among all shocks dx with
/// Mahalanobis norm sqrt(dx' Sigma^-1 dx) <= radius, the loss -w'dx is
/// maximised at dx* = -radius * Sigma w / sqrt(w' Sigma w) with loss
/// radius * sqrt(w' Sigma w).  Throws if the book has zero linear risk or
/// radius <= 0.
ReverseStress reverse_stress_linear(const std::vector<double>& exposures,
                                    const Matrix& cov, double radius);

/// As reverse_stress_linear but solving for the radius that produces
/// `loss_target`: k = loss_target / sqrt(w' Sigma w).
ReverseStress reverse_stress_for_loss(const std::vector<double>& exposures,
                                      const Matrix& cov, double loss_target);

/// Independent numerical check of the closed form: maximises the linear
/// loss over the Mahalanobis ellipsoid by projected gradient ascent in
/// whitened coordinates (dx = L y, |y| = radius) from a seeded random
/// start - it never assumes the analytic optimum.  Used in tests to
/// confirm reverse_stress_linear to 1e-6.
ReverseStress reverse_stress_numerical(const std::vector<double>& exposures,
                                       const Matrix& cov, double radius,
                                       std::uint64_t seed = 0);

}  // namespace fxvar
