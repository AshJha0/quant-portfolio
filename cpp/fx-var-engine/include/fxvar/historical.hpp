// Historical-simulation VaR: plain HS, age-weighted (BRW), filtered HS.
//
// All three variants revalue the *actual book* (full revaluation through
// CompiledBook, including forwards' rate legs) under historical
// factor-return scenarios (mirrors fx_var.historical_var):
//
//   * Plain    - each of the last T days is an equally weighted scenario.
//   * Age      - Boudoukh-Richardson-Whitelaw exponential age weights
//                w_t ~ lambda^age: recent days dominate, so the VaR reacts
//                faster after a regime change.
//   * Filtered - Filtered Historical Simulation (Barone-Adesi et al.):
//                returns are devolatilised by a per-factor EWMA sigma and
//                rescaled to today's sigma forecast, preserving the
//                empirical cross-sectional dependence while making the
//                scenario set conditionally heteroscedastic.
//
// Multi-day horizons use sqrt-time scaling of the 1-day figure (documented
// limitation for carry books with negative skew - docs/VALIDATION.md F4).
//
// Peg blindness: HS sees only what is in the window.  A pegged currency
// contributes ~zero scenarios, so the engine surfaces a warnings list
// (HistoricalResult::warnings / flagged_peg_factors) and the desk must add
// the peg-break stress add-on (stress.hpp).

#pragma once

#include <string>
#include <vector>

#include "fxvar/book.hpp"
#include "fxvar/returns.hpp"

namespace fxvar {

enum class HsMethod { kPlain, kAge, kFiltered };

struct HistoricalOptions {
  double alpha = 0.99;
  double horizon_days = 1.0;  ///< > 1 uses sqrt-time scaling
  HsMethod method = HsMethod::kPlain;
  double decay = 0.995;        ///< BRW age-weight decay (method kAge)
  double ewma_lambda = 0.94;   ///< FHS devolatilisation (method kFiltered)
  std::size_t min_obs = 60;
  bool warn_pegs = true;
};

/// Result of a historical-simulation VaR run.  `var`/`es` are positive
/// losses in the book's base currency at the requested horizon; `pnl` and
/// `weights` are the 1-day scenario P&L vector and the scenario weights
/// used for the quantile.  `warnings` surfaces peg-blindness diagnostics
/// in human-readable form; `flagged_peg_factors` lists the factors.
struct HistoricalResult {
  double var = 0.0;
  double es = 0.0;
  double alpha = 0.99;
  double horizon_days = 1.0;
  HsMethod method = HsMethod::kPlain;
  std::vector<double> pnl;
  std::vector<double> weights;
  std::vector<std::string> flagged_peg_factors;
  std::vector<std::string> warnings;
};

/// Historical-simulation VaR/ES for `book` at `market`.
///
/// `returns` must contain every factor in book.factors() ("FX:*" log
/// returns, "IR:*" absolute changes); NaNs throw.  An empty book throws
/// std::invalid_argument; a non-empty book with no risk factors (pure
/// base-ccy cash) reports exactly zero VaR/ES.
HistoricalResult historical_var(const Book& book, const Market& market,
                                const ReturnsMatrix& returns,
                                const HistoricalOptions& opts = {});

}  // namespace fxvar
