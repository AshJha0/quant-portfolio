// Multi-currency FX book: positions, base-currency P&L, triangulation.
//
// Mirrors python/fx/03-var-es-engine fx_var.book (spot + forward + cash
// subset; options stay in the Python research stack - see
// docs/METHODOLOGY.md).
//
// Factor representation
// ---------------------
// Every currency is represented by its USD factor: "FX:CCY" is the daily
// log return of the USD price of 1 unit of CCY (the log return of CCYUSD).
// USD has no FX factor - its USD price is identically 1.  A position in a
// *cross* pair (EURJPY) is decomposed into its two USD legs - long EUR,
// short JPY - so cross risk is triangulated by construction and the factor
// set is arbitrage-consistent.  "IR:CCY" is an absolute shock (decimal
// p.a.) to the continuously compounded ACT/365 zero rate (flat curve per
// currency).
//
// Position types
// --------------
//   CashPosition    : a currency balance; riskless when denominated in the
//                     book's base currency.
//   SpotPosition    : long `notional` of the pair's base ccy vs the quote
//                     ccy at `entry_rate` (defaults to the reference
//                     market's cross rate = zero initial value).
//   ForwardPosition : outright forward as spot + two deposit legs (CIP):
//                     value in USD is
//                       N e^{-r_f T} S_base - N K e^{-r_d T} S_quote,
//                     so forward points expose the position to both
//                     currencies' interest-rate factors.
//
// P&L convention
// --------------
// P&L is profit (+) / loss (-) in the book's base currency:
// PnL = V1_usd / S1_base - V0_usd / S0_base, with the base currency's own
// USD price shocked consistently.  A pure base-ccy cash balance therefore
// carries exactly zero risk.
//
// Hot path
// --------
// CompiledBook flattens the book into a struct-of-arrays leg list (one
// exp() per leg per scenario), which is what historical / Monte Carlo
// revaluation iterates over.  See bench/bench_main.cpp for throughput.

#pragma once

#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "fxvar/returns.hpp"

namespace fxvar {

/// (base, quote) legs of a 6-letter pair, upper-cased.
struct PairLegs {
  std::string base;
  std::string quote;
};

/// Split "EURUSD" -> {"EUR", "USD"} (USD per 1 EUR).  Throws
/// std::invalid_argument unless the pair is 6 alphabetic characters with
/// distinct legs.
PairLegs split_pair(const std::string& pair);

/// Factor name for the log return of CCYUSD.  Throws for USD (the pivot).
std::string fx_factor(const std::string& ccy);

/// Factor name for an absolute shock to CCY's cc zero rate (ACT/365).
std::string ir_factor(const std::string& ccy);

/// Point-in-time market snapshot.
///
/// `spot_usd` maps ccy -> USD price of 1 unit (spot_usd["EUR"] = 1.08
/// means EURUSD = 1.08); "USD" is implied at 1.0 and may be omitted (if
/// present it must equal 1.0).  `rates` maps ccy -> continuously
/// compounded zero rate, annualised, ACT/365 (flat curve per currency).
class Market {
 public:
  explicit Market(std::map<std::string, double> spot_usd,
                  std::map<std::string, double> rates = {});

  /// USD price of 1 unit of `ccy` (1.0 for USD).  Throws if unknown.
  double spot(const std::string& ccy) const;

  /// cc zero rate for `ccy`.  Throws if unknown.
  double rate(const std::string& ccy) const;

  /// Cross rate of `pair` (QUOTE per 1 BASE) by USD triangulation.
  double cross(const std::string& pair) const;

  /// CIP forward: F = X * exp((r_d - r_f) * T) (QUOTE per BASE).
  double forward(const std::string& pair, double expiry) const;

 private:
  std::map<std::string, double> spot_usd_;
  std::map<std::string, double> rates_;
};

/// A cash balance of `amount` units of `ccy`.
struct CashPosition {
  std::string ccy;
  double amount = 0.0;
};

/// Spot FX position: long `notional` of BASE ccy vs QUOTE at `entry_rate`
/// (units of the pair's base ccy; negative = short).  `entry_rate` empty
/// means struck at the reference market's cross rate (zero initial value).
struct SpotPosition {
  std::string pair;
  double notional = 0.0;
  std::optional<double> entry_rate;
};

/// Outright FX forward: long `notional` BASE at `strike`, expiry in years.
/// `strike` empty resolves to the ATM CIP forward of the reference market.
struct ForwardPosition {
  std::string pair;
  double notional = 0.0;
  double expiry = 0.0;
  std::optional<double> strike;
};

using Position = std::variant<CashPosition, SpotPosition, ForwardPosition>;

/// A multi-currency book with a designated base (reporting) currency.
class Book {
 public:
  explicit Book(std::vector<Position> positions = {},
                std::string base = "USD");

  const std::string& base() const { return base_; }
  const std::vector<Position>& positions() const { return positions_; }
  bool empty() const { return positions_.empty(); }
  void add(Position p) { positions_.push_back(std::move(p)); }

  /// All currencies the book touches, including the base ccy (sorted).
  std::vector<std::string> currencies() const;

  /// Sorted risk-factor names the book is exposed to: FX:* for every
  /// non-USD currency involved (incl. base if not USD), then IR:* for the
  /// forwards' leg currencies.
  std::vector<std::string> factors() const;

 private:
  std::vector<Position> positions_;
  std::string base_;
};

/// Struct-of-arrays compiled book: the revaluation hot path.
///
/// Each position is flattened into discounted USD legs with unshocked
/// value v0 = amount * spot0 * exp(-r0 T); a scenario revalues each leg as
/// v0 * exp(dfx - T * dr), one exp() per leg.  Construction resolves
/// default entry rates / strikes against the reference market.
///
/// Throws std::invalid_argument for an empty book (a VaR on nothing is a
/// configuration error, not a zero).
class CompiledBook {
 public:
  CompiledBook(const Book& book, const Market& market);

  /// Factor order every shock vector must follow.
  const std::vector<std::string>& factors() const { return factors_; }

  /// Unshocked book value in USD.
  double value0_usd() const { return v0_usd_; }

  /// Book value in USD for one scenario; `shocks` has factors().size()
  /// entries aligned with factors() (FX = log returns, IR = absolute).
  double value_usd(const double* shocks) const;

  /// Base-ccy P&L of one scenario (shocks aligned with factors()).
  double pnl(const double* shocks) const;

  /// Base-ccy P&L of one scenario given as a factor->shock map.  Shocks
  /// for factors the book does not carry are ignored (one scenario
  /// library serves every book); a shock on "FX:USD" throws - USD is the
  /// pivot, shock the other leg(s).
  double pnl(const std::map<std::string, double>& shocks) const;

  /// Base-ccy P&L of every scenario row.  `scenarios` must contain every
  /// factor in factors() (extra columns are ignored); no copy of the
  /// scenario matrix is made.
  std::vector<double> pnl(const ReturnsMatrix& scenarios) const;

  /// Delta exposures dPnL/dfactor by central finite differences, aligned
  /// with factors().  Units: base-ccy P&L per unit factor move - for FX:*
  /// per unit log return (base-ccy notional exposure), for IR:* per 1.00
  /// of rate.  These are the mapping weights used by the
  /// variance-covariance method and the reverse-stress closed form.
  std::vector<double> linear_exposures(double bump = 1e-6) const;

 private:
  // Struct-of-arrays legs.
  std::vector<double> value0_;  ///< discounted unshocked USD leg value
  std::vector<int> fx_idx_;     ///< index into factors_, -1 = USD leg
  std::vector<int> ir_idx_;     ///< index into factors_, -1 = no rate leg
  std::vector<double> neg_expiry_;  ///< -T (0 for undiscounted legs)

  std::vector<std::string> factors_;
  std::string base_;
  int base_fx_idx_ = -1;  ///< index of FX:<base> in factors_, -1 if USD
  double v0_usd_ = 0.0;
  double s0_base_ = 1.0;
};

}  // namespace fxvar
