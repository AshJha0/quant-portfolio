// VaR backtesting: Kupiec POF, Christoffersen independence, Basel
// traffic light (mirrors fx_var.backtesting).
//
// Exception convention: day t is an exception when the realised *loss*
// exceeds the VaR forecast made ex ante for day t: -pnl_t > var_t.
//
// Tests
// -----
//   * Kupiec POF (unconditional coverage): LR test that the exception
//     frequency equals 1 - alpha; chi2(1) via the regularised incomplete
//     gamma.
//   * Christoffersen independence: LR test against first-order Markov
//     clustering of exceptions; chi2(1).  FX desks care because
//     volatility clustering makes an unconditional method (plain HS,
//     sample-cov parametric) fail *this* test long before it fails
//     Kupiec.
//   * Conditional coverage: LR_cc = LR_uc + LR_ind; chi2(2).
//   * Basel traffic light: zones from the exact cumulative
//     Binomial(n, 1-alpha) probability of the exception count - green
//     < 95%, yellow < 99.99%, red >= 99.99% - which for the regulatory
//     250-day 99% window reproduces the table exactly: green 0-4,
//     yellow 5-9 (add-ons 0.40/0.50/0.65/0.75/0.85), red 10+
//     (multiplier 4.0).

#pragma once

#include <cstddef>
#include <vector>

namespace fxvar {

/// A likelihood-ratio test outcome.
struct LrTest {
  double lr = 0.0;  ///< test statistic (chi-square distributed under H0)
  double p = 1.0;   ///< survival-function p-value
};

/// Kupiec proportion-of-failures test: LR_uc ~ chi2(1) under H0 that the
/// exception probability is 1 - alpha.  Throws for invalid counts/alpha.
LrTest kupiec_pof(int n_exceptions, int n_obs, double alpha = 0.99);

/// Christoffersen (1998) independence test on a 0/1 exception series
/// (first-order Markov alternative); chi2(1).  Degenerate series (no
/// transitions of one kind) return LR = 0.  Requires >= 2 observations
/// of 0/1 values.
LrTest christoffersen_independence(const std::vector<int>& exceedances);

/// Christoffersen conditional coverage: LR_cc = LR_uc + LR_ind, chi2(2).
LrTest conditional_coverage(const std::vector<int>& exceedances,
                            double alpha = 0.99);

enum class Zone { kGreen, kYellow, kRed };

/// Basel traffic-light outcome for a backtest window.
struct TrafficLight {
  Zone zone = Zone::kGreen;
  int n_exceptions = 0;
  int n_obs = 0;
  double cumulative_prob = 0.0;  ///< P(X <= x) under Binomial(n, 1-alpha)
  double multiplier = 3.0;       ///< capital multiplier, 3.0 .. 4.0
};

/// Basel traffic-light zone and capital multiplier (1996 table add-ons
/// in the yellow zone; multiplier capped at 4.0 in red).
TrafficLight basel_traffic_light(int n_exceptions, int n_obs = 250,
                                 double alpha = 0.99);

/// Full VaR backtest summary over a forecast/realisation window.
struct BacktestResult {
  std::size_t n_obs = 0;
  int n_exceptions = 0;
  double exception_rate = 0.0;
  double expected_rate = 0.0;
  LrTest kupiec;
  LrTest independence;
  LrTest conditional;
  TrafficLight traffic_light;
  std::vector<int> exceedances;
};

/// Score a realised P&L series (profit +) against ex-ante positive VaR
/// forecasts for the same days.  Throws on length mismatch, < 2
/// observations, or NaNs (NaN policy: refuse).
BacktestResult evaluate_var_backtest(const std::vector<double>& pnl,
                                     const std::vector<double>& var_forecasts,
                                     double alpha = 0.99);

}  // namespace fxvar
