// VaR backtesting: Kupiec POF, Christoffersen independence, Basel traffic
// light, Acerbi-Szekely Z2.
//
// Exception convention: day t is an exception when the realised loss
// exceeds the VaR forecast made ex ante for day t: -pnl_t > var_t.
// `alpha` is the tail probability (0.01 = 99 % VaR) everywhere; values
// outside (0, 0.5) throw.
//
//   * Kupiec POF (unconditional coverage): LR test that the exception
//     frequency equals alpha; chi2(1) via the regularised incomplete gamma.
//   * Christoffersen independence: LR test against first-order Markov
//     clustering of exceptions; chi2(1).
//   * Conditional coverage: LR_cc = LR_uc + LR_ind; chi2(2).
//   * Basel traffic light, any window and alpha: F(x) = P(X <= x),
//     X ~ Binomial(n_obs, alpha) by pmf recursion. Green if F < 0.95,
//     yellow if F < 0.9999, red otherwise. Yellow plus factor: x_eq =
//     smallest x' in {5..9} with F250(x') >= F(x) - 1e-12 (F250 =
//     Binomial(250, 0.01) CDF; 9 if none), plus = {5: .40, 6: .50,
//     7: .65, 8: .75, 9: .85}[x_eq]. Multiplier k = 3 + plus (red: 4).
//     At n = 250, alpha = 0.01 this is the 1996 table.

#pragma once

#include <cstddef>
#include <vector>

namespace fxvar {

/// A likelihood-ratio test outcome.
struct LrTest {
  double lr = 0.0;  ///< test statistic (chi-square distributed under H0)
  double p = 1.0;   ///< survival-function p-value
};

/// Counts for Kupiec and Basel. Designated-initialiser friendly:
///   kupiec_pof({.n_obs = 250, .n_exceptions = 5, .alpha = 0.01})
struct KupiecInput {
  int n_obs = 0;
  int n_exceptions = 0;
  double alpha = 0.01;  ///< tail probability, in (0, 0.5)
};
using BaselInput = KupiecInput;

/// Kupiec proportion-of-failures test: LR_uc ~ chi2(1) under H0 that the
/// exception probability is alpha. Throws std::invalid_argument for alpha
/// outside (0, 0.5), n_obs < 1, n_exceptions < 0 or n_exceptions > n_obs.
LrTest kupiec_pof(const KupiecInput& in);

/// Old argument order (n_exceptions, n_obs, alpha). alpha now means the tail
/// probability, so the old default 0.99 throws.
[[deprecated("use kupiec_pof(KupiecInput{.n_obs, .n_exceptions, .alpha}). Removed next release.")]]
LrTest kupiec_pof(int n_exceptions, int n_obs, double alpha = 0.01);

/// Christoffersen (1998) independence test on a 0/1 exception series
/// (first-order Markov alternative); chi2(1). Degenerate series (no
/// transitions of one kind) return LR = 0. Requires >= 2 observations
/// of 0/1 values.
LrTest christoffersen_independence(const std::vector<int>& exceedances);

/// Christoffersen conditional coverage: LR_cc = LR_uc + LR_ind, chi2(2).
LrTest conditional_coverage(const std::vector<int>& exceedances,
                            double alpha = 0.01);

enum class Zone { kGreen, kYellow, kRed };

/// Basel traffic-light outcome for a backtest window.
struct TrafficLight {
  Zone zone = Zone::kGreen;
  int n_exceptions = 0;
  int n_obs = 0;
  double cumulative_prob = 0.0;  ///< P(X <= x) under Binomial(n, alpha)
  double plus_factor = 0.0;      ///< 0 green, 0.40 .. 0.85 yellow, 1 red
  double multiplier = 3.0;       ///< capital multiplier 3 + plus_factor
};

/// Basel traffic light for any window and tail probability (see header).
TrafficLight basel_traffic_light(const BaselInput& in);

/// Old argument order (n_exceptions, n_obs, alpha); alpha is now the tail
/// probability.
[[deprecated("use basel_traffic_light(BaselInput{.n_obs, .n_exceptions, .alpha}). "
             "Removed next release.")]]
TrafficLight basel_traffic_light(int n_exceptions, int n_obs = 250,
                                 double alpha = 0.01);

/// Acerbi-Szekely (2014) Z2 statistic for an ES backtest:
///   Z2 = sum_t pnl_t I_t / (T alpha ES_t) + 1,  I_t = 1{pnl_t < -VaR_t}.
/// Expected 0 under a correct model; NEGATIVE means ES is understated.
/// `var` and `es` are positive losses, one per day or size 1 (broadcast).
/// Requires finite inputs, VaR >= 0, ES > 0. Returns Z2 only (no p-value).
double acerbi_szekely_z2(const std::vector<double>& pnl,
                         const std::vector<double>& var,
                         const std::vector<double>& es, double alpha);

/// Full VaR backtest summary over a forecast/realisation window.
struct BacktestResult {
  std::size_t n_obs = 0;
  int n_exceptions = 0;
  double exception_rate = 0.0;
  double expected_rate = 0.0;  ///< = alpha
  LrTest kupiec;
  LrTest independence;
  LrTest conditional;
  TrafficLight traffic_light;
  std::vector<int> exceedances;
};

/// Score a realised P&L series (profit +) against ex-ante positive VaR
/// forecasts for the same days. Throws on length mismatch, < 2
/// observations, non-finite values or a negative VaR.
BacktestResult evaluate_var_backtest(const std::vector<double>& pnl,
                                     const std::vector<double>& var_forecasts,
                                     double alpha = 0.01);

}  // namespace fxvar
