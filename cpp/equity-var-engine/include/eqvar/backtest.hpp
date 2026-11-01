// eqvar/backtest.hpp — VaR backtesting: Kupiec POF, Christoffersen
// independence / conditional coverage, Basel traffic light.
//
// A backtest compares ex-ante VaR forecasts against realised P&L: an
// EXCEPTION is a day with pnl < -VaR.  LR statistics are asymptotically
// chi-squared under H0; p-values come from the regularized upper incomplete
// gamma (stats.hpp — no external deps), and the Basel zone probabilities
// from the exact binomial CDF via the regularized incomplete beta.
// Mirrors eq_var.backtesting.

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace eqvar {

/// Exception indicator per day: 1 iff pnl_t < -var_t.  `var` (positive-loss
/// convention) is scalar-broadcast if it has size 1, otherwise one entry per
/// day.  Throws std::invalid_argument on negative VaR, a size mismatch, or
/// any non-finite P&L / VaR value — a NaN would otherwise compare false and
/// be silently counted as "no exception", flattering the model under test.
[[nodiscard]] std::vector<std::uint8_t> exceptions_from_pnl(std::span<const double> pnl,
                                                            std::span<const double> var);

/// Kupiec (1995) proportion-of-failures likelihood-ratio test result.
struct KupiecResult {
    double lr = 0.0;        ///< LR_uc statistic, chi2(1) under H0.
    double pvalue = 1.0;    ///< chi2(1) survival probability of lr.
    double expected = 0.0;  ///< alpha * n_obs expected exceptions.
    double rate = 0.0;      ///< observed exception rate x / T.
};

/// Counts for Kupiec and Basel. Designated-initialiser friendly:
///   kupiec_pof({.n_obs = 250, .n_exceptions = 5, .alpha = 0.01})
struct KupiecInput {
    int n_obs = 0;
    int n_exceptions = 0;
    double alpha = 0.01;  ///< tail probability, in (0, 0.5)
};
using BaselInput = KupiecInput;

/// Kupiec POF (unconditional coverage):
///   LR_uc = -2 ln[ (1-p)^{T-x} p^x / ((1-x/T)^{T-x} (x/T)^x) ]
/// with p = alpha, T = n_obs, x = n_exceptions; 0 ln 0 = 0. Throws
/// std::invalid_argument for alpha outside (0, 0.5), n_obs < 1,
/// n_exceptions < 0 or n_exceptions > n_obs (no clamping).
[[nodiscard]] KupiecResult kupiec_pof(const KupiecInput& in);

[[deprecated("use kupiec_pof(KupiecInput{.n_obs, .n_exceptions, .alpha}). Removed next release.")]]
[[nodiscard]] KupiecResult kupiec_pof(int n_obs, int n_exceptions, double alpha = 0.01);

/// Christoffersen (1998) independence test result.
struct ChristoffersenResult {
    double lr = 0.0;      ///< LR_ind statistic, chi2(1) under independence.
    double pvalue = 1.0;
    double n00 = 0.0, n01 = 0.0, n10 = 0.0, n11 = 0.0;  ///< Markov transition counts.
    double pi01 = 0.0;    ///< P(exception today | none yesterday).
    double pi11 = 0.0;    ///< P(exception today | exception yesterday).
};

/// Independence LR against a first-order Markov alternative.  Clustered
/// exceptions inflate n11 and reject.  Degenerate series use 0 ln 0 = 0.
/// Requires at least 2 observations.
[[nodiscard]] ChristoffersenResult christoffersen_independence(
    std::span<const std::uint8_t> exceptions);

/// Conditional-coverage joint test: LR_cc = LR_uc + LR_ind ~ chi2(2) —
/// correct exception rate AND independence together.
struct ConditionalCoverageResult {
    double lr = 0.0;
    double pvalue = 1.0;
    double lr_uc = 0.0;
    double lr_ind = 0.0;
};

[[nodiscard]] ConditionalCoverageResult christoffersen_cc(
    std::span<const std::uint8_t> exceptions, double alpha = 0.01);

/// Basel traffic-light zone.
enum class BaselZone { Green, Yellow, Red };

/// Basel traffic-light result.
struct BaselResult {
    BaselZone zone = BaselZone::Green;
    double multiplier = 3.0;       ///< capital multiplier k = 3 + plus_factor.
    double plus_factor = 0.0;      ///< 0 (green), 0.40..0.85 (yellow), 1 (red).
    double cumulative_prob = 0.0;  ///< F(x) = P(X <= n_exceptions), X ~ Binomial(n_obs, alpha).
};

/// Basel traffic light for any window and tail probability.
/// zone: green if F(x) < 0.95, yellow if F(x) < 0.9999, red otherwise.
/// Yellow plus factor: x_eq = smallest x' in {5..9} with F250(x') >= F(x) - 1e-12,
/// F250 the Binomial(250, 0.01) CDF (x_eq = 9 if none); plus factor from
/// {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}. At n = 250, alpha = 0.01
/// this is the 1996 Basel table. Inputs validated as in kupiec_pof.
[[nodiscard]] BaselResult basel_traffic_light(const BaselInput& in);

/// Old argument order (n_exceptions, n_obs) at alpha = 0.01. Now zones by the
/// binomial CDF at n_obs and throws when n_exceptions > n_obs.
[[deprecated("use basel_traffic_light(BaselInput{.n_obs, .n_exceptions, .alpha}). "
             "Removed next release.")]]
[[nodiscard]] BaselResult basel_traffic_light(int n_exceptions, int n_obs = 250);

/// Acerbi-Szekely (2014) Z2 statistic for an ES backtest:
///   Z2 = sum_t pnl_t I_t / (T alpha ES_t) + 1,  I_t = 1{pnl_t < -VaR_t}.
/// Expected 0 under a correct model; NEGATIVE means ES is understated.
/// `var` and `es` are positive losses, one entry per day or size 1
/// (broadcast). Requires finite inputs, VaR >= 0 and ES > 0.
/// Returns Z2 only (no p-value; the Python engine simulates one).
[[nodiscard]] double acerbi_szekely_z2(std::span<const double> pnl, std::span<const double> var,
                                       std::span<const double> es, double alpha);

[[nodiscard]] std::string to_string(BaselZone zone);

}  // namespace eqvar
