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
/// day.  Throws std::invalid_argument on negative VaR or size mismatch.
[[nodiscard]] std::vector<std::uint8_t> exceptions_from_pnl(std::span<const double> pnl,
                                                            std::span<const double> var);

/// Kupiec (1995) proportion-of-failures likelihood-ratio test result.
struct KupiecResult {
    double lr = 0.0;        ///< LR_uc statistic, chi2(1) under H0.
    double pvalue = 1.0;    ///< chi2(1) survival probability of lr.
    double expected = 0.0;  ///< alpha * n_obs expected exceptions.
    double rate = 0.0;      ///< observed exception rate x / T.
};

/// Kupiec POF (unconditional coverage):
///   LR_uc = -2 ln[ (1-p)^{T-x} p^x / ((1-x/T)^{T-x} (x/T)^x) ]
/// with p = alpha, T = n_obs, x = n_exceptions; x = 0 and x = T use the
/// 0 ln 0 = 0 convention.  Throws std::invalid_argument on bad counts.
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

struct BaselResult {
    BaselZone zone = BaselZone::Green;
    double multiplier = 3.0;       ///< capital multiplier k = 3 + add-on.
    double cumulative_prob = 0.0;  ///< P(X <= n_exceptions), X ~ Binomial(n_obs, 0.01).
};

/// Basel (1996 supervisory framework) traffic light for 99 % VaR on the
/// standard 250-day window: 0-4 exceptions = green (multiplier 3.0), 5-9 =
/// yellow (add-ons 0.40 / 0.50 / 0.65 / 0.75 / 0.85), 10+ = red (add-on 1.0,
/// presumption of a flawed model).  Zone boundaries are the regulatory EXACT
/// counts; the cumulative binomial probability of the observed count under a
/// correct model is reported alongside.
[[nodiscard]] BaselResult basel_traffic_light(int n_exceptions, int n_obs = 250);

/// Human-readable zone name ("green" / "yellow" / "red").
[[nodiscard]] std::string to_string(BaselZone zone);

}  // namespace eqvar
