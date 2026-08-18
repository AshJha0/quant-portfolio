// Expected Shortfall (ES / CVaR) and the shared tail-quantile machinery.
//
// Conventions (mirrors fx_var.expected_shortfall):
//   * P&L arrays are profit (+) / loss (-); VaR and ES are reported as
//     positive loss amounts in the book's base currency.
//   * Empirical VaR at level alpha on n scenarios with weights w (uniform
//     by default) is the order-statistic / inverse-ECDF quantile: sort
//     losses descending, accumulate weights, VaR = the loss at which the
//     cumulative tail weight first reaches 1 - alpha.  With uniform
//     weights this is the m-th worst loss, m = ceil(n (1 - alpha)).
//   * Empirical ES uses the Acerbi-Tasche tail-splitting estimator: the
//     worst losses averaged over exactly 1 - alpha of probability mass,
//     taking only a fractional share of the atom at the VaR level.  This
//     is the coherent (subadditive) estimator; ES >= VaR holds for every
//     sample.
//   * Closed forms: Normal(mu, sigma) P&L has VaR = -mu + sigma z_a and
//     ES = -mu + sigma phi(z_a)/(1-a).  Student-t uses the *standardised*
//     (unit-variance) t so sigma is always the true P&L std - normal and
//     t figures are comparable at equal risk.

#pragma once

#include <utility>
#include <vector>

namespace fxvar {

/// Empirical VaR (positive loss) at level `alpha`.  `weights` are
/// non-negative scenario weights (normalised internally; empty = uniform).
/// Throws std::invalid_argument for empty/NaN pnl, bad alpha or weights.
double empirical_var(const std::vector<double>& pnl, double alpha = 0.99,
                     const std::vector<double>& weights = {});

/// Empirical ES (positive loss): coherent Acerbi-Tasche estimator.
double empirical_es(const std::vector<double>& pnl, double alpha = 0.99,
                    const std::vector<double>& weights = {});

/// (VaR, ES) from one pass over the sample.
std::pair<double, double> empirical_var_es(
    const std::vector<double>& pnl, double alpha = 0.99,
    const std::vector<double>& weights = {});

/// Closed-form Normal VaR: -mean + sigma * z_alpha (positive loss).
double normal_var(double sigma, double alpha = 0.99, double mean = 0.0);

/// Closed-form Normal ES: -mean + sigma * phi(z_alpha) / (1 - alpha).
double normal_es(double sigma, double alpha = 0.99, double mean = 0.0);

/// Standardised Student-t VaR with true P&L std sigma (unit-variance
/// scaling sqrt((df-2)/df)); df must be > 2 for finite variance.
double t_var(double sigma, double alpha = 0.99, double df = 6.0,
             double mean = 0.0);

/// Standardised Student-t ES:
/// E[X | X > q_a] = f(q_a) (df + q_a^2) / ((1-a)(df-1)) for standard t.
double t_es(double sigma, double alpha = 0.99, double df = 6.0,
            double mean = 0.0);

}  // namespace fxvar
