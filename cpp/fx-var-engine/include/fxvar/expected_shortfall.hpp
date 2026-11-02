// Expected Shortfall (ES / CVaR) and the shared tail-quantile machinery.
//
// Conventions (shared with the Python, Rust and C++ VaR engines):
//   * P&L arrays are profit (+) / loss (-); VaR and ES are reported as
//     positive loss amounts in the book's base currency.
//   * `alpha` is the TAIL probability: 0.01 = 99 % VaR. Values outside
//     (0, 0.5) throw, so a confidence level such as 0.99 is rejected.
//   * Empirical VaR / ES: losses L = -pnl sorted descending, k = n alpha,
//     m = ceil(k - 1e-9) clamped to [1, n]; VaR = L(m),
//     ES = [ sum_{i<m} L(i) + (k - (m-1)) L(m) ] / k (Acerbi-Tasche with the
//     exact fractional atom). Uniform weights use std::nth_element (O(n)).
//   * Weighted samples: weights normalised to sum 1 and stable-sorted with
//     the losses; W_j cumulative; m = first j with W_j >= alpha - 1e-12;
//     ES = [ sum_{i<m} w_i L(i) + (alpha - W_{m-1}) L(m) ] / alpha.
//   * Closed forms: Normal(mu, sigma) P&L has VaR = -mu + sigma z,
//     ES = -mu + sigma phi(z) / alpha, z = Phi^{-1}(1 - alpha). Student-t
//     uses the standardised (unit-variance) t, so sigma is the P&L std.

#pragma once

#include <utility>
#include <vector>

namespace fxvar {

/// Empirical VaR (positive loss) at tail probability `alpha`. `weights` are
/// non-negative scenario weights (normalised internally; empty = uniform).
/// Throws std::invalid_argument for empty or non-finite pnl, bad alpha or
/// weights.
double empirical_var(const std::vector<double>& pnl, double alpha = 0.01,
                     const std::vector<double>& weights = {});

/// Empirical ES (positive loss): Acerbi-Tasche with the fractional atom.
double empirical_es(const std::vector<double>& pnl, double alpha = 0.01,
                    const std::vector<double>& weights = {});

/// (VaR, ES) from one pass over the sample.
std::pair<double, double> empirical_var_es(const std::vector<double>& pnl, double alpha = 0.01,
                                           const std::vector<double>& weights = {});

/// Closed-form Normal VaR: -mean + sigma * Phi^{-1}(1 - alpha).
double normal_var(double sigma, double alpha = 0.01, double mean = 0.0);

/// Closed-form Normal ES: -mean + sigma * phi(z) / alpha.
double normal_es(double sigma, double alpha = 0.01, double mean = 0.0);

/// Standardised Student-t VaR with P&L std sigma (unit-variance scaling
/// sqrt((df-2)/df)); df must be finite and > 2.
double t_var(double sigma, double alpha = 0.01, double df = 6.0, double mean = 0.0);

/// Standardised Student-t ES: f(q) (df + q^2) / (alpha (df-1)) for the
/// standard t at q = t^{-1}(1 - alpha), rescaled to unit variance.
double t_es(double sigma, double alpha = 0.01, double df = 6.0, double mean = 0.0);

}  // namespace fxvar
