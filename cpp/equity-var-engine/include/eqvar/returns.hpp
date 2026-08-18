// Returns and linear-portfolio P&L mapping.
//
// Conventions (shared with the Python eq_var reference):
//   * returns panels are (T, n): one row per day, one column per factor;
//   * `exposures` are dollar sensitivities per unit factor return (the vector
//     w with sigma_p^2 = w' Sigma w);
//   * P&L is in currency units, losses negative; VaR/ES functions elsewhere
//     report positive numbers for losses.
#pragma once

#include <span>
#include <vector>

#include "eqvar/matrix.hpp"

namespace eqvar {

/// Log returns ln(p_t / p_{t-1}) from a positive price series (n >= 2).
std::vector<double> log_returns(std::span<const double> prices);

/// Simple (arithmetic) returns p_t / p_{t-1} - 1 from a positive price series.
std::vector<double> simple_returns(std::span<const double> prices);

/// Linear-portfolio scenario P&L: pnl_t = sum_j exposures_j * returns(t, j).
///
/// Throws std::invalid_argument for an empty portfolio (no exposures) or a
/// column-count mismatch with the panel.
std::vector<double> portfolio_pnl(const Matrix& returns, std::span<const double> exposures);

/// Validate a P&L series: at least `min_obs` finite observations.
/// Throws std::invalid_argument otherwise. Shared by the historical estimators.
void validate_pnl(std::span<const double> pnl, std::size_t min_obs);

/// Validate a tail probability alpha in (0, 0.5); throws otherwise.
void validate_alpha(double alpha);

}  // namespace eqvar
