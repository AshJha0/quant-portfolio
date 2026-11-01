// Cox-Ross-Rubinstein binomial tree for FX options.
//
// The foreign interest rate enters exactly like a continuous dividend
// yield: risk-neutral drift of the spot under the domestic measure is
// r_d - r_f, so the up-move probability is
//
//   p = (e^{(r_d - r_f) dt} - d) / (u - d),  u = e^{sigma sqrt(dt)},  d = 1/u.
//
// Supports European and American exercise.  American FX options trade OTC;
// the economically interesting case is an American *call* on a
// high-yielding foreign currency (r_f > r_d): the foreign carry lost by
// holding the option instead of the currency makes early exercise optimal,
// giving the American call a strictly positive premium over European --
// mirroring the dividend-yield story for equities.

#pragma once

#include <cstddef>
#include <span>

#include "fxopt/common.hpp"

namespace fxopt {

/// Exercise style.
enum class Exercise { European, American };

/// CRR binomial price of an FX option, domestic ccy per unit foreign
/// notional.  Throws std::invalid_argument on invalid inputs, steps < 1,
/// or if the tree probability falls outside [0, 1] (time step too coarse
/// for the drift/vol combination).  sigma = 0 defers to the analytic limit
/// (European) or deterministic exercise optimisation (American).
double binomial_price(double S, double K, double T, double r_d, double r_f,
                      double sigma, OptionType type, int steps = 500,
                      Exercise exercise = Exercise::European);

/// Doubles of workspace try_binomial_price() needs for `steps`: `3 n + 2`
/// (2n+1 node exercise values plus n+1 option values).
constexpr std::size_t binomial_workspace_size(int steps) noexcept {
    return steps < 1 ? 0 : 3 * static_cast<std::size_t>(steps) + 2;
}

/// noexcept CRR price in a caller-owned workspace. Same arguments, value
/// and error policy as binomial_price(); `workspace` must hold at least
/// binomial_workspace_size(steps) doubles (Errc::WorkspaceTooSmall
/// otherwise). No allocation.
[[nodiscard]] Result<double> try_binomial_price(double S, double K, double T, double r_d,
                                                double r_f, double sigma, OptionType type,
                                                int steps, Exercise exercise,
                                                std::span<double> workspace) noexcept;

}  // namespace fxopt
