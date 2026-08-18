/// \file binomial.hpp
/// \brief Cox–Ross–Rubinstein binomial tree for European and American options.
///
/// Conventions match eqopt::bs_price (and the Python reference
/// `eq_options/binomial.py`): continuously compounded annualised `r` and `q`
/// (ACT/365F), `T` in years, `sigma` annualised.
///
/// The backward induction runs in a single reused vector — O(n) memory,
/// O(n^2) time — updated in place from the terminal payoff slice.
///
/// Edge-case policy (matches Python):
///  - `T == 0` -> intrinsic value;
///  - `sigma == 0` -> deterministic world: European = discounted forward
///    intrinsic (identical to Black–Scholes); American = maximum over the
///    time grid of the discounted intrinsic along the deterministic path
///    `S exp((r - q) t)`;
///  - negative `S`, `K`, `T`, `sigma` throw std::invalid_argument;
///    `n_steps < 1` throws std::invalid_argument.

#ifndef EQOPT_BINOMIAL_HPP
#define EQOPT_BINOMIAL_HPP

#include "eqopt/black_scholes.hpp"

namespace eqopt {

/// Exercise style of the option.
enum class ExerciseStyle { European, American };

/// \brief CRR binomial price of a European or American option.
///
/// Uses `u = exp(sigma sqrt(dt))`, `d = 1/u` and risk-neutral probability
/// `p = (exp((r - q) dt) - d) / (u - d)`. Backward induction updates one
/// value vector in place (O(n) memory); American options compare
/// continuation value with intrinsic at every node, with node prices
/// evaluated in log space for stability at large `n`.
///
/// \param S       Spot price (currency units), `S >= 0`.
/// \param K       Strike price (currency units), `K >= 0`.
/// \param T       Time to expiry in years (ACT/365F), `T >= 0`.
/// \param r       Continuously compounded annualised risk-free rate
///                (negative rates supported).
/// \param sigma   Annualised log-return volatility, `sigma >= 0`.
/// \param q       Continuously compounded annualised dividend yield.
/// \param type    Option payoff direction.
/// \param exercise Exercise style.
/// \param n_steps Number of time steps, `>= 1`. European convergence to
///                Black–Scholes is O(1/n) with an oscillating odd/even term.
/// \return Present value in currency units.
/// \throws std::invalid_argument on negative/NaN inputs, `n_steps < 1`, or
///         if the risk-neutral probability falls outside (0, 1) — a sign
///         that `dt` is too large for the given `r - q` and `sigma`.
double crr_price(double S, double K, double T, double r, double sigma,
                 double q = 0.0, OptionType type = OptionType::Call,
                 ExerciseStyle exercise = ExerciseStyle::European,
                 int n_steps = 500);

/// \brief American-minus-European value on the same CRR tree.
///
/// Using the *same* tree for both legs cancels the O(1/n) discretisation
/// error, so the premium is accurate to much better than either price.
///
/// \param S,K,T,r,sigma,q As in crr_price().
/// \param type    Option payoff direction (puts carry the premium when
///                `q < r`).
/// \param n_steps Tree steps shared by both legs.
/// \return Early-exercise premium in currency units; floored at 0 to remove
///         residual floating-point noise.
double early_exercise_premium(double S, double K, double T, double r,
                              double sigma, double q = 0.0,
                              OptionType type = OptionType::Put,
                              int n_steps = 500);

}  // namespace eqopt

#endif  // EQOPT_BINOMIAL_HPP
