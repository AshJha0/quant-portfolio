// Leisen-Reimer binomial tree (port of eq_options.lr_tree).
//
// Leisen and Reimer (1996), Applied Mathematical Finance 3(4), 319-346,
// Peizer-Pratt method 2 inversion, eq. (6). n is forced odd. European error
// falls like 1/n^2. Cross-checked against the Python reference by golden
// values in tests/test_phase5.cpp.
#ifndef EQOPT_LR_TREE_HPP
#define EQOPT_LR_TREE_HPP

#include "eqopt/binomial.hpp"
#include "eqopt/black_scholes.hpp"

namespace eqopt {

/// Peizer-Pratt method 2: binomial probability matching N(z) for odd n.
/// \throws std::invalid_argument if n is not a positive odd integer.
double peizer_pratt(double z, int n);

/// Leisen-Reimer price. S, K, T, sigma must be strictly positive; an even
/// n_steps is raised by one.
/// \throws std::invalid_argument on invalid inputs or an invalid tree.
double lr_price(double S, double K, double T, double r, double sigma, double q = 0.0,
                OptionType type = OptionType::Call,
                ExerciseStyle exercise = ExerciseStyle::European, int n_steps = 201);

}  // namespace eqopt

#endif  // EQOPT_LR_TREE_HPP
