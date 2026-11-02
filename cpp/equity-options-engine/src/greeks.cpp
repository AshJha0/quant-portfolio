#include "eqopt/greeks.hpp"

namespace eqopt {

BSGreeks bs_greeks(double S, double K, double T, double r, double sigma, double q,
                   OptionType type) {
    return value_or_throw(try_bs_greeks(S, K, T, r, sigma, q, type));
}

}  // namespace eqopt
