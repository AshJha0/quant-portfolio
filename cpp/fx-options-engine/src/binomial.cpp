#include "fxopt/binomial.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <vector>

#include "fxopt/garman_kohlhagen.hpp"

namespace fxopt {

double binomial_price(double S, double K, double T, double r_d, double r_f,
                      double sigma, OptionType type, int steps,
                      Exercise exercise) {
    const double p_sign = phi(type);
    validate_inputs(S, K, T, r_d, r_f, sigma);
    if (steps < 1) {
        throw std::invalid_argument("steps must be a positive integer, got " +
                                    std::format("{}", steps));
    }
    if (T == 0.0) {
        return std::max(p_sign * (S - K), 0.0);
    }
    if (sigma == 0.0) {
        // Degenerate tree; defer to the analytic limit (European) or
        // deterministic exercise optimisation (American on a drifting spot).
        if (exercise == Exercise::European) {
            return gk_price(S, K, T, r_d, r_f, sigma, type);
        }
        double best = 0.0;
        for (int i = 0; i <= steps; ++i) {
            const double t = T * static_cast<double>(i) /
                             static_cast<double>(steps);
            // Discounted payoff without forming the forward (overflow-safe).
            const double value = std::max(
                p_sign * (S * std::exp(-r_f * t) - K * std::exp(-r_d * t)), 0.0);
            best = std::max(best, value);
        }
        return detail::finite_result(best, "binomial_price");
    }

    const double dt = T / static_cast<double>(steps);
    const double u = std::exp(sigma * std::sqrt(dt));
    const double d = 1.0 / u;
    const double growth = std::exp((r_d - r_f) * dt);
    const double p = (growth - d) / (u - d);
    if (!(p >= 0.0 && p <= 1.0)) {
        throw std::invalid_argument(
            std::format("risk-neutral probability {} outside [0, 1]; increase steps "
                        "(dt too large for |r_d - r_f| vs sigma)", p));
    }
    const double disc = std::exp(-r_d * dt);

    // Node spots S u^m, m = -n..n, computed once (O(n) powers); step i,
    // node j sits at m = 2j - i, so backward induction does no pow().
    const std::size_t n = static_cast<std::size_t>(steps);
    std::vector<double> exercise_value(2 * n + 1);
    for (std::size_t idx = 0; idx <= 2 * n; ++idx) {
        const double m = static_cast<double>(idx) - static_cast<double>(n);
        exercise_value[idx] = p_sign * (S * std::pow(u, m) - K);
    }
    std::vector<double> values(n + 1);
    for (std::size_t j = 0; j <= n; ++j) values[j] = std::max(exercise_value[2 * j], 0.0);

    const bool american = exercise == Exercise::American;
    for (std::size_t step = n; step-- > 0;) {
        const std::size_t offset = n - step;  // index of m = -step
        for (std::size_t j = 0; j <= step; ++j) {
            const double v = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
            values[j] = american ? std::max(v, exercise_value[offset + 2 * j]) : v;
        }
    }
    return detail::finite_result(values[0], "binomial_price");
}

}  // namespace fxopt
