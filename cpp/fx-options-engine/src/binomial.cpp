#include "fxopt/binomial.hpp"

#include <algorithm>
#include <cmath>
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
                                    std::to_string(steps));
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
        const double drift = r_d - r_f;
        double best = 0.0;
        for (int i = 0; i <= steps; ++i) {
            const double t = T * static_cast<double>(i) /
                             static_cast<double>(steps);
            const double value =
                std::exp(-r_d * t) *
                std::max(p_sign * (S * std::exp(drift * t) - K), 0.0);
            best = std::max(best, value);
        }
        return best;
    }

    const double dt = T / static_cast<double>(steps);
    const double u = std::exp(sigma * std::sqrt(dt));
    const double d = 1.0 / u;
    const double growth = std::exp((r_d - r_f) * dt);
    const double p = (growth - d) / (u - d);
    if (!(p >= 0.0 && p <= 1.0)) {
        throw std::invalid_argument(
            "risk-neutral probability " + std::to_string(p) +
            " outside [0, 1]; increase steps (dt too large for |r_d - r_f| "
            "vs sigma)");
    }
    const double disc = std::exp(-r_d * dt);

    // Terminal nodes, low -> high: spot_j = S u^{2j - steps}.
    std::vector<double> values(static_cast<std::size_t>(steps) + 1);
    for (int j = 0; j <= steps; ++j) {
        const double spot = S * std::pow(u, 2.0 * j - steps);
        values[static_cast<std::size_t>(j)] =
            std::max(p_sign * (spot - K), 0.0);
    }

    for (int step = steps - 1; step >= 0; --step) {
        for (int j = 0; j <= step; ++j) {
            const std::size_t sj = static_cast<std::size_t>(j);
            values[sj] = disc * (p * values[sj + 1] + (1.0 - p) * values[sj]);
        }
        if (exercise == Exercise::American) {
            for (int j = 0; j <= step; ++j) {
                const double spot = S * std::pow(u, 2.0 * j - step);
                values[static_cast<std::size_t>(j)] =
                    std::max(values[static_cast<std::size_t>(j)],
                             p_sign * (spot - K));
            }
        }
    }
    return values[0];
}

}  // namespace fxopt
