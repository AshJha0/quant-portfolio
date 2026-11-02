#include "fxopt/binomial.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "fxopt/garman_kohlhagen.hpp"

namespace fxopt {

namespace {

/// Backward induction with the exercise style hoisted out of the inner loop
/// (one instantiation each, no per-node branch).
template <bool American>
double induct(double* values, const double* exercise_value, std::size_t n, double disc,
              double p) noexcept {
    for (std::size_t step = n; step-- > 0;) {
        const std::size_t offset = n - step;  // index of m = -step
        for (std::size_t j = 0; j <= step; ++j) {
            const double v = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
            if constexpr (American) {
                values[j] = std::max(v, exercise_value[offset + 2 * j]);
            } else {
                values[j] = v;
            }
        }
    }
    return values[0];
}

}  // namespace

Result<double> try_binomial_price(double S, double K, double T, double r_d, double r_f,
                                  double sigma, OptionType type, int steps, Exercise exercise,
                                  std::span<double> workspace) noexcept {
    const double p_sign = phi(type);
    if (auto e = try_validate_inputs(S, K, T, r_d, r_f, sigma); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    if (steps < 1) [[unlikely]] {
        return std::unexpected(Error{Errc::BadSteps, "steps", static_cast<double>(steps)});
    }
    if (T == 0.0) {
        return std::max(p_sign * (S - K), 0.0);
    }
    if (sigma == 0.0) {
        // Degenerate tree; defer to the analytic limit (European) or
        // deterministic exercise optimisation (American on a drifting spot).
        if (exercise == Exercise::European) {
            return try_gk_price(S, K, T, r_d, r_f, sigma, type);
        }
        double best = 0.0;
        for (int i = 0; i <= steps; ++i) {
            const double t = T * static_cast<double>(i) / static_cast<double>(steps);
            // Discounted payoff without forming the forward (overflow-safe).
            const double value =
                std::max(p_sign * (S * std::exp(-r_f * t) - K * std::exp(-r_d * t)), 0.0);
            best = std::max(best, value);
        }
        return try_finite_result(best, "binomial_price");
    }

    const double dt = T / static_cast<double>(steps);
    const double u = std::exp(sigma * std::sqrt(dt));
    const double d = 1.0 / u;
    const double growth = std::exp((r_d - r_f) * dt);
    const double p = (growth - d) / (u - d);
    if (!(p >= 0.0 && p <= 1.0)) [[unlikely]] {
        return std::unexpected(Error{Errc::ProbabilityOutOfRange, "p", p});
    }
    const auto n = static_cast<std::size_t>(steps);
    if (workspace.size() < binomial_workspace_size(steps)) [[unlikely]] {
        return std::unexpected(Error{Errc::WorkspaceTooSmall, "try_binomial_price",
                                     static_cast<double>(workspace.size()),
                                     static_cast<double>(binomial_workspace_size(steps))});
    }
    const double disc = std::exp(-r_d * dt);

    // Node spots S u^m, m = -n..n, computed once (O(n) powers); step i,
    // node j sits at m = 2j - i, so backward induction does no pow().
    double* exercise_value = workspace.data();      // 2n + 1
    double* values = workspace.data() + 2 * n + 1;  // n + 1
    for (std::size_t idx = 0; idx <= 2 * n; ++idx) {
        const double m = static_cast<double>(idx) - static_cast<double>(n);
        exercise_value[idx] = p_sign * (S * std::pow(u, m) - K);
    }
    for (std::size_t j = 0; j <= n; ++j) values[j] = std::max(exercise_value[2 * j], 0.0);

    const double v0 = exercise == Exercise::American
                          ? induct<true>(values, exercise_value, n, disc, p)
                          : induct<false>(values, exercise_value, n, disc, p);
    return try_finite_result(v0, "binomial_price");
}

double binomial_price(double S, double K, double T, double r_d, double r_f, double sigma,
                      OptionType type, int steps, Exercise exercise) {
    std::vector<double> workspace(binomial_workspace_size(steps));
    return value_or_throw(
        try_binomial_price(S, K, T, r_d, r_f, sigma, type, steps, exercise, workspace));
}

}  // namespace fxopt
