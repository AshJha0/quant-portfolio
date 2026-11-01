#include "eqopt/binomial.hpp"

#include <cmath>
#include <vector>

#include "eqopt/black_scholes.hpp"

namespace eqopt {

namespace {

/// Price under `sigma == 0`: the stock grows deterministically at r - q.
Result<double> deterministic_price(double S, double K, double T, double r, double q,
                                   OptionType type, ExerciseStyle exercise,
                                   int n_steps) noexcept {
    if (exercise == ExerciseStyle::European) {
        return try_bs_price(S, K, T, r, 0.0, q, type);
    }
    const double sign = type == OptionType::Call ? 1.0 : -1.0;
    double best = 0.0;
    for (int i = 0; i <= n_steps; ++i) {
        const double t = T * static_cast<double>(i) /
                         static_cast<double>(n_steps);
        // Discounted payoff without forming the forward (overflow-safe).
        const double value =
            std::fmax(sign * (S * std::exp(-q * t) - K * std::exp(-r * t)), 0.0);
        best = std::fmax(best, value);
    }
    return try_finite(best, "crr_price");
}

/// Backward induction with the American/European choice hoisted out of the
/// inner loop (one instantiation each, no per-node branch).
template <bool American>
double induct(double* values, const double* exercise_value, std::size_t n, double pu,
              double pd) noexcept {
    for (std::size_t i = n; i-- > 0;) {
        const std::size_t offset = n - i;  // index of m = -i
        for (std::size_t j = 0; j <= i; ++j) {
            double cont = pu * values[j + 1] + pd * values[j];
            if constexpr (American) cont = std::fmax(cont, exercise_value[offset + 2 * j]);
            values[j] = cont;
        }
    }
    return values[0];
}

}  // namespace

Result<double> try_crr_price(double S, double K, double T, double r, double sigma,
                             double q, OptionType type, ExerciseStyle exercise,
                             int n_steps, std::span<double> workspace) noexcept {
    if (auto e = try_validate_inputs(S, K, T, sigma); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    if (auto e = try_validate_rates(r, q); !e) [[unlikely]] return std::unexpected(e.error());
    if (n_steps < 1) [[unlikely]] {
        return std::unexpected(Error{Errc::BadSteps, "n_steps", static_cast<double>(n_steps)});
    }

    const double sign = type == OptionType::Call ? 1.0 : -1.0;
    if (T == 0.0) {
        return std::fmax(sign * (S - K), 0.0);
    }
    if (S == 0.0) {
        if (type == OptionType::Call) return 0.0;
        // Exercise now pays K; waiting pays K e^{-rT}, larger when r < 0.
        const auto european = try_finite(K * std::exp(-r * T), "crr_price");
        if (!european) return european;
        return exercise == ExerciseStyle::American ? std::fmax(K, *european) : *european;
    }
    if (K == 0.0) {
        if (type == OptionType::Put) return 0.0;
        // Zero-strike call: American early exercise is optimal iff q > 0.
        if (exercise == ExerciseStyle::American && q > 0.0) return S;
        return try_finite(S * std::exp(-q * T), "crr_price");
    }
    if (sigma == 0.0) {
        return deterministic_price(S, K, T, r, q, type, exercise, n_steps);
    }

    const double dt = T / n_steps;
    const double log_u = sigma * std::sqrt(dt);
    const double u = std::exp(log_u);
    const double d = 1.0 / u;
    const double growth = std::exp((r - q) * dt);
    const double p = (growth - d) / (u - d);
    if (!(0.0 < p && p < 1.0)) [[unlikely]] {
        return std::unexpected(Error{Errc::ProbabilityOutOfRange, "p", p});
    }
    const std::size_t n = static_cast<std::size_t>(n_steps);
    if (workspace.size() < crr_workspace_size(n_steps)) [[unlikely]] {
        return std::unexpected(Error{Errc::WorkspaceTooSmall, "try_crr_price",
                                     static_cast<double>(workspace.size()),
                                     static_cast<double>(crr_workspace_size(n_steps))});
    }
    const double disc = std::exp(-r * dt);
    const double pu = disc * p;
    const double pd = disc * (1.0 - p);
    const double log_s0 = std::log(S);

    // Every node spot is S u^m for m = -n..n; form the exercise values once
    // in log space (O(n) exps) so backward induction does no transcendental
    // work. Step i, node j sits at m = 2j - i.
    double* exercise_value = workspace.data();      // 2n + 1
    double* values = workspace.data() + 2 * n + 1;  // n + 1
    for (std::size_t idx = 0; idx <= 2 * n; ++idx) {
        const double m = static_cast<double>(idx) - static_cast<double>(n);
        exercise_value[idx] = sign * (std::exp(log_s0 + m * log_u) - K);
    }
    for (std::size_t j = 0; j <= n; ++j) values[j] = std::fmax(exercise_value[2 * j], 0.0);

    const double v0 = exercise == ExerciseStyle::American
                          ? induct<true>(values, exercise_value, n, pu, pd)
                          : induct<false>(values, exercise_value, n, pu, pd);
    return try_finite(v0, "crr_price");
}

double crr_price(double S, double K, double T, double r, double sigma,
                 double q, OptionType type, ExerciseStyle exercise,
                 int n_steps) {
    std::vector<double> workspace(crr_workspace_size(n_steps));
    return value_or_throw(
        try_crr_price(S, K, T, r, sigma, q, type, exercise, n_steps, workspace));
}

double early_exercise_premium(double S, double K, double T, double r,
                              double sigma, double q, OptionType type,
                              int n_steps) {
    std::vector<double> workspace(crr_workspace_size(n_steps));
    const double amer = value_or_throw(try_crr_price(
        S, K, T, r, sigma, q, type, ExerciseStyle::American, n_steps, workspace));
    const double euro = value_or_throw(try_crr_price(
        S, K, T, r, sigma, q, type, ExerciseStyle::European, n_steps, workspace));
    return std::fmax(amer - euro, 0.0);
}

}  // namespace eqopt
