#include "eqopt/binomial.hpp"

#include <cmath>
#include <vector>

#include "eqopt/black_scholes.hpp"
#include "qp/crr.hpp"

namespace eqopt {

namespace {

/// Price under `sigma == 0`: the stock grows deterministically at r - q.
Result<double> deterministic_price(double S, double K, double T, double r, double q,
                                   OptionType type, ExerciseStyle exercise, int n_steps) noexcept {
    if (exercise == ExerciseStyle::European) {
        return try_bs_price(S, K, T, r, 0.0, q, type);
    }
    const double sign = type == OptionType::Call ? 1.0 : -1.0;
    double best = 0.0;
    for (int i = 0; i <= n_steps; ++i) {
        const double t = T * static_cast<double>(i) / static_cast<double>(n_steps);
        // Discounted payoff without forming the forward (overflow-safe).
        const double value = std::fmax(sign * (S * std::exp(-q * t) - K * std::exp(-r * t)), 0.0);
        best = std::fmax(best, value);
    }
    return try_finite(best, "crr_price");
}

}  // namespace

Result<double> try_crr_price(double S, double K, double T, double r, double sigma, double q,
                             OptionType type, ExerciseStyle exercise, int n_steps,
                             std::span<double> workspace) noexcept {
    if (auto e = try_validate_inputs(S, K, T, sigma); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    if (auto e = try_validate_rates(r, q); !e) [[unlikely]]
        return std::unexpected(e.error());
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

    const auto n = static_cast<std::size_t>(n_steps);
    const qp::CrrStep st = qp::crr_step(T, r, q, sigma, n);
    if (!(0.0 < st.p && st.p < 1.0)) [[unlikely]] {
        return std::unexpected(Error{Errc::ProbabilityOutOfRange, "p", st.p});
    }
    if (workspace.size() < crr_workspace_size(n_steps)) [[unlikely]] {
        return std::unexpected(Error{Errc::WorkspaceTooSmall, "try_crr_price",
                                     static_cast<double>(workspace.size()),
                                     static_cast<double>(crr_workspace_size(n_steps))});
    }
    // Shared lattice (qp/crr.hpp, W20), also used by fxopt.
    const double v0 =
        qp::crr_lattice(S, K, sign, exercise == ExerciseStyle::American, n, st, workspace.data());
    return try_finite(v0, "crr_price");
}

double crr_price(double S, double K, double T, double r, double sigma, double q, OptionType type,
                 ExerciseStyle exercise, int n_steps) {
    std::vector<double> workspace(crr_workspace_size(n_steps));
    return value_or_throw(try_crr_price(S, K, T, r, sigma, q, type, exercise, n_steps, workspace));
}

double early_exercise_premium(double S, double K, double T, double r, double sigma, double q,
                              OptionType type, int n_steps) {
    std::vector<double> workspace(crr_workspace_size(n_steps));
    const double amer = value_or_throw(
        try_crr_price(S, K, T, r, sigma, q, type, ExerciseStyle::American, n_steps, workspace));
    const double euro = value_or_throw(
        try_crr_price(S, K, T, r, sigma, q, type, ExerciseStyle::European, n_steps, workspace));
    return std::fmax(amer - euro, 0.0);
}

}  // namespace eqopt
