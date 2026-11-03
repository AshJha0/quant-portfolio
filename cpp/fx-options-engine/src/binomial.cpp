#include "fxopt/binomial.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "fxopt/garman_kohlhagen.hpp"
#include "qp/crr.hpp"

namespace fxopt {

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

    const auto n = static_cast<std::size_t>(steps);
    const qp::CrrStep st = qp::crr_step(T, r_d, r_f, sigma, n);
    if (!(0.0 < st.p && st.p < 1.0)) [[unlikely]] {
        return std::unexpected(Error{Errc::ProbabilityOutOfRange, "p", st.p});
    }
    if (workspace.size() < binomial_workspace_size(steps)) [[unlikely]] {
        return std::unexpected(Error{Errc::WorkspaceTooSmall, "try_binomial_price",
                                     static_cast<double>(workspace.size()),
                                     static_cast<double>(binomial_workspace_size(steps))});
    }
    // Shared lattice (qp/crr.hpp, W20): the equity tree with q = r_f.
    const double v0 =
        qp::crr_lattice(S, K, p_sign, exercise == Exercise::American, n, st, workspace.data());
    return try_finite_result(v0, "binomial_price");
}

double binomial_price(double S, double K, double T, double r_d, double r_f, double sigma,
                      OptionType type, int steps, Exercise exercise) {
    std::vector<double> workspace(binomial_workspace_size(steps));
    return value_or_throw(
        try_binomial_price(S, K, T, r_d, r_f, sigma, type, steps, exercise, workspace));
}

}  // namespace fxopt
