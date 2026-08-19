#include "eqopt/binomial.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "eqopt/black_scholes.hpp"

namespace eqopt {

namespace {

/// Price under `sigma == 0`: the stock grows deterministically at r - q.
double deterministic_price(double S, double K, double T, double r, double q,
                           OptionType type, ExerciseStyle exercise,
                           int n_steps) {
    if (exercise == ExerciseStyle::European) {
        return bs_price(S, K, T, r, 0.0, q, type);
    }
    const double sign = type == OptionType::Call ? 1.0 : -1.0;
    double best = 0.0;
    for (int i = 0; i <= n_steps; ++i) {
        const double t = T * static_cast<double>(i) /
                         static_cast<double>(n_steps);
        const double s_t = S * std::exp((r - q) * t);
        const double value =
            std::exp(-r * t) * std::fmax(sign * (s_t - K), 0.0);
        best = std::fmax(best, value);
    }
    return best;
}

}  // namespace

double crr_price(double S, double K, double T, double r, double sigma,
                 double q, OptionType type, ExerciseStyle exercise,
                 int n_steps) {
    validate_inputs(S, K, T, sigma);
    validate_rates(r, q);
    if (n_steps < 1) {
        std::ostringstream oss;
        oss << "n_steps must be >= 1, got " << n_steps;
        throw std::invalid_argument(oss.str());
    }

    const double sign = type == OptionType::Call ? 1.0 : -1.0;
    if (T == 0.0) {
        return std::fmax(sign * (S - K), 0.0);
    }
    if (S == 0.0) {
        if (type == OptionType::Call) return 0.0;
        return exercise == ExerciseStyle::American ? K : K * std::exp(-r * T);
    }
    if (K == 0.0) {
        if (type == OptionType::Put) return 0.0;
        // Zero-strike call: American early exercise is optimal iff q > 0.
        if (exercise == ExerciseStyle::American && q > 0.0) return S;
        return S * std::exp(-q * T);
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
    if (!(0.0 < p && p < 1.0)) {
        std::ostringstream oss;
        oss << "risk-neutral probability p=" << p
            << " outside (0, 1); increase n_steps or check r, q, sigma";
        throw std::invalid_argument(oss.str());
    }
    const double disc = std::exp(-r * dt);
    const double pu = disc * p;
    const double pd = disc * (1.0 - p);
    const double log_s0 = std::log(S);

    // Terminal payoffs S * u^j * d^(n-j), j = 0..n, evaluated in log space
    // for numerical stability at large n. Single value vector, updated in
    // place during backward induction: O(n) memory.
    std::vector<double> values(static_cast<std::size_t>(n_steps) + 1);
    for (int j = 0; j <= n_steps; ++j) {
        const double log_s = log_s0 + (2.0 * j - n_steps) * log_u;
        values[static_cast<std::size_t>(j)] =
            std::fmax(sign * (std::exp(log_s) - K), 0.0);
    }

    const bool american = exercise == ExerciseStyle::American;
    for (int i = n_steps - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            const std::size_t sj = static_cast<std::size_t>(j);
            double cont = pu * values[sj + 1] + pd * values[sj];
            if (american) {
                const double node =
                    std::exp(log_s0 + (2.0 * j - i) * log_u);
                cont = std::fmax(cont, sign * (node - K));
            }
            values[sj] = cont;
        }
    }
    return values[0];
}

double early_exercise_premium(double S, double K, double T, double r,
                              double sigma, double q, OptionType type,
                              int n_steps) {
    const double amer = crr_price(S, K, T, r, sigma, q, type,
                                  ExerciseStyle::American, n_steps);
    const double euro = crr_price(S, K, T, r, sigma, q, type,
                                  ExerciseStyle::European, n_steps);
    return std::fmax(amer - euro, 0.0);
}

}  // namespace eqopt
