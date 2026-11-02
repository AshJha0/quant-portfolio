// Leisen-Reimer tree; see include/eqopt/lr_tree.hpp.
#include "eqopt/lr_tree.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace eqopt {

double peizer_pratt(double z, int n) {
    if (n < 1 || n % 2 == 0)
        throw std::invalid_argument("peizer_pratt: n must be a positive odd integer");
    const auto nd = static_cast<double>(n);
    const double a = z / (nd + 1.0 / 3.0 + 0.1 / (nd + 1.0));
    const double root = std::sqrt(-std::expm1(-a * a * (nd + 1.0 / 6.0)));
    return 0.5 + std::copysign(0.5, z) * root;
}

double lr_price(double S, double K, double T, double r, double sigma, double q, OptionType type,
                ExerciseStyle exercise, int n_steps) {
    validate_inputs(S, K, T, sigma);
    validate_rates(r, q);
    if (n_steps < 1) throw std::invalid_argument("lr_price: n_steps must be >= 1");
    if (S <= 0.0 || K <= 0.0 || T <= 0.0 || sigma <= 0.0)
        throw std::invalid_argument("lr_price: S, K, T and sigma must be strictly positive");

    const int n = n_steps % 2 == 1 ? n_steps : n_steps + 1;
    const double dt = T / n;
    const auto [d1, d2] = d1_d2(S, K, T, r, sigma, q);
    const double p_prime = peizer_pratt(d1, n);
    const double p = peizer_pratt(d2, n);
    const double growth = std::exp((r - q) * dt);
    const double u = growth * p_prime / p;
    const double d = (growth - p * u) / (1.0 - p);
    if (!(p > 0.0 && p < 1.0) || !(d < u) || !(d > 0.0))
        throw std::invalid_argument("lr_price: Leisen-Reimer tree invalid for these inputs");

    const double disc = std::exp(-r * T / n);
    const double pu = disc * p;
    const double pd = disc * (1.0 - p);
    const double sign = type == OptionType::Call ? 1.0 : -1.0;
    const bool american = exercise == ExerciseStyle::American;
    const double log_u = std::log(u);
    const double log_d = std::log(d);

    std::vector<double> v(static_cast<std::size_t>(n) + 1);
    for (int j = 0; j <= n; ++j)
        v[static_cast<std::size_t>(j)] =
            std::fmax(sign * (S * std::exp(j * log_u + (n - j) * log_d) - K), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        for (int k = 0; k <= i; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            double cont = pu * v[kk + 1] + pd * v[kk];
            if (american)
                cont = std::fmax(cont, sign * (S * std::exp(k * log_u + (i - k) * log_d) - K));
            v[kk] = cont;
        }
    }
    return v[0];
}

}  // namespace eqopt
