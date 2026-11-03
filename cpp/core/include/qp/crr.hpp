// qp/crr.hpp: the one Cox-Ross-Rubinstein lattice shared by the equity and
// FX engines (W20, Round 8).
//
// Plain idea. Cut [0, T] into n steps; the spot moves up by u or down by
// d = 1/u each step with risk-neutral probability p; roll the payoff back
// to today, taking the larger of continuation and exercise at every node
// for American options.
//
// Formal statement. dt = T/n, u = exp(sigma sqrt(dt)), d = 1/u,
// p = (exp((r - q) dt) - d) / (u - d) (Cox, Ross and Rubinstein 1979,
// J. Financial Economics 7, eqs. 2-3, with Merton's continuous yield q).
// FX is the same lattice with r = r_d and q = r_f (Garman and Kohlhagen
// 1983). V_{i,k} = pu V_{i+1,k+1} + pd V_{i+1,k}, pu = e^{-r dt} p,
// pd = e^{-r dt} (1 - p). Node spots S u^m, m = -n..n, are formed once as
// exp(ln S + m ln u), so the induction does no transcendental work. Same
// operations and order as the Python kernel qp_core.trees.crr_lattice.
//
// Worked example. S = K = 100, T = 1, r = 5 %, q = 2 %, sigma = 20 %,
// European call, n = 1: p = 0.52504, value 11.07354 (Python corpus row 0,
// tests/data/kernel_corpus.json).
//
// The caller validates inputs, handles T = 0, sigma = 0, S = 0 and K = 0,
// checks that p lies in (0, 1) and sizes the workspace (3 n + 2 doubles);
// these functions only do arithmetic and never fail.

#pragma once

#include <cmath>
#include <cstddef>

namespace qp {

/// Per-step constants of a CRR lattice.
struct CrrStep {
    double log_u;  ///< sigma sqrt(dt)
    double p;      ///< risk-neutral up-probability; valid when 0 < p < 1
    double pu;     ///< e^{-r dt} p
    double pd;     ///< e^{-r dt} (1 - p)
};

/// Step constants for n steps over T with rates r and q (r_d, r_f for FX).
[[nodiscard]] inline CrrStep crr_step(double T, double r, double q, double sigma,
                                      std::size_t n) noexcept {
    const double dt = T / static_cast<double>(n);
    const double log_u = sigma * std::sqrt(dt);
    const double u = std::exp(log_u);
    const double d = 1.0 / u;
    const double p = (std::exp((r - q) * dt) - d) / (u - d);
    const double disc = std::exp(-r * dt);
    return {log_u, p, disc * p, disc * (1.0 - p)};
}

namespace detail {
template <bool American>
double crr_induct(double* values, const double* exercise_value, std::size_t n, double pu,
                  double pd) noexcept {
    for (std::size_t i = n; i-- > 0;) {
        const std::size_t offset = n - i;  // index of m = -i
        for (std::size_t j = 0; j <= i; ++j) {
            double cont = pu * values[j + 1] + pd * values[j];
            if constexpr (American) {
                // A plain compare, not std::fmax: fmax's NaN rule makes GCC
                // emit a libm call per node (5x slower at n = 1000). Values
                // are finite here, where the two agree bit for bit.
                const double ex = exercise_value[offset + 2 * j];
                cont = cont < ex ? ex : cont;
            }
            values[j] = cont;
        }
    }
    return values[0];
}
}  // namespace detail

/// Lattice value for spot S > 0, strike K > 0, phi = +1 (call) or -1 (put),
/// n >= 1 steps with constants `st` (0 < st.p < 1), in a caller-owned
/// workspace of at least 3 n + 2 doubles.
[[nodiscard]] inline double crr_lattice(double S, double K, double phi, bool american,
                                        std::size_t n, const CrrStep& st,
                                        double* workspace) noexcept {
    double* exercise_value = workspace;      // 2n + 1
    double* values = workspace + 2 * n + 1;  // n + 1
    const double log_s0 = std::log(S);
    for (std::size_t idx = 0; idx <= 2 * n; ++idx) {
        const double m = static_cast<double>(idx) - static_cast<double>(n);
        exercise_value[idx] = phi * (std::exp(log_s0 + m * st.log_u) - K);
    }
    for (std::size_t j = 0; j <= n; ++j) values[j] = std::fmax(exercise_value[2 * j], 0.0);
    return american ? detail::crr_induct<true>(values, exercise_value, n, st.pu, st.pd)
                    : detail::crr_induct<false>(values, exercise_value, n, st.pu, st.pd);
}

}  // namespace qp
