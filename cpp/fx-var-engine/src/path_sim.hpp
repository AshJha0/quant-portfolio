// path_sim.hpp (private to fxvar): one Monte Carlo factor scenario on its
// own Philox stream (W4(b)), shared by simulate_factor_returns and
// try_simulate_factor_returns. The Rust twin is
// fx_var_engine::monte_carlo::simulate_path; the library is built with
// -ffp-contract=off so the two agree to the last bit on the same libm.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "fxvar/monte_carlo.hpp"
#include "qp/path_rng.hpp"

// ------------------------------------------------------------ W4(b) paths
namespace fxvar::detail {

/// Poisson(mu) by sequential inversion from one uniform of the path stream,
/// mu <= 700 so exp(-mu) does not underflow. Same algorithm and operation
/// order as the Rust twin (fx_var_engine::monte_carlo::poisson).
inline std::uint64_t path_poisson(qp::PathRng& rng, double mu) noexcept {
  if (mu <= 0.0) return 0;
  const double u = rng.uniform();
  double p = std::exp(-mu);
  double cdf = p;
  std::uint64_t k = 0;
  while (u > cdf && k < 100000) {
    ++k;
    p *= mu / static_cast<double>(k);
    cdf += p;
    if (p == 0.0 && cdf < u) break;
  }
  return k;
}

/// Scenario s of a factor simulation (W4(b)): Philox stream (seed, s), the
/// k normals first (AS 241, batched), x = L z, then the Student-t mixing
/// draw or the jump count and one normal per jump factor with sd > 0.
/// `row` receives the k factor returns: the normals are drawn into it and
/// x = L z is formed in place from the last row up (row i needs z_0..z_i
/// only, and z_i is overwritten after its last use).
inline void simulate_path(const double* lower, std::size_t k, McDist dist, double df,
                          double t_scale, double jump_mu, const int* jump_col,
                          const double* jump_mean, const double* jump_std, std::size_t n_jump,
                          std::uint64_t seed, std::uint64_t s, double* row) noexcept {
  qp::PathRng rng(seed, s);
  for (std::size_t j0 = 0; j0 < k; j0 += 2 * qp::kPathLanes)
    rng.fill_normals(row + j0, std::min(k - j0, 2 * qp::kPathLanes));
  for (std::size_t i = k; i-- > 0;) {
    const double* li = lower + i * k;
    double acc = 0.0;
    for (std::size_t j = 0; j <= i; ++j) acc += li[j] * row[j];
    row[i] = acc;
  }
  if (dist == McDist::kStudentT) {
    const double w = rng.chi_squared(df) / df;
    const double m = t_scale / std::sqrt(w);
    for (std::size_t i = 0; i < k; ++i) row[i] *= m;
  } else if (dist == McDist::kJump) {
    const auto nj = static_cast<double>(path_poisson(rng, jump_mu));
    for (std::size_t j = 0; j < n_jump; ++j) {
      const double zj = jump_std[j] > 0.0 ? rng.normal() : 0.0;
      if (nj > 0.0) {
        double& x = row[jump_col[j]];
        x = x + nj * jump_mean[j] + jump_std[j] * std::sqrt(nj) * zj;
      }
    }
  }
}

}  // namespace fxvar::detail
