// eqvar/kernels.hpp: noexcept kernel API for the hot paths.
//
// Each try_* function has the conventions, validation order and arithmetic
// of its throwing counterpart (which is now a thin wrapper over it), so the
// returned numbers are the same bit for bit. Work buffers come from the
// caller as std::span or a reusable workspace object; once those are sized,
// no kernel allocates, throws or creates a thread. Errors come back as
// eqvar::Error values (see error.hpp).
//
// Monte Carlo: try_monte_carlo_var_factored takes the Cholesky factor the
// caller computed once (try_cholesky_lower or cholesky()) and fuses scenario
// generation with the P&L map, so it never stores the n_paths x n scenario
// panel. It draws exactly the same numbers as monte_carlo_var().

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "eqvar/error.hpp"
#include "eqvar/historical.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/monte_carlo.hpp"
#include "eqvar/parametric.hpp"

namespace eqvar {

// ---------------------------------------------------------------- checks

/// noexcept form of validate_alpha().
[[nodiscard]] Result<void> try_validate_alpha(double alpha) noexcept;

/// noexcept form of validate_pnl().
[[nodiscard]] Result<void> try_validate_pnl(std::span<const double> pnl,
                                            std::size_t min_obs) noexcept;

/// noexcept form of validate_exposures().
[[nodiscard]] Result<void> try_validate_exposures(std::span<const double> exposures,
                                                  const char* who) noexcept;

// ----------------------------------------------------------- historical

/// noexcept var_es_order_statistic(). `workspace` must hold pnl.size()
/// doubles; the loss vector is partitioned there with std::nth_element
/// (O(n) selection, no sort).
[[nodiscard]] Result<TailRisk> try_var_es_order_statistic(std::span<const double> pnl,
                                                          double alpha,
                                                          std::span<double> workspace) noexcept;

/// noexcept historical_var(); workspace as above.
[[nodiscard]] Result<double> try_historical_var(std::span<const double> pnl, double alpha,
                                                std::span<double> workspace) noexcept;

/// noexcept expected_shortfall(); workspace as above.
[[nodiscard]] Result<double> try_expected_shortfall(std::span<const double> pnl, double alpha,
                                                    std::span<double> workspace) noexcept;

namespace detail {
/// Engine of the deprecated quantile_linear(); workspace holds x.size()
/// doubles. Two selections (nth_element for x(lo), then the minimum of the
/// upper part for x(lo+1)) give the same two order statistics as the old
/// full sort. Not part of the supported API.
[[nodiscard]] Result<double> try_quantile_linear(std::span<const double> x, double q,
                                                 std::span<double> workspace) noexcept;
}  // namespace detail

// ---------------------------------------------------------------- matrix

/// noexcept matvec() into caller storage `y` (a.rows() doubles). Rows are
/// processed four at a time with independent accumulators; each row is
/// still summed left to right, so every y[i] equals matvec() exactly.
/// This is also the scenario-P&L batch kernel: pnl = panel * exposures.
[[nodiscard]] Result<void> try_matvec(const Matrix& a, std::span<const double> x,
                                      std::span<double> y) noexcept;

/// noexcept quad_form().
[[nodiscard]] Result<double> try_quad_form(std::span<const double> w, const Matrix& s) noexcept;

/// Doubles of scratch try_cholesky_lower() needs: n * n.
constexpr std::size_t cholesky_workspace_size(std::size_t n) noexcept { return n * n; }

/// Cholesky with the same validation, plain attempt and capped jitter
/// ladder as cholesky(cov, Repair::None), into caller storage. `lower`
/// (n * n, row-major) receives L with zeros above the diagonal, identical
/// to cholesky().lower; `scratch` holds the jittered copy. Returns the
/// jitter added (0 when none). Past the jitter cap it returns
/// Errc::CholIndefinite; that message omits the most negative eigenvalue
/// (computing it would allocate). PSD repair is not offered here: repair
/// is a start-of-day step, use cholesky(cov, Repair) for it.
[[nodiscard]] Result<double> try_cholesky_lower(const Matrix& cov, std::span<double> lower,
                                                std::span<double> scratch) noexcept;

// ------------------------------------------------------------ parametric

/// noexcept portfolio_sigma().
[[nodiscard]] Result<double> try_portfolio_sigma(std::span<const double> exposures,
                                                 const Matrix& cov) noexcept;

/// noexcept parametric_var().
[[nodiscard]] Result<double> try_parametric_var(std::span<const double> exposures,
                                                const Matrix& cov, double alpha = 0.01,
                                                Dist dist = Dist::Normal, double df = 6.0,
                                                double mean = 0.0,
                                                int horizon_days = 1) noexcept;

// ----------------------------------------------------------- Monte Carlo

/// Reusable buffers for the Monte Carlo kernels. Size once with reserve();
/// the kernels then never allocate. One workspace per concurrent caller.
class McVarWorkspace {
public:
    /// Room for `n_factors` factors and `n_paths` scenarios. May allocate.
    void reserve(std::size_t n_factors, std::size_t n_paths) {
        if (z.size() < n_factors) z.resize(n_factors);
        if (row.size() < n_factors) row.resize(n_factors);
        if (pnl.size() < n_paths) pnl.resize(n_paths);
        if (loss.size() < n_paths) loss.resize(n_paths);
        if (work.size() < n_paths) work.resize(n_paths);
    }
    std::vector<double> z, row, pnl, loss, work;
};

/// noexcept mc_tail_metrics(): VaR, ES, bootstrap SE and KDE SE of one
/// scenario P&L sample, using ws.loss and ws.work (pnl.size() each).
[[nodiscard]] Result<MonteCarloResult> try_mc_tail_metrics(std::span<const double> pnl,
                                                           double alpha, std::size_t n_boot,
                                                           std::uint64_t seed,
                                                           McVarWorkspace& ws) noexcept;

/// noexcept Monte Carlo VaR from a precomputed factor `lower` (n * n,
/// row-major, the Cholesky factor of cov for Normal, of cov (df-2)/df for
/// StudentT; that is what monte_carlo_var() factors). Draws, P&L and
/// metrics equal monte_carlo_var() with the same seed bit for bit. The
/// scenario P&L stays in ws.pnl for the caller.
[[nodiscard]] Result<MonteCarloResult> try_monte_carlo_var_factored(
    std::span<const double> exposures, std::span<const double> lower, double alpha,
    std::size_t n_paths, Dist dist, double df, std::uint64_t seed, std::size_t n_boot,
    McVarWorkspace& ws) noexcept;

}  // namespace eqvar
