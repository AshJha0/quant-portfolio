/// \file monte_carlo.hpp
/// \brief Monte Carlo pricing of European options under GBM (exact scheme).
///
/// The terminal stock price is simulated exactly:
/// `S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z)`, `Z ~ N(0,1)`,
/// so there is *no* time-discretisation bias: only statistical error,
/// reported as a standard error and 95% confidence interval.
///
/// Variance reduction (mirrors the Python reference `eq_options/monte_carlo.py`):
///  - antithetic variates: pairs `(Z, -Z)`; the standard error is computed
///    on the pair averages (the correct estimator for correlated pairs);
///  - control variate: the discounted terminal stock `exp(-rT) S_T` with
///    known mean `S exp(-qT)` (martingale property); the optimal
///    coefficient is estimated from the sample covariance.
///
/// Reproducibility: driven by std::mt19937_64 with an explicit seed. Normal
/// deviates are produced by an in-house inverse-CDF transform (Acklam
/// initial guess + one Halley refinement against the erfc-based CDF) rather
/// than std::normal_distribution, whose algorithm is implementation-defined.
/// Same (seed, threads) => identical results on a given compiler and libm.
/// Across platforms the last bits can differ (libm exp/log are not
/// correctly rounded; compilers may contract to FMA).
/// Multithreading partitions the paths into `threads` chunks with
/// independently seeded (splitmix64-derived) RNG streams, so the result is
/// deterministic for a given `threads` value. The chunks run on a persistent
/// ThreadPool (default_thread_pool() for mc_price(), or the caller's pool
/// for try_mc_price()); no thread is created per call. Which pool thread
/// runs a chunk does not change the result.
///
/// Conventions: continuously compounded annualised `r`, `q` (ACT/365F),
/// `T` in years, `sigma` annualised.

#ifndef EQOPT_MONTE_CARLO_HPP
#define EQOPT_MONTE_CARLO_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "eqopt/black_scholes.hpp"
#include "eqopt/thread_pool.hpp"

namespace eqopt {

/// Monte Carlo estimate with statistical error bars.
struct MCResult {
    double price;      ///< Point estimate, currency units.
    double std_error;  ///< Standard error of the estimator.
    double ci_low;     ///< Lower edge of the two-sided 95% CI.
    double ci_high;    ///< Upper edge of the two-sided 95% CI.
    std::size_t n_paths;  ///< Simulated paths (antithetic pairs count as 2).

    /// \brief True if `x` lies inside the 95% confidence interval.
    [[nodiscard]] bool contains(double x) const noexcept {
        return ci_low <= x && x <= ci_high;
    }

    /// Field-by-field exact equality (used by reproducibility tests).
    bool operator==(const MCResult&) const = default;
};

/// \brief Deterministic standard normal deviate from a mt19937_64 draw.
///
/// Maps a 64-bit draw to a uniform in (0, 1) (53-bit mantissa, offset to
/// avoid 0), then applies the inverse normal CDF: Acklam's rational
/// approximation refined with one Halley step against eqopt::norm_cdf,
/// giving ~1e-15 relative accuracy. Exposed for testing.
/// \param u64 Raw 64-bit uniform integer draw.
/// \return Standard normal deviate.
double normal_from_u64(std::uint64_t u64) noexcept;

/// \brief Monte Carlo price of a European option under exact-scheme GBM.
///
/// \param S,K   Spot and strike (currency units), `>= 0`.
/// \param T     Time to expiry in years (ACT/365F), `>= 0`.
/// \param r     Continuously compounded annualised risk-free rate.
/// \param sigma Annualised volatility, `>= 0`.
/// \param q     Continuously compounded annualised dividend yield.
/// \param type  Option payoff direction.
/// \param n_paths Total number of paths (rounded up to even when
///              `antithetic` is set).
/// \param antithetic Use antithetic pairs `(Z, -Z)`.
/// \param control_variate Use the discounted terminal stock as a control
///              variate; the coefficient is fitted on the antithetic pair
///              averages (the independent samples).
/// \param seed  Explicit std::mt19937_64 seed. Same (seed, threads) =>
///              identical result on the same toolchain.
/// \param threads Number of path chunks, `>= 1`. Each chunk is a
///              disjoint, deterministically assigned range of paths with its
///              own splitmix64-derived RNG stream; chunks run in parallel on
///              default_thread_pool(). Different `threads` values use
///              different streams (statistically equivalent, not bit-equal).
/// \return Price estimate with standard error and 95% CI.
/// \throws std::invalid_argument on invalid inputs, fewer than two
///         independent samples (three with the control variate; an
///         antithetic pair is one sample), `threads == 0`, or simulated
///         prices that overflow.
/// \note `T == 0` or `sigma == 0` are deterministic; the exact
///       Black–Scholes value is returned with `std_error = 0`.
MCResult mc_price(double S, double K, double T, double r, double sigma,
                  double q = 0.0, OptionType type = OptionType::Call,
                  std::size_t n_paths = 100000, bool antithetic = true,
                  bool control_variate = true, std::uint64_t seed = 42,
                  unsigned threads = 1);

/// \brief Reusable path buffers for try_mc_price().
///
/// Size it once with reserve() for the largest run; try_mc_price() then
/// never allocates. Not thread-safe: one workspace per concurrent caller.
class McWorkspace {
public:
    /// \brief Make room for `n_paths` paths (and the control column when
    ///        `control_variate`). May allocate; call it outside the hot path.
    void reserve(std::size_t n_paths, bool control_variate) {
        const std::size_t n = n_paths + 1;  // antithetic rounds up to even
        if (payoff_.size() < n) payoff_.resize(n);
        if (control_variate && control_.size() < n) control_.resize(n);
    }
    [[nodiscard]] std::size_t capacity() const noexcept { return payoff_.size(); }

private:
    friend Result<MCResult> try_mc_price(double, double, double, double, double, double,
                                         OptionType, std::size_t, bool, bool, std::uint64_t,
                                         unsigned, McWorkspace&, ThreadPool*) noexcept;
    std::vector<double> payoff_;
    std::vector<double> control_;
};

/// \brief noexcept Monte Carlo price on caller-owned buffers and pool.
///
/// Same arguments, draws and arithmetic as mc_price(), so for the same
/// `(seed, threads)` the result is bit-for-bit the same. `threads` is the
/// number of deterministic RNG chunks; they run on `pool` (or on the calling
/// thread when `pool` is null), and the result does not depend on the pool
/// size. `ws` must have been reserved for `n_paths` (Errc::WorkspaceTooSmall
/// otherwise). No allocation and no thread creation on any path.
[[nodiscard]] Result<MCResult> try_mc_price(double S, double K, double T, double r,
                                            double sigma, double q, OptionType type,
                                            std::size_t n_paths, bool antithetic,
                                            bool control_variate, std::uint64_t seed,
                                            unsigned threads, McWorkspace& ws,
                                            ThreadPool* pool) noexcept;

}  // namespace eqopt

#endif  // EQOPT_MONTE_CARLO_HPP
