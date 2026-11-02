// Monte Carlo pricing of FX options under the domestic risk-neutral measure.
//
// Under the domestic money-market numeraire the spot follows
//   dS/S = (r_d - r_f) dt + sigma dW^d,
// so terminal spot is sampled exactly:
//   S_T = S exp((r_d - r_f - sigma^2/2) T + sigma sqrt(T) Z).
//
// Variance reduction:
//   * Antithetic variates -- pairs (Z, -Z), estimator averaged per pair.
//   * Control variate -- the discounted terminal spot e^{-r_d T} S_T has
//     known mean S e^{-r_f T} (the value today of one unit of foreign
//     currency delivered at T); the coefficient is fitted on the antithetic
//     pair averages, the independent samples.
//
// Determinism: every run takes an explicit std::mt19937_64 seed, draws
// normals by inverse-CDF transform of the raw generator output (no
// implementation-defined std::normal_distribution), and mc_price() is
// single-threaded -- identical seeds give identical results on the same
// toolchain. try_mc_price() can split the draws into `threads` chunks on a
// persistent ThreadPool; threads = 1 is the legacy stream bit for bit. Across
// platforms the last bits can differ (libm exp/log are not correctly
// rounded; compilers may contract to FMA).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fxopt/common.hpp"
#include "fxopt/thread_pool.hpp"

namespace fxopt {

/// rief Fewest independent draws (paths, or antithetic pairs) for which
///        try_mc_price() hands its chunks to a thread pool. Below it the
///        chunks run inline on the caller: results are bit-identical
///        either way. Measured crossover (Core 5 120U, WSL2, GCC 15.2,
///        antithetic + control variate): 4 threads lose at 10k paths
///        (531 us vs 424 us) and win from about 20k paths up; see
///        docs/METHODOLOGY.md, "Monte Carlo threading crossover".
inline constexpr std::size_t kMcParallelMinDraws = 16384;

/// Monte Carlo estimate with sampling-error diagnostics.
struct MCResult {
    double price;          ///< point estimate, domestic ccy per foreign notional
    double std_error;      ///< standard error of the estimate
    double ci_low;         ///< 95% CI lower bound (price - 1.96 SE)
    double ci_high;        ///< 95% CI upper bound (price + 1.96 SE)
    std::int64_t n_paths;  ///< underlying draws (antithetic pairs count as 2)
    std::string method;    ///< variance-reduction techniques applied
};

/// Monte Carlo GK price of a European FX vanilla.  Requires T > 0 and at
/// least two independent samples (throws otherwise).  n_paths is rounded up to even when
/// antithetic.  The RNG is std::mt19937_64 with the given seed;
/// single-threaded and fully deterministic.
///
/// The standard error is computed from *independent* samples: the pair
/// averages when antithetic (mirrored draws are perfectly dependent), the
/// raw payoffs otherwise.  Fewer than two independent samples (three with
/// the control variate, whose fit uses one degree of freedom) throws
/// std::invalid_argument: one antithetic pair has no standard error.
/// Explicit RNG seed. The constructor is explicit so that a bool or an
/// integer meant for another parameter cannot bind to the seed:
/// `mc_price(..., 100000, true, true, 42)` does not compile.
struct Seed {
    std::uint64_t value;
    constexpr explicit Seed(std::uint64_t v) noexcept : value(v) {}
};

/// Monte Carlo settings as one strongly typed value.
struct McConfig {
    std::int64_t n_paths = 100'000;  ///< draws (antithetic pairs count as 2)
    Seed seed{0};                    ///< RNG seed
    bool antithetic = true;          ///< mirror each draw z -> -z
    bool control_variate = true;     ///< discounted terminal spot as control
};

MCResult mc_price(double S, double K, double T, double r_d, double r_f, double sigma,
                  OptionType type, const McConfig& config);

MCResult mc_price(double S, double K, double T, double r_d, double r_f, double sigma,
                  OptionType type, std::int64_t n_paths = 100'000, Seed seed = Seed{0},
                  bool antithetic = true, bool control_variate = true);

/// Former positional signature with a raw integer seed: deleted, because a
/// bool argument converted to it silently. Use Seed{...} or McConfig.
MCResult mc_price(double S, double K, double T, double r_d, double r_f, double sigma,
                  OptionType type, std::int64_t n_paths, std::uint64_t seed, bool antithetic = true,
                  bool control_variate = true) = delete;

/// Monte Carlo estimate without the std::string of MCResult (the method is
/// a static string), so returning it never allocates.
struct McEstimate {
    double price;
    double std_error;
    double ci_low;
    double ci_high;
    std::int64_t n_paths;
    const char* method;
};

/// Reusable path buffers for try_mc_price(). Size it once with reserve()
/// for the largest run; try_mc_price() then never allocates. One workspace
/// per concurrent caller.
class McWorkspace {
public:
    /// Make room for `n_paths` paths (and the control column when
    /// `control_variate`). May allocate; call it outside the hot path.
    void reserve(std::int64_t n_paths, bool control_variate) {
        const std::size_t n = static_cast<std::size_t>(n_paths) + 1;  // antithetic rounds up
        if (discounted_.size() < n) discounted_.resize(n);
        if (control_variate && control_.size() < n) control_.resize(n);
    }
    [[nodiscard]] std::size_t capacity() const noexcept { return discounted_.size(); }

private:
    friend Result<McEstimate> try_mc_price(double, double, double, double, double, double,
                                           OptionType, std::int64_t, Seed, bool, bool, unsigned,
                                           McWorkspace&, ThreadPool*) noexcept;
    std::vector<double> discounted_;
    std::vector<double> control_;
};

/// noexcept Monte Carlo price on caller-owned buffers and pool.
///
/// Same checks and arithmetic as mc_price(). With `threads == 1` the draws
/// come from mt19937_64(seed), the legacy stream, so the result equals
/// mc_price() bit for bit. With `threads > 1` the draws are split into
/// `threads` chunks, chunk c seeded with splitmix64(seed + k c); the result
/// is then fixed for a given (seed, threads) whatever `pool` runs it (null
/// runs the chunks on the calling thread), and statistically equivalent to,
/// not bit-equal with, the single stream. `ws` must have been reserved for
/// `n_paths` (Errc::WorkspaceTooSmall otherwise). No allocation, no thread
/// creation.
[[nodiscard]] Result<McEstimate> try_mc_price(double S, double K, double T, double r_d, double r_f,
                                              double sigma, OptionType type, std::int64_t n_paths,
                                              Seed seed, bool antithetic, bool control_variate,
                                              unsigned threads, McWorkspace& ws,
                                              ThreadPool* pool) noexcept;

}  // namespace fxopt
