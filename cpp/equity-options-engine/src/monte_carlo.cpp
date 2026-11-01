#include "eqopt/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>

#include "eqopt/black_scholes.hpp"

namespace eqopt {

namespace {

constexpr double kZ95 = 1.959963984540054;  // two-sided 95% normal quantile

/// Acklam's rational approximation to the inverse normal CDF (~1.15e-9
/// relative error), used as the initial guess before Halley refinement.
double inv_norm_cdf_acklam(double p) noexcept {
    // Coefficients from P. J. Acklam (2003).
    constexpr double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                             -2.759285104469687e+02, 1.383577518672690e+02,
                             -3.066479806614716e+01, 2.506628277459239e+00};
    constexpr double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                             -1.556989798598866e+02, 6.680131188771972e+01,
                             -1.328068155288572e+01};
    constexpr double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                             -2.400758277161838e+00, -2.549732539343734e+00,
                             4.374664141464968e+00,  2.938163982698783e+00};
    constexpr double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                             2.445134137142996e+00, 3.754408661907416e+00};
    constexpr double p_low = 0.02425;

    if (p < p_low) {
        const double t = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t +
                c[5]) /
               ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0);
    }
    if (p <= 1.0 - p_low) {
        const double u = p - 0.5;
        const double t = u * u;
        return (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t +
                a[5]) *
               u /
               (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t +
                1.0);
    }
    const double t = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t +
             c[5]) /
           ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0);
}

/// splitmix64 mix step: derives statistically independent per-thread seeds
/// from (base seed, chunk index) deterministically.
std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

}  // namespace

double normal_from_u64(std::uint64_t u64) noexcept {
    // 53-bit uniform in (0, 1): (k + 0.5) * 2^-53 never hits 0 or 1.
    const double u =
        (static_cast<double>(u64 >> 11) + 0.5) * 1.1102230246251565e-16;
    double x = inv_norm_cdf_acklam(u);
    // One Halley step against the erfc-based CDF: pushes the ~1e-9 error of
    // the rational approximation down to ~1e-15.
    const double e = norm_cdf(x) - u;
    const double f = norm_pdf(x);
    x -= e / (f + 0.5 * x * e);
    return x;
}

namespace {

struct McTask {
    double S, K, disc, sign, drift, vol_sqrt_t;
    std::uint64_t seed;
    std::size_t base, half;
    unsigned n_chunks;
    double* payoff;
    double* control;
};

/// One deterministic chunk of draws. Antithetic and control-variate choices
/// are template parameters, so the inner loop has no per-path branch.
/// `x > 0 ? x : 0` is std::fmax(x, 0.0) for every x (NaN gives 0 in both;
/// -0.0 gives +0.0, which sums identically) and compiles to one maxsd.
template <bool Antithetic, bool Control>
void simulate_chunk(const McTask& k, unsigned c) noexcept {
    const std::size_t lo = k.base * c / k.n_chunks;
    const std::size_t hi = k.base * (c + 1) / k.n_chunks;
    std::mt19937_64 rng(splitmix64(k.seed + 0x51ed2701ULL * c));
    for (std::size_t i = lo; i < hi; ++i) {
        const double z = normal_from_u64(rng());
        const double s_up = k.S * std::exp(k.drift + k.vol_sqrt_t * z);
        const double x_up = k.sign * (s_up - k.K);
        k.payoff[i] = k.disc * (x_up > 0.0 ? x_up : 0.0);
        if constexpr (Control) k.control[i] = k.disc * s_up;
        if constexpr (Antithetic) {
            const double s_dn = k.S * std::exp(k.drift - k.vol_sqrt_t * z);
            const double x_dn = k.sign * (s_dn - k.K);
            k.payoff[k.half + i] = k.disc * (x_dn > 0.0 ? x_dn : 0.0);
            if constexpr (Control) k.control[k.half + i] = k.disc * s_dn;
        }
    }
}

template <bool Antithetic, bool Control>
void chunk_entry(void* ctx, unsigned c) noexcept {
    simulate_chunk<Antithetic, Control>(*static_cast<const McTask*>(ctx), c);
}

}  // namespace

Result<MCResult> try_mc_price(double S, double K, double T, double r, double sigma,
                              double q, OptionType type, std::size_t n_paths,
                              bool antithetic, bool control_variate, std::uint64_t seed,
                              unsigned threads, McWorkspace& ws, ThreadPool* pool) noexcept {
    if (auto e = try_validate_inputs(S, K, T, sigma); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    if (auto e = try_validate_rates(r, q); !e) [[unlikely]] return std::unexpected(e.error());
    // A standard error needs two independent samples (paths, or antithetic
    // pairs) left after fitting: three with the control variate, since two
    // points fit the regression line exactly and leave a zero residual.
    const std::size_t units = antithetic ? (n_paths + 1) / 2 : n_paths;
    const std::size_t needed = control_variate ? 3 : 2;
    if (units < needed) [[unlikely]] return std::unexpected(Error{Errc::TooFewPaths});
    if (threads == 0) [[unlikely]] return std::unexpected(Error{Errc::BadThreads});
    if (T == 0.0 || sigma == 0.0) {
        const auto exact = try_bs_price(S, K, T, r, sigma, q, type);
        if (!exact) return std::unexpected(exact.error());
        return MCResult{*exact, 0.0, *exact, *exact, n_paths};
    }

    // `base` independent draws; with antithetic each is mirrored to -Z.
    const std::size_t half = antithetic ? (n_paths + 1) / 2 : 0;
    const std::size_t base = antithetic ? half : n_paths;
    const std::size_t n_eff = antithetic ? 2 * half : n_paths;
    if (ws.payoff_.size() < n_eff || (control_variate && ws.control_.size() < n_eff))
        [[unlikely]] {
        return std::unexpected(Error{Errc::WorkspaceTooSmall, "try_mc_price",
                                     static_cast<double>(ws.payoff_.size()),
                                     static_cast<double>(n_eff)});
    }
    double* payoff = ws.payoff_.data();
    double* control = control_variate ? ws.control_.data() : nullptr;

    // Deterministic chunking: chunk c covers [c*base/W, (c+1)*base/W) with
    // its own RNG stream seeded from splitmix64(seed, c). For a fixed
    // (seed, threads) the draws, and therefore every output bit, are the
    // same run to run and whatever pool executes the chunks.
    McTask task{S,
                K,
                std::exp(-r * T),
                type == OptionType::Call ? 1.0 : -1.0,
                (r - q - 0.5 * sigma * sigma) * T,
                sigma * std::sqrt(T),
                seed,
                base,
                half,
                static_cast<unsigned>(std::min<std::size_t>(threads, base)),
                payoff,
                control};
    ThreadPool::TaskFn fn = antithetic
                                ? (control_variate ? &chunk_entry<true, true>
                                                   : &chunk_entry<true, false>)
                                : (control_variate ? &chunk_entry<false, true>
                                                   : &chunk_entry<false, false>);
    if (pool != nullptr && task.n_chunks > 1) {
        pool->run(task.n_chunks, fn, &task);
    } else {
        for (unsigned c = 0; c < task.n_chunks; ++c) fn(&task, c);
    }

    // Reduce to one value per independent draw first (the antithetic pair
    // average), then fit the control-variate beta on those same units: the
    // optimal beta for the pair-average estimator is
    // Cov(pair payoff, pair control) / Var(pair control).
    std::size_t n_samples = n_eff;
    if (antithetic) {
        for (std::size_t i = 0; i < half; ++i) {
            payoff[i] = 0.5 * (payoff[i] + payoff[half + i]);
        }
        if (control_variate) {
            for (std::size_t i = 0; i < half; ++i) {
                control[i] = 0.5 * (control[i] + control[half + i]);
            }
        }
        n_samples = half;
    }
    if (control_variate) {
        double mean_p = 0.0;
        double mean_c = 0.0;
        for (std::size_t i = 0; i < n_samples; ++i) {
            mean_p += payoff[i];
            mean_c += control[i];
        }
        mean_p /= static_cast<double>(n_samples);
        mean_c /= static_cast<double>(n_samples);
        double cov_pc = 0.0;
        double var_c = 0.0;
        for (std::size_t i = 0; i < n_samples; ++i) {
            const double dc = control[i] - mean_c;
            cov_pc += (payoff[i] - mean_p) * dc;
            var_c += dc * dc;
        }
        const double beta = var_c > 0.0 ? cov_pc / var_c : 0.0;
        const double control_mean = S * std::exp(-q * T);  // martingale mean
        for (std::size_t i = 0; i < n_samples; ++i) {
            payoff[i] -= beta * (control[i] - control_mean);
        }
    }
    const double* samples = payoff;

    double mean = 0.0;
    for (std::size_t i = 0; i < n_samples; ++i) mean += samples[i];
    mean /= static_cast<double>(n_samples);
    double var = 0.0;
    for (std::size_t i = 0; i < n_samples; ++i) {
        const double d = samples[i] - mean;
        var += d * d;
    }
    const double se = std::sqrt(var / static_cast<double>(n_samples - 1) /
                                static_cast<double>(n_samples));
    if (!std::isfinite(mean) || !std::isfinite(se)) [[unlikely]] {
        return std::unexpected(Error{Errc::McNotFinite});
    }
    return MCResult{mean, se, mean - kZ95 * se, mean + kZ95 * se, n_eff};
}

MCResult mc_price(double S, double K, double T, double r, double sigma,
                  double q, OptionType type, std::size_t n_paths,
                  bool antithetic, bool control_variate, std::uint64_t seed,
                  unsigned threads) {
    McWorkspace ws;
    // Validation errors take precedence over sizing, as before.
    if (auto e = try_validate_inputs(S, K, T, sigma); !e) throw_error(e.error());
    if (auto e = try_validate_rates(r, q); !e) throw_error(e.error());
    ws.reserve(n_paths, control_variate);
    ThreadPool* pool = threads > 1 ? &default_thread_pool() : nullptr;
    return value_or_throw(try_mc_price(S, K, T, r, sigma, q, type, n_paths, antithetic,
                                       control_variate, seed, threads, ws, pool));
}

}  // namespace eqopt
