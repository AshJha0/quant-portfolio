#include "fxopt/monte_carlo.hpp"

#include "qp/rng.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <random>

namespace fxopt {

namespace {

// Standard normal draw by inverse-CDF transform of a raw mt19937_64 word.
// std::normal_distribution's algorithm is implementation-defined, so it is
// only reproducible within one standard library; norm_ppf on an explicitly
// constructed uniform is bit-reproducible everywhere.  The top 53 bits map
// to u in (0, 1) strictly (offset by half an ULP so u is never 0 or 1).
inline double normal_draw(std::mt19937_64& rng) noexcept {
    const double u =
        (static_cast<double>(rng() >> 11) + 0.5) * 0x1.0p-53;
    return fxopt::norm_ppf_unchecked(u);
}

/// splitmix64 mix step: per-chunk seeds for threads > 1.
using qp::splitmix64;

struct McTask {
    double S, K, df_d, p_sign, drift, vol;
    std::uint64_t seed;
    std::size_t base, half;
    unsigned n_chunks;
    double* discounted;
    double* control;
    std::atomic<bool> overflow{false};
};

/// One deterministic chunk of draws. With one chunk the generator is
/// mt19937_64(seed), the legacy single stream; with several, chunk c owns
/// draws [c*base/W, (c+1)*base/W) and the stream splitmix64(seed + k*c).
/// Antithetic and control-variate choices are template parameters, so the
/// inner loop has no per-path branch. `x < 0 ? 0 : x` is std::max(x, 0.0)
/// exactly and compiles to one maxsd.
template <bool Antithetic, bool Control>
void simulate_chunk(McTask& k, unsigned c) noexcept {
    const std::size_t lo = k.base * c / k.n_chunks;
    const std::size_t hi = k.base * (c + 1) / k.n_chunks;
    std::mt19937_64 rng(k.n_chunks == 1 ? k.seed : splitmix64(k.seed + 0x51ed2701ULL * c));
    bool overflow = false;
    const auto fill = [&](std::size_t idx, double z) {
        const double s_t = k.S * std::exp(k.drift + k.vol * z);
        overflow = overflow || !std::isfinite(s_t);
        const double x = k.p_sign * (s_t - k.K);
        k.discounted[idx] = k.df_d * (x < 0.0 ? 0.0 : x);
        if constexpr (Control) k.control[idx] = k.df_d * s_t;
    };
    for (std::size_t i = lo; i < hi; ++i) {
        const double z = normal_draw(rng);
        fill(i, z);
        if constexpr (Antithetic) fill(i + k.half, -z);
    }
    if (overflow) k.overflow.store(true, std::memory_order_relaxed);
}

template <bool Antithetic, bool Control>
void chunk_entry(void* ctx, unsigned c) noexcept {
    simulate_chunk<Antithetic, Control>(*static_cast<McTask*>(ctx), c);
}

double mean_of(const double* x, std::size_t n) noexcept {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += x[i];
    return s / static_cast<double>(n);
}

}  // namespace

Result<McEstimate> try_mc_price(double S, double K, double T, double r_d, double r_f,
                                double sigma, OptionType type, std::int64_t n_paths,
                                Seed seed_value, bool antithetic, bool control_variate,
                                unsigned threads, McWorkspace& ws, ThreadPool* pool) noexcept {
    const std::uint64_t seed = seed_value.value;
    const double p_sign = phi(type);
    if (auto e = try_validate_inputs(S, K, T, r_d, r_f, sigma); !e) [[unlikely]] {
        return std::unexpected(e.error());
    }
    if (T <= 0.0) [[unlikely]] return std::unexpected(Error{Errc::McExpired});
    // A standard error needs two independent samples (paths, or antithetic
    // pairs) left after fitting: three with the control variate, since two
    // points fit the regression line exactly and leave a zero residual.
    const std::int64_t units = antithetic ? (n_paths + 1) / 2 : n_paths;
    const std::int64_t needed = control_variate ? 3 : 2;
    if (units < needed) [[unlikely]] {
        return std::unexpected(Error{Errc::TooFewPaths, "", static_cast<double>(n_paths),
                                     static_cast<double>(units),
                                     static_cast<double>(needed)});
    }
    if (threads == 0) [[unlikely]] return std::unexpected(Error{Errc::BadThreads});

    // Layout [z_0..z_{h-1}, -z_0..-z_{h-1}] when antithetic, so pair i is
    // (i, i + half).
    std::size_t total;
    std::size_t half = 0;
    if (antithetic) {
        half = static_cast<std::size_t>((n_paths + 1) / 2);
        total = 2 * half;
    } else {
        total = static_cast<std::size_t>(n_paths);
    }
    if (ws.discounted_.size() < total || (control_variate && ws.control_.size() < total))
        [[unlikely]] {
        return std::unexpected(Error{Errc::WorkspaceTooSmall, "try_mc_price",
                                     static_cast<double>(ws.discounted_.size()),
                                     static_cast<double>(total)});
    }
    double* discounted = ws.discounted_.data();
    double* control = control_variate ? ws.control_.data() : nullptr;
    const std::size_t base = antithetic ? half : total;

    McTask task{S,
                K,
                std::exp(-r_d * T),
                p_sign,
                (r_d - r_f - 0.5 * sigma * sigma) * T,
                sigma * std::sqrt(T),
                seed,
                base,
                half,
                static_cast<unsigned>(std::min<std::size_t>(threads, base)),
                discounted,
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
    if (task.overflow.load(std::memory_order_relaxed)) [[unlikely]] {
        return std::unexpected(Error{Errc::McOverflow});
    }

    // Reduce to one value per independent draw first (the antithetic pair
    // average), then fit the control-variate beta on those same units: the
    // optimal beta for the pair-average estimator is
    // Cov(pair payoff, pair control) / Var(pair control).
    double* samples = discounted;
    std::size_t n_samples = total;
    if (antithetic) {
        for (std::size_t i = 0; i < half; ++i) {
            samples[i] = 0.5 * (discounted[i] + discounted[i + half]);
        }
        if (control_variate) {
            for (std::size_t i = 0; i < half; ++i) {
                control[i] = 0.5 * (control[i] + control[i + half]);
            }
        }
        n_samples = half;
    }
    if (control_variate) {
        const double control_mean_known = S * std::exp(-r_f * T);
        const double mp = mean_of(samples, n_samples);
        const double mc = mean_of(control, n_samples);
        double cov = 0.0;
        double var_c = 0.0;
        for (std::size_t i = 0; i < n_samples; ++i) {
            const double dc = control[i] - mc;
            cov += (samples[i] - mp) * dc;
            var_c += dc * dc;
        }
        const double beta = var_c > 0.0 ? cov / var_c : 0.0;
        for (std::size_t i = 0; i < n_samples; ++i) {
            samples[i] -= beta * (control[i] - control_mean_known);
        }
    }
    const double price = mean_of(samples, n_samples);
    double ss = 0.0;
    for (std::size_t i = 0; i < n_samples; ++i) {
        const double d = samples[i] - price;
        ss += d * d;
    }
    const double n = static_cast<double>(n_samples);
    // At least two independent samples are guaranteed by the check above.
    const double se = std::sqrt(ss / (n - 1.0)) / std::sqrt(n);

    const char* method = antithetic ? (control_variate ? "antithetic+control_variate"
                                                       : "antithetic+plain")
                                    : (control_variate ? "control_variate" : "plain");
    return McEstimate{price,
                      se,
                      price - 1.96 * se,
                      price + 1.96 * se,
                      static_cast<std::int64_t>(total),
                      method};
}

MCResult mc_price(double S, double K, double T, double r_d, double r_f,
                  double sigma, OptionType type, const McConfig& config) {
    return mc_price(S, K, T, r_d, r_f, sigma, type, config.n_paths, config.seed,
                    config.antithetic, config.control_variate);
}

MCResult mc_price(double S, double K, double T, double r_d, double r_f,
                  double sigma, OptionType type, std::int64_t n_paths,
                  Seed seed_value, bool antithetic, bool control_variate) {
    // Validation errors take precedence over sizing, as before.
    value_or_throw(try_validate_inputs(S, K, T, r_d, r_f, sigma));
    McWorkspace ws;
    if (n_paths > 0) ws.reserve(n_paths, control_variate);
    const McEstimate e = value_or_throw(try_mc_price(S, K, T, r_d, r_f, sigma, type, n_paths,
                                                     seed_value, antithetic, control_variate,
                                                     1, ws, nullptr));
    return MCResult{e.price, e.std_error, e.ci_low, e.ci_high, e.n_paths, e.method};
}

}  // namespace fxopt
