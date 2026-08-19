#include "fxopt/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace fxopt {

namespace {

// Sample mean.
double mean_of(const std::vector<double>& x) {
    double s = 0.0;
    for (const double v : x) s += v;
    return s / static_cast<double>(x.size());
}

// Standard normal draw by inverse-CDF transform of a raw mt19937_64 word.
// std::normal_distribution's algorithm is implementation-defined, so it is
// only reproducible within one standard library; norm_ppf on an explicitly
// constructed uniform is bit-reproducible everywhere.  The top 53 bits map
// to u in (0, 1) strictly (offset by half an ULP so u is never 0 or 1).
double normal_draw(std::mt19937_64& rng) {
    const double u =
        (static_cast<double>(rng() >> 11) + 0.5) * 0x1.0p-53;
    return fxopt::norm_ppf(u);
}

}  // namespace

MCResult mc_price(double S, double K, double T, double r_d, double r_f,
                  double sigma, OptionType type, std::int64_t n_paths,
                  std::uint64_t seed, bool antithetic, bool control_variate) {
    const double p_sign = phi(type);
    validate_inputs(S, K, T, r_d, r_f, sigma);
    if (T <= 0.0) {
        throw std::invalid_argument("mc_price requires T > 0");
    }
    if (n_paths < 2) {
        throw std::invalid_argument("n_paths must be >= 2, got " +
                                    std::to_string(n_paths));
    }

    const double drift = (r_d - r_f - 0.5 * sigma * sigma) * T;
    const double vol = sigma * std::sqrt(T);
    const double df_d = std::exp(-r_d * T);

    // Terminal spots: draws z then mirrored -z when antithetic
    // (layout [z_0..z_{h-1}, -z_0..-z_{h-1}] so pair i = (i, i+half)).
    std::mt19937_64 rng(seed);
    std::size_t total;
    std::size_t half = 0;
    if (antithetic) {
        half = static_cast<std::size_t>((n_paths + 1) / 2);
        total = 2 * half;
    } else {
        total = static_cast<std::size_t>(n_paths);
    }
    std::vector<double> discounted(total);
    std::vector<double> control;
    if (control_variate) control.resize(total);

    const auto fill = [&](std::size_t idx, double z) {
        const double s_t = S * std::exp(drift + vol * z);
        discounted[idx] = df_d * std::max(p_sign * (s_t - K), 0.0);
        if (control_variate) control[idx] = df_d * s_t;
    };
    if (antithetic) {
        for (std::size_t i = 0; i < half; ++i) {
            const double z = normal_draw(rng);
            fill(i, z);
            fill(i + half, -z);
        }
    } else {
        for (std::size_t i = 0; i < total; ++i) {
            fill(i, normal_draw(rng));
        }
    }

    // Control-variate adjustment: subtract beta * (control - known mean),
    // beta = sample cov(payoff, control) / var(control).
    if (control_variate) {
        const double control_mean_known = S * std::exp(-r_f * T);
        const double mp = mean_of(discounted);
        const double mc = mean_of(control);
        double cov = 0.0;
        double var_c = 0.0;
        for (std::size_t i = 0; i < total; ++i) {
            const double dc = control[i] - mc;
            cov += (discounted[i] - mp) * dc;
            var_c += dc * dc;
        }
        const double beta = var_c > 0.0 ? cov / var_c : 0.0;
        for (std::size_t i = 0; i < total; ++i) {
            discounted[i] -= beta * (control[i] - control_mean_known);
        }
    }

    // SE from independent samples (pair averages when antithetic).
    std::vector<double> samples;
    if (antithetic) {
        samples.resize(half);
        for (std::size_t i = 0; i < half; ++i) {
            samples[i] = 0.5 * (discounted[i] + discounted[i + half]);
        }
    } else {
        samples = std::move(discounted);
    }
    const double price = mean_of(samples);
    double ss = 0.0;
    for (const double v : samples) {
        const double d = v - price;
        ss += d * d;
    }
    const double n = static_cast<double>(samples.size());
    // With fewer than two *independent* samples the sample variance is 0/0.
    // That happens only for the degenerate n_paths <= 2 antithetic case
    // (one mirrored pair); report SE = 0 rather than propagating NaN into
    // the price's confidence interval.  Documented in monte_carlo.hpp.
    const double se =
        samples.size() > 1 ? std::sqrt(ss / (n - 1.0)) / std::sqrt(n) : 0.0;

    std::string method = antithetic ? "antithetic+" : "";
    method += control_variate ? "control_variate" : "plain";
    return MCResult{price,
                    se,
                    price - 1.96 * se,
                    price + 1.96 * se,
                    static_cast<std::int64_t>(total),
                    method};
}

}  // namespace fxopt
