#include "fxopt/monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
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
                  double sigma, OptionType type, const McConfig& config) {
    return mc_price(S, K, T, r_d, r_f, sigma, type, config.n_paths, config.seed,
                    config.antithetic, config.control_variate);
}

MCResult mc_price(double S, double K, double T, double r_d, double r_f,
                  double sigma, OptionType type, std::int64_t n_paths,
                  Seed seed_value, bool antithetic, bool control_variate) {
    const std::uint64_t seed = seed_value.value;
    const double p_sign = phi(type);
    validate_inputs(S, K, T, r_d, r_f, sigma);
    if (T <= 0.0) {
        throw std::invalid_argument("mc_price requires T > 0");
    }
    // A standard error needs two independent samples (paths, or antithetic
    // pairs) left after fitting: three with the control variate, since two
    // points fit the regression line exactly and leave a zero residual.
    const std::int64_t units = antithetic ? (n_paths + 1) / 2 : n_paths;
    const std::int64_t needed = control_variate ? 3 : 2;
    if (units < needed) {
        throw std::invalid_argument(std::format(
            "n_paths = {} gives {} independent sample(s); at least {} are needed "
            "for a standard error", n_paths, units, needed));
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

    bool overflow = false;
    const auto fill = [&](std::size_t idx, double z) {
        const double s_t = S * std::exp(drift + vol * z);
        overflow = overflow || !std::isfinite(s_t);
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

    if (overflow) {
        throw std::invalid_argument("simulated spots overflow a double; drift or vol too large");
    }

    // Reduce to one value per independent draw first (the antithetic pair
    // average), then fit the control-variate beta on those same units: the
    // optimal beta for the pair-average estimator is
    // Cov(pair payoff, pair control) / Var(pair control).
    std::vector<double> samples;
    if (antithetic) {
        samples.resize(half);
        for (std::size_t i = 0; i < half; ++i) {
            samples[i] = 0.5 * (discounted[i] + discounted[i + half]);
            if (control_variate) control[i] = 0.5 * (control[i] + control[i + half]);
        }
        if (control_variate) control.resize(half);
    } else {
        samples = std::move(discounted);
    }
    if (control_variate) {
        const double control_mean_known = S * std::exp(-r_f * T);
        const double mp = mean_of(samples);
        const double mc = mean_of(control);
        double cov = 0.0;
        double var_c = 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            const double dc = control[i] - mc;
            cov += (samples[i] - mp) * dc;
            var_c += dc * dc;
        }
        const double beta = var_c > 0.0 ? cov / var_c : 0.0;
        for (std::size_t i = 0; i < samples.size(); ++i) {
            samples[i] -= beta * (control[i] - control_mean_known);
        }
    }
    const double price = mean_of(samples);
    double ss = 0.0;
    for (const double v : samples) {
        const double d = v - price;
        ss += d * d;
    }
    const double n = static_cast<double>(samples.size());
    // At least two independent samples are guaranteed by the check above.
    const double se = std::sqrt(ss / (n - 1.0)) / std::sqrt(n);

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
