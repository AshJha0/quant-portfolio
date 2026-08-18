#include "eqvar/backtest.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "eqvar/stats.hpp"

namespace eqvar {

namespace {

// 0 * ln 0 = 0 convention shared by the Kupiec and Christoffersen likelihoods.
double xlogy(double a, double b) { return (a > 0.0 && b > 0.0) ? a * std::log(b) : 0.0; }

}  // namespace

std::vector<std::uint8_t> exceptions_from_pnl(std::span<const double> pnl,
                                              std::span<const double> var) {
    if (pnl.empty()) throw std::invalid_argument("exceptions_from_pnl: empty pnl");
    if (var.size() != 1 && var.size() != pnl.size()) {
        throw std::invalid_argument(
            "exceptions_from_pnl: var must be scalar (size 1) or one entry per day");
    }
    for (double v : var) {
        if (v < 0.0) {
            throw std::invalid_argument("exceptions_from_pnl: VaR must be a positive loss");
        }
    }
    std::vector<std::uint8_t> ex(pnl.size());
    for (std::size_t t = 0; t < pnl.size(); ++t) {
        const double v = (var.size() == 1) ? var[0] : var[t];
        ex[t] = pnl[t] < -v ? 1 : 0;
    }
    return ex;
}

KupiecResult kupiec_pof(int n_obs, int n_exceptions, double alpha) {
    if (n_obs < 1) {
        throw std::invalid_argument("kupiec_pof: n_obs must be >= 1, got " +
                                    std::to_string(n_obs));
    }
    if (n_exceptions < 0 || n_exceptions > n_obs) {
        throw std::invalid_argument("kupiec_pof: n_exceptions must be in [0, n_obs]");
    }
    if (!(alpha > 0.0 && alpha < 1.0)) {
        throw std::invalid_argument("kupiec_pof: alpha must be in (0, 1), got " +
                                    std::to_string(alpha));
    }
    const double t = static_cast<double>(n_obs);
    const double x = static_cast<double>(n_exceptions);
    const double pihat = x / t;

    // Binomial log-likelihood at a given exception probability p.
    const auto ll = [t, x](double p) {
        return xlogy(t - x, 1.0 - p) + xlogy(x, p);
    };
    // Degenerate pihat in {0, 1}: the alternative's log-likelihood is 0.
    const double ll_alt = (pihat == 0.0 || pihat == 1.0) ? 0.0 : ll(pihat);
    const double lr = std::max(-2.0 * (ll(alpha) - ll_alt), 0.0);

    KupiecResult res;
    res.lr = lr;
    res.pvalue = chi2_sf(lr, 1.0);
    res.expected = alpha * t;
    res.rate = pihat;
    return res;
}

ChristoffersenResult christoffersen_independence(std::span<const std::uint8_t> exceptions) {
    if (exceptions.size() < 2) {
        throw std::invalid_argument("christoffersen_independence: need at least 2 observations");
    }
    ChristoffersenResult res;
    for (std::size_t t = 1; t < exceptions.size(); ++t) {
        const bool prev = exceptions[t - 1] != 0;
        const bool curr = exceptions[t] != 0;
        if (!prev && !curr) res.n00 += 1.0;
        if (!prev && curr) res.n01 += 1.0;
        if (prev && !curr) res.n10 += 1.0;
        if (prev && curr) res.n11 += 1.0;
    }
    const double n0 = res.n00 + res.n01;
    const double n1 = res.n10 + res.n11;
    res.pi01 = (n0 > 0.0) ? res.n01 / n0 : 0.0;
    res.pi11 = (n1 > 0.0) ? res.n11 / n1 : 0.0;
    const double pi = (res.n01 + res.n11) / (n0 + n1);
    const double ll_markov = xlogy(res.n00, 1.0 - res.pi01) + xlogy(res.n01, res.pi01) +
                             xlogy(res.n10, 1.0 - res.pi11) + xlogy(res.n11, res.pi11);
    const double ll_iid = xlogy(res.n00 + res.n10, 1.0 - pi) + xlogy(res.n01 + res.n11, pi);
    res.lr = std::max(-2.0 * (ll_iid - ll_markov), 0.0);
    res.pvalue = chi2_sf(res.lr, 1.0);
    return res;
}

ConditionalCoverageResult christoffersen_cc(std::span<const std::uint8_t> exceptions,
                                            double alpha) {
    int count = 0;
    for (std::uint8_t e : exceptions) count += (e != 0) ? 1 : 0;
    const KupiecResult uc = kupiec_pof(static_cast<int>(exceptions.size()), count, alpha);
    const ChristoffersenResult ind = christoffersen_independence(exceptions);
    ConditionalCoverageResult res;
    res.lr_uc = uc.lr;
    res.lr_ind = ind.lr;
    res.lr = uc.lr + ind.lr;
    res.pvalue = chi2_sf(res.lr, 2.0);
    return res;
}

BaselResult basel_traffic_light(int n_exceptions, int n_obs) {
    if (n_exceptions < 0) {
        throw std::invalid_argument("basel_traffic_light: n_exceptions must be >= 0");
    }
    if (n_obs < 1) throw std::invalid_argument("basel_traffic_light: n_obs must be >= 1");
    BaselResult res;
    double addon;
    if (n_exceptions <= 4) {
        res.zone = BaselZone::Green;
        addon = 0.0;
    } else if (n_exceptions <= 9) {
        res.zone = BaselZone::Yellow;
        // Regulatory add-on schedule for 5..9 exceptions.
        static constexpr double kYellowAddon[] = {0.40, 0.50, 0.65, 0.75, 0.85};
        addon = kYellowAddon[n_exceptions - 5];
    } else {
        res.zone = BaselZone::Red;
        addon = 1.0;
    }
    res.multiplier = 3.0 + addon;
    res.cumulative_prob = binomial_cdf(std::min(n_exceptions, n_obs), n_obs, 0.01);
    return res;
}

std::string to_string(BaselZone zone) {
    switch (zone) {
        case BaselZone::Green: return "green";
        case BaselZone::Yellow: return "yellow";
        case BaselZone::Red: return "red";
    }
    return "unknown";
}

}  // namespace eqvar
