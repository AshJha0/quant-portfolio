#include "eqvar/backtest.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "eqvar/returns.hpp"
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
        // NaN fails every comparison, so an unchecked NaN VaR would silently
        // record *zero* exceptions and make a broken model look perfect.
        if (!std::isfinite(v)) {
            throw std::invalid_argument("exceptions_from_pnl: VaR must be finite");
        }
        if (v < 0.0) {
            throw std::invalid_argument("exceptions_from_pnl: VaR must be a positive loss");
        }
    }
    for (double v : pnl) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument(
                "exceptions_from_pnl: pnl contains NaN or infinite values");
        }
    }
    std::vector<std::uint8_t> ex(pnl.size());
    for (std::size_t t = 0; t < pnl.size(); ++t) {
        const double v = (var.size() == 1) ? var[0] : var[t];
        ex[t] = pnl[t] < -v ? 1 : 0;
    }
    return ex;
}

namespace {

void validate_counts(const KupiecInput& in, const char* who) {
    validate_alpha(in.alpha);
    if (in.n_obs < 1) {
        throw std::invalid_argument(std::string(who) + ": n_obs must be >= 1, got " +
                                    std::to_string(in.n_obs));
    }
    if (in.n_exceptions < 0 || in.n_exceptions > in.n_obs) {
        throw std::invalid_argument(std::string(who) + ": n_exceptions must be in [0, n_obs], got " +
                                    std::to_string(in.n_exceptions));
    }
}

}  // namespace

KupiecResult kupiec_pof(const KupiecInput& in) {
    validate_counts(in, "kupiec_pof");
    const double alpha = in.alpha;
    const double t = static_cast<double>(in.n_obs);
    const double x = static_cast<double>(in.n_exceptions);
    const double pihat = x / t;
    const auto ll = [t, x](double p) { return xlogy(t - x, 1.0 - p) + xlogy(x, p); };
    // Degenerate pihat in {0, 1}: the alternative's log-likelihood is 0.
    const double ll_alt = (in.n_exceptions == 0 || in.n_exceptions == in.n_obs) ? 0.0 : ll(pihat);
    const double lr = std::max(-2.0 * (ll(alpha) - ll_alt), 0.0);
    KupiecResult res;
    res.lr = lr;
    res.pvalue = chi2_sf(lr, 1.0);
    res.expected = alpha * t;
    res.rate = pihat;
    return res;
}

KupiecResult kupiec_pof(int n_obs, int n_exceptions, double alpha) {
    return kupiec_pof(KupiecInput{.n_obs = n_obs, .n_exceptions = n_exceptions, .alpha = alpha});
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
    const KupiecResult uc = kupiec_pof(KupiecInput{
        .n_obs = static_cast<int>(exceptions.size()), .n_exceptions = count, .alpha = alpha});
    const ChristoffersenResult ind = christoffersen_independence(exceptions);
    ConditionalCoverageResult res;
    res.lr_uc = uc.lr;
    res.lr_ind = ind.lr;
    res.lr = uc.lr + ind.lr;
    res.pvalue = chi2_sf(res.lr, 2.0);
    return res;
}

BaselResult basel_traffic_light(const BaselInput& in) {
    validate_counts(in, "basel_traffic_light");
    BaselResult res;
    res.cumulative_prob = binomial_cdf(in.n_exceptions, in.n_obs, in.alpha);
    const double f = res.cumulative_prob;
    if (f < 0.95) {
        res.zone = BaselZone::Green;
        res.plus_factor = 0.0;
    } else if (f < 0.9999) {
        res.zone = BaselZone::Yellow;
        static constexpr double kPlus[] = {0.40, 0.50, 0.65, 0.75, 0.85};
        int x_eq = 9;
        for (int xp = 5; xp <= 9; ++xp) {
            if (binomial_cdf(xp, 250, 0.01) >= f - 1e-12) {
                x_eq = xp;
                break;
            }
        }
        res.plus_factor = kPlus[x_eq - 5];
    } else {
        res.zone = BaselZone::Red;
        res.plus_factor = 1.0;
    }
    res.multiplier = 3.0 + res.plus_factor;
    return res;
}

BaselResult basel_traffic_light(int n_exceptions, int n_obs) {
    return basel_traffic_light(
        BaselInput{.n_obs = n_obs, .n_exceptions = n_exceptions, .alpha = 0.01});
}

double acerbi_szekely_z2(std::span<const double> pnl, std::span<const double> var,
                         std::span<const double> es, double alpha) {
    validate_alpha(alpha);
    if (pnl.empty()) throw std::invalid_argument("acerbi_szekely_z2: empty pnl");
    const std::size_t n = pnl.size();
    if ((var.size() != 1 && var.size() != n) || (es.size() != 1 && es.size() != n)) {
        throw std::invalid_argument(
            "acerbi_szekely_z2: var and es must have size 1 or one entry per day");
    }
    for (double v : pnl) {
        if (!std::isfinite(v)) throw std::invalid_argument("acerbi_szekely_z2: pnl must be finite");
    }
    for (double v : var) {
        if (!std::isfinite(v) || v < 0.0) {
            throw std::invalid_argument("acerbi_szekely_z2: VaR must be finite and >= 0");
        }
    }
    for (double v : es) {
        if (!std::isfinite(v) || !(v > 0.0)) {
            throw std::invalid_argument("acerbi_szekely_z2: ES must be finite and > 0");
        }
    }
    const double t = static_cast<double>(n);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double v = var.size() == 1 ? var[0] : var[i];
        const double e = es.size() == 1 ? es[0] : es[i];
        if (pnl[i] < -v) sum += pnl[i] / (t * alpha * e);
    }
    return sum + 1.0;
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
