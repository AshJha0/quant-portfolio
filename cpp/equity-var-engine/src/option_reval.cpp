// eqvar/option_reval.cpp: see option_reval.hpp.
#include "eqvar/option_reval.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace eqvar {
namespace {

double ncdf(double x) { return 0.5 * std::erfc(-x / std::numbers::sqrt2); }
double npdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi); }

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("option_reval: ") + what);
}

void check_leg(const OptionLeg& l) {
    require(std::isfinite(l.quantity) && std::isfinite(l.rate) && std::isfinite(l.carry),
            "quantity, rate and carry must be finite");
    require(std::isfinite(l.spot) && l.spot > 0.0, "spot must be > 0");
    require(std::isfinite(l.strike) && l.strike > 0.0, "strike must be > 0");
    require(std::isfinite(l.tau) && l.tau > 0.0, "tau must be > 0");
    require(std::isfinite(l.vol) && l.vol > 0.0, "vol must be > 0");
}

void check_scenarios(std::span<const OptionLeg> legs, std::span<const double> x,
                     std::span<const double> dv, std::size_t nf) {
    require(!legs.empty(), "legs must not be empty");
    require(nf > 0 && x.size() % nf == 0, "scenario size must be a multiple of n_factors");
    require(dv.empty() || dv.size() == x.size(), "vol_shocks must match spot_log_returns");
    for (const auto& l : legs) {
        check_leg(l);
        require(l.factor < nf, "leg factor index out of range");
    }
    for (double v : x) require(std::isfinite(v), "spot_log_returns must be finite");
    for (double v : dv) require(std::isfinite(v), "vol_shocks must be finite");
}

double price_unchecked(double S, double K, double t, double r, double b, double v,
                       OptionKind kind) {
    const double s = kind == OptionKind::Call ? 1.0 : -1.0;
    if (t <= 0.0) return std::max(s * (S - K), 0.0);
    v = std::max(v, 1e-12);
    const double sq = v * std::sqrt(t);
    const double d1 = (std::log(S / K) + (r - b + 0.5 * v * v) * t) / sq;
    const double d2 = d1 - sq;
    return s * (S * std::exp(-b * t) * ncdf(s * d1) - K * std::exp(-r * t) * ncdf(s * d2));
}

}  // namespace

double bsm_price(double spot, double strike, double tau, double rate, double carry, double vol,
                 OptionKind kind) {
    require(std::isfinite(spot) && spot > 0.0 && std::isfinite(strike) && strike > 0.0,
            "spot and strike must be > 0");
    require(std::isfinite(tau) && std::isfinite(rate) && std::isfinite(carry), "non-finite input");
    require(std::isfinite(vol) && vol > 0.0, "vol must be > 0");
    return price_unchecked(spot, strike, tau, rate, carry, vol, kind);
}

BsmGreeks bsm_greeks(const OptionLeg& l) {
    check_leg(l);
    const double s = l.kind == OptionKind::Call ? 1.0 : -1.0;
    const double rt = std::sqrt(l.tau);
    const double sq = l.vol * rt;
    const double d1 =
        (std::log(l.spot / l.strike) + (l.rate - l.carry + 0.5 * l.vol * l.vol) * l.tau) / sq;
    const double d2 = d1 - sq;
    const double eb = std::exp(-l.carry * l.tau);
    const double er = std::exp(-l.rate * l.tau);
    const double pdf = npdf(d1);
    const double vega = l.spot * eb * pdf * rt;
    BsmGreeks g{};
    g.delta = s * eb * ncdf(s * d1);
    g.gamma = eb * pdf / (l.spot * sq);
    g.vega = vega;
    g.vanna = -eb * pdf * d2 / l.vol;
    g.volga = vega * d1 * d2 / l.vol;
    g.theta = -l.spot * eb * pdf * l.vol / (2.0 * rt) + s * l.carry * l.spot * eb * ncdf(s * d1) -
              s * l.rate * l.strike * er * ncdf(s * d2);
    return g;
}

std::vector<double> full_revaluation_pnl(std::span<const OptionLeg> legs, std::span<const double> x,
                                         std::span<const double> dv, std::size_t nf, double dt) {
    return full_revaluation_pnl(legs, x, dv, nf, dt, VolMode::StickyStrike, {});
}

std::vector<double> full_revaluation_pnl(std::span<const OptionLeg> legs, std::span<const double> x,
                                         std::span<const double> dv, std::size_t nf, double dt,
                                         VolMode mode, std::span<const double> skews) {
    check_scenarios(legs, x, dv, nf);
    require(std::isfinite(dt) && dt >= 0.0, "dt must be >= 0");
    require(skews.empty() || skews.size() == legs.size(), "skews must be empty or one per leg");
    for (double k : skews) require(std::isfinite(k), "skews must be finite");
    require(mode == VolMode::StickyStrike || mode == VolMode::StickyDelta ||
                mode == VolMode::HistoricalAtm,
            "unknown vol mode");
    const std::size_t n = x.size() / nf;
    std::vector<double> pnl(n, 0.0);
    for (std::size_t k = 0; k < legs.size(); ++k) {
        const OptionLeg& l = legs[k];
        const double skew = skews.empty() ? 0.0 : skews[k];
        const double p0 = price_unchecked(l.spot, l.strike, l.tau, l.rate, l.carry, l.vol, l.kind);
        const double t1 = std::max(l.tau - dt, 0.0);
        for (std::size_t r = 0; r < n; ++r) {
            const double xr = x[r * nf + l.factor];
            const double d = dv.empty() ? 0.0 : dv[r * nf + l.factor];
            const double s1 = l.spot * std::exp(xr);
            double v1 = l.vol + d;
            if (mode == VolMode::StickyDelta) v1 = l.vol - skew * xr + d;
            if (mode == VolMode::HistoricalAtm) v1 = l.vol * std::exp(d);
            pnl[r] +=
                l.quantity * (price_unchecked(s1, l.strike, t1, l.rate, l.carry, v1, l.kind) - p0);
        }
    }
    return pnl;
}

std::vector<double> delta_gamma_pnl(std::span<const OptionLeg> legs, std::span<const double> x,
                                    std::size_t nf) {
    check_scenarios(legs, x, {}, nf);
    const std::size_t n = x.size() / nf;
    std::vector<double> pnl(n, 0.0);
    for (const auto& l : legs) {
        const BsmGreeks g = bsm_greeks(l);
        for (std::size_t i = 0; i < n; ++i) {
            const double ds = l.spot * std::expm1(x[i * nf + l.factor]);
            pnl[i] += l.quantity * (g.delta * ds + 0.5 * g.gamma * ds * ds);
        }
    }
    return pnl;
}

std::vector<double> option_book_pnl(std::span<const OptionLeg> legs,
                                    std::span<const double> exposures, const Matrix& r, double dt) {
    const std::size_t n = r.rows();
    const std::size_t nf = r.cols();
    require(n > 0 && nf > 0, "factor_returns must be non-empty");
    require(exposures.empty() || exposures.size() == nf,
            "exposures must be empty or match columns");
    require(!legs.empty() || !exposures.empty(), "book has no positions");
    for (double w : exposures) require(std::isfinite(w), "exposures must be finite");
    const std::span<const double> x(r.data(), n * nf);
    std::vector<double> pnl =
        legs.empty() ? std::vector<double>(n, 0.0) : full_revaluation_pnl(legs, x, {}, nf, dt);
    if (legs.empty()) {
        for (double v : x) require(std::isfinite(v), "spot_log_returns must be finite");
    }
    if (!exposures.empty()) {
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < nf; ++j) s += exposures[j] * x[i * nf + j];
            pnl[i] += s;
        }
    }
    return pnl;
}

TailRisk historical_option_var_es(std::span<const OptionLeg> legs,
                                  std::span<const double> exposures, const Matrix& r, double alpha,
                                  double dt) {
    require(r.rows() >= kMinHistObs, "historical option VaR needs at least kMinHistObs rows");
    require(std::isfinite(alpha) && alpha > 0.0 && alpha < 0.5, "alpha must be in (0, 0.5)");
    const std::vector<double> pnl = option_book_pnl(legs, exposures, r, dt);
    return var_es_order_statistic(pnl, alpha);
}

MonteCarloResult monte_carlo_option_var(std::span<const OptionLeg> legs,
                                        std::span<const double> exposures, const Matrix& cov,
                                        double alpha, std::size_t n_paths, Dist dist, double df,
                                        std::uint64_t seed, std::size_t n_boot, Repair repair,
                                        double dt) {
    if (legs.empty()) {
        return monte_carlo_var(exposures, cov, alpha, n_paths, dist, df, seed, n_boot, repair);
    }
    require(cov.rows() == cov.cols() && cov.rows() > 0, "cov must be square and non-empty");
    require(exposures.empty() || exposures.size() == cov.rows(),
            "exposures must be empty or match cov");
    require(std::isfinite(alpha) && alpha > 0.0 && alpha < 0.5, "alpha must be in (0, 0.5)");
    const Matrix scen = simulate_factor_returns(cov, n_paths, dist, df, seed, repair);
    const std::vector<double> pnl = option_book_pnl(legs, exposures, scen, dt);
    MonteCarloResult res = mc_tail_metrics(pnl, alpha, n_boot, seed ^ 0x9E3779B97F4A7C15ULL);
    res.n_paths = n_paths;
    return res;
}

}  // namespace eqvar
