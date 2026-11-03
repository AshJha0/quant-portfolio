// Equity VaR engine (eqvar::). alpha is the tail probability, in (0, 0.5).
#include "common.hpp"

#include <vector>

#include "eqvar/backtest.hpp"
#include "eqvar/error.hpp"
#include "eqvar/kernels.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/monte_carlo.hpp"
#include "eqvar/parametric.hpp"

namespace qpb {
namespace {

template <class T>
T unwrap(eqvar::Result<T>&& r) {
    if (!r) raise_engine([&] { eqvar::throw_error(r.error()); });
    return *std::move(r);
}

eqvar::Dist parse_dist(const std::string& s) {
    if (s == "normal") return eqvar::Dist::Normal;
    if (s == "t") return eqvar::Dist::StudentT;
    value_error("dist must be 'normal' or 't', got '" + s + "'");
}

eqvar::Matrix to_matrix(const InMat& a) {
    const std::size_t r = a.shape(0), c = a.shape(1);
    return eqvar::Matrix(r, c, std::vector<double>(a.data(), a.data() + r * c));
}

}  // namespace

void bind_eqvar(nb::module_& m) {
    using namespace nb::literals;

    m.def(
        "eq_historical_var",
        [](const InVec& pnl, double alpha) {
            std::vector<double> ws(pnl.shape(0));
            eqvar::Result<double> r;
            {
                nb::gil_scoped_release nogil;
                r = eqvar::try_historical_var(span_of(pnl), alpha, ws);
            }
            return unwrap(std::move(r));
        },
        "pnl"_a, nb::kw_only(), "alpha"_a = 0.01,
        "Historical-simulation VaR, order statistic L(m) (eqvar::try_historical_var).");

    m.def(
        "eq_expected_shortfall",
        [](const InVec& pnl, double alpha) {
            std::vector<double> ws(pnl.shape(0));
            eqvar::Result<double> r;
            {
                nb::gil_scoped_release nogil;
                r = eqvar::try_expected_shortfall(span_of(pnl), alpha, ws);
            }
            return unwrap(std::move(r));
        },
        "pnl"_a, nb::kw_only(), "alpha"_a = 0.01,
        "Empirical expected shortfall with the fractional atom "
        "(eqvar::try_expected_shortfall).");

    m.def(
        "eq_parametric_var",
        [](const InVec& exposures, const InMat& cov, double alpha, const std::string& dist,
           double df, double mean, int horizon_days) {
            const eqvar::Matrix c = to_matrix(cov);
            return unwrap(eqvar::try_parametric_var(span_of(exposures), c, alpha,
                                                    parse_dist(dist), df, mean, horizon_days));
        },
        "exposures"_a, "cov"_a, nb::kw_only(), "alpha"_a = 0.01, "dist"_a = "normal",
        "df"_a = 6.0, "mean"_a = 0.0, "horizon_days"_a = 1,
        "Delta-normal or Student-t parametric VaR (eqvar::try_parametric_var).");

    m.def(
        "eq_monte_carlo_var",
        [](const InVec& exposures, const InMat& cov, double alpha, std::size_t n_paths,
           const std::string& dist, double df, std::uint64_t seed) {
            const eqvar::Matrix c = to_matrix(cov);
            const eqvar::Dist d = parse_dist(dist);
            return guarded([&] {
                nb::gil_scoped_release nogil;
                return eqvar::monte_carlo_var(span_of(exposures), c, alpha, n_paths, d, df,
                                              seed)
                    .var;
            });
        },
        "exposures"_a, "cov"_a, nb::kw_only(), "alpha"_a = 0.01, "n_paths"_a = 100000,
        "dist"_a = "normal", "df"_a = 6.0, "seed"_a = 0,
        "Full Monte Carlo VaR on a linear book (eqvar::monte_carlo_var). Returns the VaR. "
        "Uses mt19937_64, so it agrees with the Python engine statistically, not by value.");

    m.def(
        "eq_kupiec_pof",
        [](int n_obs, int n_exceptions, double alpha) {
            const auto k = guarded([&] {
                return eqvar::kupiec_pof(
                    eqvar::KupiecInput{.n_obs = n_obs, .n_exceptions = n_exceptions,
                                       .alpha = alpha});
            });
            nb::dict d;
            d["lr"] = k.lr;
            d["pvalue"] = k.pvalue;
            d["expected"] = k.expected;
            d["rate"] = k.rate;
            return d;
        },
        nb::kw_only(), "n_obs"_a, "n_exceptions"_a, "alpha"_a = 0.01,
        "Kupiec POF test (eqvar::kupiec_pof). alpha is the tail probability.");

    m.def(
        "eq_basel_traffic_light",
        [](int n_obs, int n_exceptions, double alpha) {
            const auto b = guarded([&] {
                return eqvar::basel_traffic_light(
                    eqvar::BaselInput{.n_obs = n_obs, .n_exceptions = n_exceptions,
                                      .alpha = alpha});
            });
            nb::dict d;
            d["zone"] = eqvar::to_string(b.zone);
            d["multiplier"] = b.multiplier;
            d["plus_factor"] = b.plus_factor;
            d["cumulative_prob"] = b.cumulative_prob;
            return d;
        },
        nb::kw_only(), "n_obs"_a, "n_exceptions"_a, "alpha"_a = 0.01,
        "Basel traffic light (eqvar::basel_traffic_light). alpha is the tail probability.");
}

}  // namespace qpb
