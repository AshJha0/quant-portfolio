// FX VaR engine (fxvar::). alpha is the tail probability, in (0, 0.5).
#include "common.hpp"

#include <nanobind/stl/pair.h>

#include <vector>

#include "fxvar/backtest.hpp"
#include "fxvar/error.hpp"
#include "fxvar/kernels.hpp"
#include "fxvar/matrix.hpp"
#include "fxvar/parametric.hpp"

namespace qpb {
namespace {

template <class T>
T unwrap(fxvar::Result<T>&& r) {
    if (!r) raise_engine([&] { fxvar::throw_error(r.error()); });
    return *std::move(r);
}

fxvar::TailDist parse_dist(const std::string& s) {
    if (s == "normal") return fxvar::TailDist::kNormal;
    if (s == "t") return fxvar::TailDist::kStudentT;
    value_error("dist must be 'normal' or 't', got '" + s + "'");
}

fxvar::Matrix to_matrix(const InMat& a) {
    const std::size_t r = a.shape(0), c = a.shape(1);
    fxvar::Matrix out(r, c);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) out(i, j) = a.data()[i * c + j];
    return out;
}

const char* zone_name(fxvar::Zone z) {
    switch (z) {
        case fxvar::Zone::kGreen: return "green";
        case fxvar::Zone::kYellow: return "yellow";
        case fxvar::Zone::kRed: return "red";
    }
    return "red";
}

}  // namespace

void bind_fxvar(nb::module_& m) {
    using namespace nb::literals;

    m.def(
        "fx_empirical_var_es",
        [](const InVec& pnl, double alpha) {
            std::vector<double> ws(pnl.shape(0));
            fxvar::Result<std::pair<double, double>> r;
            {
                nb::gil_scoped_release nogil;
                r = fxvar::try_empirical_var_es(span_of(pnl), alpha, ws);
            }
            return unwrap(std::move(r));
        },
        "pnl"_a, nb::kw_only(), "alpha"_a = 0.01,
        "Uniform-weight empirical (VaR, ES) (fxvar::try_empirical_var_es).");

    m.def(
        "fx_normal_var",
        [](double sigma, double alpha, double mean) {
            return unwrap(fxvar::try_normal_var(sigma, alpha, mean));
        },
        "sigma"_a, nb::kw_only(), "alpha"_a = 0.01, "mean"_a = 0.0,
        "Closed-form Normal VaR (fxvar::try_normal_var).");
    m.def(
        "fx_normal_es",
        [](double sigma, double alpha, double mean) {
            return unwrap(fxvar::try_normal_es(sigma, alpha, mean));
        },
        "sigma"_a, nb::kw_only(), "alpha"_a = 0.01, "mean"_a = 0.0,
        "Closed-form Normal ES (fxvar::try_normal_es).");
    m.def(
        "fx_t_var",
        [](double sigma, double alpha, double df, double mean) {
            return unwrap(fxvar::try_t_var(sigma, alpha, df, mean));
        },
        "sigma"_a, nb::kw_only(), "alpha"_a = 0.01, "df"_a = 6.0, "mean"_a = 0.0,
        "Standardised Student-t VaR (fxvar::try_t_var).");
    m.def(
        "fx_t_es",
        [](double sigma, double alpha, double df, double mean) {
            return unwrap(fxvar::try_t_es(sigma, alpha, df, mean));
        },
        "sigma"_a, nb::kw_only(), "alpha"_a = 0.01, "df"_a = 6.0, "mean"_a = 0.0,
        "Standardised Student-t ES (fxvar::try_t_es).");

    m.def(
        "fx_var_covar",
        [](const InVec& exposures, const InMat& cov, double alpha, double horizon_days,
           const std::string& dist, double df, double mean) {
            const fxvar::Matrix c = to_matrix(cov);
            const auto v = unwrap(fxvar::try_var_covar(span_of(exposures), c, alpha,
                                                       horizon_days, parse_dist(dist), df, mean));
            return std::make_pair(v.var, v.es);
        },
        "exposures"_a, "cov"_a, nb::kw_only(), "alpha"_a = 0.01, "horizon_days"_a = 1.0,
        "dist"_a = "normal", "df"_a = 6.0, "mean"_a = 0.0,
        "Variance-covariance (VaR, ES) (fxvar::try_var_covar).");

    m.def(
        "fx_kupiec_pof",
        [](int n_obs, int n_exceptions, double alpha) {
            const auto k = guarded([&] {
                return fxvar::kupiec_pof(fxvar::KupiecInput{
                    .n_obs = n_obs, .n_exceptions = n_exceptions, .alpha = alpha});
            });
            return std::make_pair(k.lr, k.p);
        },
        nb::kw_only(), "n_obs"_a, "n_exceptions"_a, "alpha"_a = 0.01,
        "Kupiec POF test (fxvar::kupiec_pof): (LR_uc, p_value). alpha is the tail "
        "probability.");

    m.def(
        "fx_basel_traffic_light",
        [](int n_obs, int n_exceptions, double alpha) {
            const auto b = guarded([&] {
                return fxvar::basel_traffic_light(fxvar::BaselInput{
                    .n_obs = n_obs, .n_exceptions = n_exceptions, .alpha = alpha});
            });
            nb::dict d;
            d["zone"] = zone_name(b.zone);
            d["n_exceptions"] = b.n_exceptions;
            d["n_obs"] = b.n_obs;
            d["cumulative_prob"] = b.cumulative_prob;
            d["multiplier"] = b.multiplier;
            d["plus_factor"] = b.plus_factor;
            return d;
        },
        nb::kw_only(), "n_obs"_a, "n_exceptions"_a, "alpha"_a = 0.01,
        "Basel traffic light (fxvar::basel_traffic_light). alpha is the tail probability.");
}

}  // namespace qpb
