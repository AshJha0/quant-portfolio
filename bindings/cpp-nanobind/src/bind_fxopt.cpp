// FX options engine (fxopt::): Garman-Kohlhagen.
#include "common.hpp"

#include <memory>
#include <vector>

#include "fxopt/batch.hpp"
#include "fxopt/binomial.hpp"
#include "fxopt/error.hpp"
#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/greeks.hpp"
#include "fxopt/implied_vol.hpp"
#include "fxopt/monte_carlo.hpp"
#include "fxopt/thread_pool.hpp"

namespace qpb {
namespace {

using fxopt::OptionType;

OptionType ot(bool call) { return call ? OptionType::Call : OptionType::Put; }
OptionType ot(const std::string& s) { return ot(parse_call(s)); }

template <class T>
T unwrap(fxopt::Result<T>&& r) {
    if (!r) raise_engine([&] { fxopt::throw_error(r.error()); });
    return *std::move(r);
}

[[noreturn]] void batch_fail(const fxopt::BatchError& e) {
    raise_engine([&] { fxopt::throw_error(e.error); }, "element " + std::to_string(e.index) + ": ");
}

constexpr const char* kGreekNames[10] = {"price",        "delta_spot",  "delta_forward",
                                         "gamma",        "vega",        "theta",
                                         "rho_domestic", "rho_foreign", "vanna",
                                         "volga"};

nb::dict greeks_dict(const fxopt::GreeksResult& g) {
    const double v[10] = {g.price, g.delta_spot,   g.delta_forward, g.gamma, g.vega,
                          g.theta, g.rho_domestic, g.rho_foreign,   g.vanna, g.volga};
    nb::dict d;
    for (int i = 0; i < 10; ++i) d[kGreekNames[i]] = v[i];
    return d;
}

struct Batch {
    std::size_t n;
    std::vector<OptionType> types;
    fxopt::GkBatchInputs in;
};

Batch make_batch(const InVec& S, const InVec& K, const InVec& T, const InVec& r_d,
                 const InVec& r_f, const InVec& sigma, const InBool& is_call) {
    const std::size_t n = S.shape(0);
    check_same_len(n, K.shape(0), "K");
    check_same_len(n, T.shape(0), "T");
    check_same_len(n, r_d.shape(0), "r_d");
    check_same_len(n, r_f.shape(0), "r_f");
    check_same_len(n, sigma.shape(0), "sigma");
    check_same_len(n, is_call.shape(0), "is_call");
    Batch b{n, std::vector<OptionType>(n), {}};
    for (std::size_t i = 0; i < n; ++i) b.types[i] = ot(is_call.data()[i]);
    b.in = {span_of(S),     span_of(K),     span_of(T), span_of(r_d), span_of(r_f),
            span_of(sigma), std::span<const OptionType>(b.types)};
    return b;
}

}  // namespace

void bind_fxopt(nb::module_& m) {
    using namespace nb::literals;

    m.def(
        "fx_gk_price",
        [](double S, double K, double T, double r_d, double r_f, double sigma,
           const std::string& option_type) {
            return unwrap(fxopt::try_gk_price(S, K, T, r_d, r_f, sigma, ot(option_type)));
        },
        "S"_a, "K"_a, "T"_a, "r_d"_a, "r_f"_a, "sigma"_a, "option_type"_a,
        "Garman-Kohlhagen price (fxopt::try_gk_price).");

    m.def(
        "fx_gk_greeks",
        [](double S, double K, double T, double r_d, double r_f, double sigma,
           const std::string& option_type) {
            return greeks_dict(
                unwrap(fxopt::try_analytic_greeks(S, K, T, r_d, r_f, sigma, ot(option_type))));
        },
        "S"_a, "K"_a, "T"_a, "r_d"_a, "r_f"_a, "sigma"_a, "option_type"_a,
        "Price and analytic Greeks (fxopt::try_analytic_greeks) as a dict.");

    m.def(
        "fx_gk_price_batch",
        [](const InVec& S, const InVec& K, const InVec& T, const InVec& r_d, const InVec& r_f,
           const InVec& sigma, const InBool& is_call) {
            Batch b = make_batch(S, K, T, r_d, r_f, sigma, is_call);
            Owned out = new_array(b.n);
            std::expected<void, fxopt::BatchError> res;
            {
                nb::gil_scoped_release nogil;
                res = fxopt::gk_price_batch(b.in, std::span<double>(out.data, b.n));
            }
            if (!res) batch_fail(res.error());
            return out.array;
        },
        "S"_a, "K"_a, "T"_a, "r_d"_a, "r_f"_a, "sigma"_a, "is_call"_a,
        "Prices for equal-length float64 arrays (fxopt::gk_price_batch). GIL released.");

    m.def(
        "fx_gk_greeks_batch",
        [](const InVec& S, const InVec& K, const InVec& T, const InVec& r_d, const InVec& r_f,
           const InVec& sigma, const InBool& is_call) {
            Batch b = make_batch(S, K, T, r_d, r_f, sigma, is_call);
            std::vector<Owned> o;
            o.reserve(10);
            for (int i = 0; i < 10; ++i) o.push_back(new_array(b.n));
            auto sp = [&](int i) { return std::span<double>(o[i].data, b.n); };
            fxopt::GkGreeksBatchOutputs outs{sp(0), sp(1), sp(2), sp(3), sp(4),
                                             sp(5), sp(6), sp(7), sp(8), sp(9)};
            std::expected<void, fxopt::BatchError> res;
            {
                nb::gil_scoped_release nogil;
                res = fxopt::gk_greeks_batch(b.in, outs);
            }
            if (!res) batch_fail(res.error());
            nb::dict d;
            for (int i = 0; i < 10; ++i) d[kGreekNames[i]] = o[i].array;
            return d;
        },
        "S"_a, "K"_a, "T"_a, "r_d"_a, "r_f"_a, "sigma"_a, "is_call"_a,
        "Greeks for equal-length arrays (fxopt::gk_greeks_batch), dict of arrays. "
        "GIL released.");

    m.def(
        "fx_implied_vol",
        [](double price, double S, double K, double T, double r_d, double r_f,
           const std::string& option_type) {
            return unwrap(fxopt::try_implied_vol(price, S, K, T, r_d, r_f, ot(option_type)));
        },
        "price"_a, "S"_a, "K"_a, "T"_a, "r_d"_a, "r_f"_a, "option_type"_a,
        "Implied volatility (fxopt::try_implied_vol).");

    m.def(
        "fx_implied_vol_batch",
        [](const InVec& price, const InVec& S, const InVec& K, const InVec& T, const InVec& r_d,
           const InVec& r_f, const InBool& is_call) {
            const std::size_t n = price.shape(0);
            check_same_len(n, S.shape(0), "S");
            check_same_len(n, K.shape(0), "K");
            check_same_len(n, T.shape(0), "T");
            check_same_len(n, r_d.shape(0), "r_d");
            check_same_len(n, r_f.shape(0), "r_f");
            check_same_len(n, is_call.shape(0), "is_call");
            Owned out = new_array(n);
            std::size_t bad = n;
            fxopt::Error err{};
            {
                nb::gil_scoped_release nogil;
                for (std::size_t i = 0; i < n; ++i) {
                    auto v = fxopt::try_implied_vol(price.data()[i], S.data()[i], K.data()[i],
                                                    T.data()[i], r_d.data()[i], r_f.data()[i],
                                                    ot(is_call.data()[i]));
                    if (!v) {
                        bad = i;
                        err = v.error();
                        break;
                    }
                    out.data[i] = *v;
                }
            }
            if (bad < n) batch_fail({bad, err});
            return out.array;
        },
        "price"_a, "S"_a, "K"_a, "T"_a, "r_d"_a, "r_f"_a, "is_call"_a,
        "Implied vols for equal-length arrays (loop over fxopt::try_implied_vol). "
        "GIL released.");

    m.def(
        "fx_binomial_price",
        [](double S, double K, double T, double r_d, double r_f, double sigma,
           const std::string& option_type, int steps, bool american) {
            std::vector<double> ws(fxopt::binomial_workspace_size(steps));
            return unwrap(fxopt::try_binomial_price(
                S, K, T, r_d, r_f, sigma, ot(option_type), steps,
                american ? fxopt::Exercise::American : fxopt::Exercise::European, ws));
        },
        "S"_a, "K"_a, "T"_a, "r_d"_a, "r_f"_a, "sigma"_a, "option_type"_a, nb::kw_only(),
        "steps"_a = 500, "american"_a = true,
        "CRR tree price (fxopt::try_binomial_price).");

    m.def(
        "fx_mc_price",
        [](double S, double K, double T, double r_d, double r_f, double sigma,
           const std::string& option_type, std::int64_t n_paths, std::uint64_t seed,
           bool antithetic, bool control_variate, unsigned threads) {
            const OptionType type = ot(option_type);
            if (n_paths < 0) value_error("n_paths must be >= 0");
            fxopt::McWorkspace ws;
            ws.reserve(n_paths, control_variate);
            fxopt::Result<fxopt::McEstimate> res;
            {
                nb::gil_scoped_release nogil;
                std::unique_ptr<fxopt::ThreadPool> pool;
                if (threads > 1) pool = std::make_unique<fxopt::ThreadPool>(threads - 1);
                res = fxopt::try_mc_price(S, K, T, r_d, r_f, sigma, type, n_paths,
                                          fxopt::Seed{seed}, antithetic, control_variate,
                                          threads, ws, pool.get());
            }
            const auto v = unwrap(std::move(res));
            nb::dict d;
            d["price"] = v.price;
            d["std_error"] = v.std_error;
            d["ci_low"] = v.ci_low;
            d["ci_high"] = v.ci_high;
            d["n_paths"] = v.n_paths;
            d["method"] = std::string(v.method);
            return d;
        },
        "S"_a, "K"_a, "T"_a, "r_d"_a, "r_f"_a, "sigma"_a, "option_type"_a, nb::kw_only(),
        "n_paths"_a = 100000, "seed"_a = 0, "antithetic"_a = true, "control_variate"_a = true,
        "threads"_a = 1,
        "Monte Carlo price (fxopt::try_mc_price). threads == 1 is the legacy mt19937_64 "
        "stream; threads > 1 is fixed for a given (seed, threads).");
}

}  // namespace qpb
