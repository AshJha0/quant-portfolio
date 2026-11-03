// Equity options engine (eqopt::): Black-Scholes-Merton with dividend yield.
#include "common.hpp"

#include <memory>
#include <vector>

#include "eqopt/batch.hpp"
#include "eqopt/binomial.hpp"
#include "eqopt/black_scholes.hpp"
#include "eqopt/error.hpp"
#include "eqopt/greeks.hpp"
#include "eqopt/implied_vol.hpp"
#include "eqopt/monte_carlo.hpp"
#include "eqopt/thread_pool.hpp"

namespace qpb {
namespace {

using eqopt::OptionType;

OptionType ot(bool call) { return call ? OptionType::Call : OptionType::Put; }
OptionType ot(const std::string& s) { return ot(parse_call(s)); }

template <class T>
T unwrap(eqopt::Result<T>&& r) {
    if (!r) raise_engine([&] { eqopt::throw_error(r.error()); });
    return *std::move(r);
}

[[noreturn]] void batch_fail(const eqopt::BatchError& e) {
    raise_engine([&] { eqopt::throw_error(e.error); }, "element " + std::to_string(e.index) + ": ");
}

nb::dict greeks_dict(const eqopt::BSGreeks& g) {
    nb::dict d;
    d["price"] = g.price;
    d["delta"] = g.delta;
    d["gamma"] = g.gamma;
    d["vega"] = g.vega;
    d["theta"] = g.theta;
    d["rho"] = g.rho;
    d["vanna"] = g.vanna;
    d["volga"] = g.volga;
    return d;
}

struct Batch {
    std::size_t n;
    std::vector<OptionType> types;
    eqopt::BsBatchInputs in;
};

Batch make_batch(const InVec& S, const InVec& K, const InVec& T, const InVec& r,
                 const InVec& sigma, const InVec& q, const InBool& is_call) {
    const std::size_t n = S.shape(0);
    check_same_len(n, K.shape(0), "K");
    check_same_len(n, T.shape(0), "T");
    check_same_len(n, r.shape(0), "r");
    check_same_len(n, sigma.shape(0), "sigma");
    check_same_len(n, q.shape(0), "q");
    check_same_len(n, is_call.shape(0), "is_call");
    Batch b{n, std::vector<OptionType>(n), {}};
    for (std::size_t i = 0; i < n; ++i) b.types[i] = ot(is_call.data()[i]);
    b.in = {span_of(S), span_of(K), span_of(T), span_of(r), span_of(sigma), span_of(q),
            std::span<const OptionType>(b.types)};
    return b;
}

}  // namespace

void bind_eqopt(nb::module_& m) {
    using namespace nb::literals;

    m.def(
        "eq_bs_price",
        [](double S, double K, double T, double r, double sigma, double q,
           const std::string& option_type) {
            return unwrap(eqopt::try_bs_price(S, K, T, r, sigma, q, ot(option_type)));
        },
        "S"_a, "K"_a, "T"_a, "r"_a, "sigma"_a, "q"_a = 0.0, "option_type"_a = "call",
        "Black-Scholes-Merton price (eqopt::try_bs_price).");

    m.def(
        "eq_bs_greeks",
        [](double S, double K, double T, double r, double sigma, double q,
           const std::string& option_type) {
            return greeks_dict(
                unwrap(eqopt::try_bs_greeks(S, K, T, r, sigma, q, ot(option_type))));
        },
        "S"_a, "K"_a, "T"_a, "r"_a, "sigma"_a, "q"_a = 0.0, "option_type"_a = "call",
        "Price and Greeks (eqopt::try_bs_greeks) as a dict.");

    m.def(
        "eq_bs_price_batch",
        [](const InVec& S, const InVec& K, const InVec& T, const InVec& r, const InVec& sigma,
           const InVec& q, const InBool& is_call, bool exact) {
            Batch b = make_batch(S, K, T, r, sigma, q, is_call);
            Owned out = new_array(b.n);
            std::expected<void, eqopt::BatchError> res;
            {
                nb::gil_scoped_release nogil;
                const std::span<double> o(out.data, b.n);
                res = exact ? eqopt::bs_price_batch(b.in, o) : eqopt::bs_price_batch_fast(b.in, o);
            }
            if (!res) batch_fail(res.error());
            return out.array;
        },
        "S"_a, "K"_a, "T"_a, "r"_a, "sigma"_a, "q"_a, "is_call"_a, nb::kw_only(),
        "exact"_a = false,
        "Prices for equal-length float64 arrays. GIL released. Default: the throughput "
        "kernel eqopt::bs_price_batch_fast, within 1e-13 * max(1, S, K) of the exact "
        "price; exact=True calls eqopt::bs_price_batch, equal to eq_bs_price bit for bit.");

    m.def(
        "eq_bs_greeks_batch",
        [](const InVec& S, const InVec& K, const InVec& T, const InVec& r, const InVec& sigma,
           const InVec& q, const InBool& is_call, bool exact) {
            Batch b = make_batch(S, K, T, r, sigma, q, is_call);
            const char* names[8] = {"price", "delta", "gamma", "vega",
                                    "theta", "rho",   "vanna", "volga"};
            Owned o[8] = {new_array(b.n), new_array(b.n), new_array(b.n), new_array(b.n),
                          new_array(b.n), new_array(b.n), new_array(b.n), new_array(b.n)};
            auto sp = [&](int i) { return std::span<double>(o[i].data, b.n); };
            eqopt::BsGreeksBatchOutputs outs{sp(0), sp(1), sp(2), sp(3),
                                             sp(4), sp(5), sp(6), sp(7)};
            std::expected<void, eqopt::BatchError> res;
            {
                nb::gil_scoped_release nogil;
                res = exact ? eqopt::bs_greeks_batch(b.in, outs)
                            : eqopt::bs_greeks_batch_fast(b.in, outs);
            }
            if (!res) batch_fail(res.error());
            nb::dict d;
            for (int i = 0; i < 8; ++i) d[names[i]] = o[i].array;
            return d;
        },
        "S"_a, "K"_a, "T"_a, "r"_a, "sigma"_a, "q"_a, "is_call"_a, nb::kw_only(),
        "exact"_a = false,
        "Greeks for equal-length arrays, dict of arrays. GIL released. Default: "
        "eqopt::bs_greeks_batch_fast, each field within 1e-12 * max(1, S, K, |exact|); "
        "exact=True calls eqopt::bs_greeks_batch, equal to eq_bs_greeks bit for bit.");

    m.def(
        "eq_implied_vol",
        [](double price, double S, double K, double T, double r, double q,
           const std::string& option_type) {
            return unwrap(eqopt::try_implied_vol(price, S, K, T, r, q, ot(option_type)));
        },
        "price"_a, "S"_a, "K"_a, "T"_a, "r"_a, "q"_a = 0.0, "option_type"_a = "call",
        "Implied volatility (eqopt::try_implied_vol).");

    m.def(
        "eq_implied_vol_batch",
        [](const InVec& price, const InVec& S, const InVec& K, const InVec& T, const InVec& r,
           const InVec& q, const InBool& is_call) {
            const std::size_t n = price.shape(0);
            check_same_len(n, S.shape(0), "S");
            check_same_len(n, K.shape(0), "K");
            check_same_len(n, T.shape(0), "T");
            check_same_len(n, r.shape(0), "r");
            check_same_len(n, q.shape(0), "q");
            check_same_len(n, is_call.shape(0), "is_call");
            Owned out = new_array(n);
            std::size_t bad = n;
            eqopt::Error err{};
            {
                nb::gil_scoped_release nogil;
                for (std::size_t i = 0; i < n; ++i) {
                    auto v = eqopt::try_implied_vol(price.data()[i], S.data()[i], K.data()[i],
                                                    T.data()[i], r.data()[i], q.data()[i],
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
        "price"_a, "S"_a, "K"_a, "T"_a, "r"_a, "q"_a, "is_call"_a,
        "Implied vols for equal-length arrays (loop over eqopt::try_implied_vol). "
        "GIL released.");

    m.def(
        "eq_crr_price",
        [](double S, double K, double T, double r, double sigma, double q,
           const std::string& option_type, bool american, int n_steps) {
            std::vector<double> ws(eqopt::crr_workspace_size(n_steps));
            return unwrap(eqopt::try_crr_price(
                S, K, T, r, sigma, q, ot(option_type),
                american ? eqopt::ExerciseStyle::American : eqopt::ExerciseStyle::European,
                n_steps, ws));
        },
        "S"_a, "K"_a, "T"_a, "r"_a, "sigma"_a, "q"_a = 0.0, "option_type"_a = "call",
        nb::kw_only(), "american"_a = true, "n_steps"_a = 500,
        "Cox-Ross-Rubinstein tree price (eqopt::try_crr_price).");

    m.def(
        "eq_mc_price",
        [](double S, double K, double T, double r, double sigma, double q,
           const std::string& option_type, std::size_t n_paths, bool antithetic,
           bool control_variate, std::uint64_t seed, unsigned threads) {
            const OptionType type = ot(option_type);
            eqopt::McWorkspace ws;
            ws.reserve(n_paths, control_variate);
            eqopt::Result<eqopt::MCResult> res;
            {
                nb::gil_scoped_release nogil;
                std::unique_ptr<eqopt::ThreadPool> pool;
                if (threads > 1) pool = std::make_unique<eqopt::ThreadPool>(threads - 1);
                res = eqopt::try_mc_price(S, K, T, r, sigma, q, type, n_paths, antithetic,
                                          control_variate, seed, threads, ws, pool.get());
            }
            const auto v = unwrap(std::move(res));
            nb::dict d;
            d["price"] = v.price;
            d["std_error"] = v.std_error;
            d["ci_low"] = v.ci_low;
            d["ci_high"] = v.ci_high;
            d["n_paths"] = v.n_paths;
            return d;
        },
        "S"_a, "K"_a, "T"_a, "r"_a, "sigma"_a, "q"_a = 0.0, "option_type"_a = "call",
        nb::kw_only(), "n_paths"_a = 100000, "antithetic"_a = true, "control_variate"_a = true,
        "seed"_a = 42, "threads"_a = 1,
        "Monte Carlo price (eqopt::try_mc_price). Deterministic for a given (seed, threads).");
}

}  // namespace qpb
