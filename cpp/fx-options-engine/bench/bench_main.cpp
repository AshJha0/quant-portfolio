// Per-call latency benchmark for the fxopt engine.
//
//   ./fxopt_bench [--out results.json] [--label after] [--quick]
//
// Every kernel is timed call by call (see latency.hpp) over inputs drawn
// from a fixed pseudo-random table, so branch predictors and caches see a
// realistic mix rather than one repeated quote. Numbers depend on the
// machine; under WSL2 the hypervisor adds scheduling jitter that shows up in
// p99.9 and max.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>
#include <vector>

#include "fxopt/binomial.hpp"
#include "fxopt/black76.hpp"
#include "fxopt/deltas.hpp"
#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/greeks.hpp"
#include "fxopt/implied_vol.hpp"
#include "fxopt/monte_carlo.hpp"
#include "qp/bench/latency.hpp"

#if __has_include("fxopt/batch.hpp")
#include "fxopt/batch.hpp"
#include "fxopt/thread_pool.hpp"
#define FXOPT_BENCH_NEW_API 1
#endif

namespace {

using fxopt::OptionType;

struct Quote {
    double S, K, T, rd, rf, sigma;
    OptionType type;
};

constexpr std::size_t kTable = 1 << 14;  // 16384 distinct quotes

std::vector<Quote> make_quotes() {
    qpbench::InputGen g(20261101);
    std::vector<Quote> v(kTable);
    for (std::size_t i = 0; i < kTable; ++i) {
        const double S = g.uniform(1.0, 1.2);
        v[i] = Quote{S, S * g.uniform(0.85, 1.15), g.uniform(0.02, 2.0),
                     g.uniform(-0.01, 0.06), g.uniform(-0.01, 0.05), g.uniform(0.04, 0.3),
                     (i & 1U) != 0 ? OptionType::Put : OptionType::Call};
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    qpbench::Bench b(argc, argv, "fx-options");
    const std::vector<Quote> qs = make_quotes();
    const auto at = [&](std::size_t i) -> const Quote& { return qs[i & (kTable - 1)]; };
    double acc = 0.0;

    b.run("gk_price", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += fxopt::gk_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type);
    }, 100000, 1000000);

    b.run("analytic_greeks", [&](std::size_t i) {
        const Quote& x = at(i);
        const auto g = fxopt::analytic_greeks(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type);
        acc += g.delta_spot + g.vega;
    }, 100000, 1000000);

    b.run("black76_price", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += fxopt::black76_price(x.S, x.K, x.T, x.rd, x.sigma, x.type);
    }, 100000, 1000000);

    // Implied-vol table: quotes whose price is strictly inside the
    // no-arbitrage bounds (deep OTM short-dated prices round to the bound).
    std::vector<Quote> ivq;
    std::vector<double> prices;
    for (std::size_t i = 0; ivq.size() < kTable; ++i) {
        const Quote& x = qs[i % kTable];
        const double p = fxopt::gk_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type);
        try {
            if (fxopt::implied_vol(p, x.S, x.K, x.T, x.rd, x.rf, x.type) <= 0.0) continue;
        } catch (const std::exception&) {
            continue;
        }
        ivq.push_back(x);
        prices.push_back(p);
    }
    b.run("implied_vol", [&](std::size_t i) {
        const Quote& x = ivq[i & (kTable - 1)];
        acc += fxopt::implied_vol(prices[i & (kTable - 1)], x.S, x.K, x.T, x.rd, x.rf, x.type);
    }, 20000, 300000);

    b.run("strike_from_delta 25d spot", [&](std::size_t i) {
        const Quote& x = at(i);
        const double d = x.type == OptionType::Call ? 0.25 : -0.25;
        acc += fxopt::strike_from_delta(d, x.S, x.T, x.rd, x.rf, x.sigma, x.type,
                                        fxopt::DeltaConvention::Spot);
    }, 20000, 300000);

    b.run("binomial_price eu n=1000", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += fxopt::binomial_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type, 1000,
                                     fxopt::Exercise::European);
    }, 200, 3000);

    b.run("binomial_price am n=1000", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += fxopt::binomial_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type, 1000,
                                     fxopt::Exercise::American);
    }, 200, 3000);

    b.run("mc_price 10k paths 1 thr", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += fxopt::mc_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type, 10000,
                               fxopt::Seed{42 + i}, true, true).price;
    }, 50, 2000);

    b.run("mc_price 1M paths 1 thr", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += fxopt::mc_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type, 1000000,
                               fxopt::Seed{42 + i}, true, true).price;
    }, 3, 40);

#ifdef FXOPT_BENCH_NEW_API
    b.run("try_gk_price", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += fxopt::try_gk_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type).value_or(0.0);
    }, 100000, 1000000);

    b.run("try_analytic_greeks", [&](std::size_t i) {
        const Quote& x = at(i);
        const auto g = fxopt::try_analytic_greeks(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type);
        acc += g ? g->delta_spot + g->vega : 0.0;
    }, 100000, 1000000);

    b.run("try_implied_vol", [&](std::size_t i) {
        const Quote& x = ivq[i & (kTable - 1)];
        acc += fxopt::try_implied_vol(prices[i & (kTable - 1)], x.S, x.K, x.T, x.rd, x.rf,
                                      x.type).value_or(0.0);
    }, 20000, 300000);

    std::vector<double> tree_ws(fxopt::binomial_workspace_size(1000));
    b.run("try_binomial_price am n=1000 (ws)", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += fxopt::try_binomial_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type, 1000,
                                         fxopt::Exercise::American, tree_ws)
                   .value_or(0.0);
    }, 200, 3000);

    fxopt::ThreadPool pool(4);
    fxopt::McWorkspace mc_ws;
    mc_ws.reserve(1000000, true);
    b.run("try_mc_price 10k paths 1 thr (ws)", [&](std::size_t i) {
        const Quote& x = at(i);
        const auto r = fxopt::try_mc_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type, 10000,
                                           fxopt::Seed{42 + i}, true, true, 1, mc_ws, nullptr);
        acc += r ? r->price : 0.0;
    }, 50, 2000);
    b.run("try_mc_price 10k paths 4 thr (pool)", [&](std::size_t i) {
        const Quote& x = at(i);
        const auto r = fxopt::try_mc_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type, 10000,
                                           fxopt::Seed{42 + i}, true, true, 4, mc_ws, &pool);
        acc += r ? r->price : 0.0;
    }, 50, 2000);
    b.run("try_mc_price 1M paths 4 thr (pool)", [&](std::size_t i) {
        const Quote& x = at(i);
        const auto r = fxopt::try_mc_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type, 1000000,
                                           fxopt::Seed{42 + i}, true, true, 4, mc_ws, &pool);
        acc += r ? r->price : 0.0;
    }, 3, 40);

    // Batch: 1024 quotes per call, structure of arrays.
    constexpr std::size_t kBatch = 1024;
    const std::size_t n_batches = kTable / kBatch;
    std::vector<double> S(kTable), K(kTable), T(kTable), rd(kTable), rf(kTable), sig(kTable);
    std::vector<OptionType> ty(kTable);
    for (std::size_t i = 0; i < kTable; ++i) {
        S[i] = qs[i].S; K[i] = qs[i].K; T[i] = qs[i].T; rd[i] = qs[i].rd;
        rf[i] = qs[i].rf; sig[i] = qs[i].sigma; ty[i] = qs[i].type;
    }
    std::vector<double> o[10];
    for (auto& v : o) v.resize(kBatch);
    auto slice = [&](const auto& v, std::size_t i) {
        return std::span(v).subspan((i % n_batches) * kBatch, kBatch);
    };
    b.run("gk_price_batch x1024", [&](std::size_t i) {
        const fxopt::GkBatchInputs in{slice(S, i), slice(K, i), slice(T, i), slice(rd, i),
                                      slice(rf, i), slice(sig, i), slice(ty, i)};
        (void)fxopt::gk_price_batch(in, o[0]);
        acc += o[0][i & (kBatch - 1)];
    }, 200, 20000);
    b.run("gk_greeks_batch x1024", [&](std::size_t i) {
        const fxopt::GkBatchInputs in{slice(S, i), slice(K, i), slice(T, i), slice(rd, i),
                                      slice(rf, i), slice(sig, i), slice(ty, i)};
        const fxopt::GkGreeksBatchOutputs out{o[0], o[1], o[2], o[3], o[4],
                                              o[5], o[6], o[7], o[8], o[9]};
        (void)fxopt::gk_greeks_batch(in, out);
        acc += o[1][i & (kBatch - 1)];
    }, 200, 20000);
#endif

    qpbench::g_sink = acc;
    return 0;
}
