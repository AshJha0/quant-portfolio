// Per-call latency benchmark for the eqopt engine.
//
//   ./eqopt_bench [--out results.json] [--label after] [--quick]
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

#include "eqopt/binomial.hpp"
#include "eqopt/black76.hpp"
#include "eqopt/black_scholes.hpp"
#include "eqopt/greeks.hpp"
#include "eqopt/implied_vol.hpp"
#include "eqopt/monte_carlo.hpp"
#include "latency.hpp"

#if __has_include("eqopt/batch.hpp")
#include "eqopt/batch.hpp"
#include "eqopt/thread_pool.hpp"
#define EQOPT_BENCH_NEW_API 1
#endif

namespace {

using eqopt::OptionType;

struct Quote {
    double S, K, T, r, sigma, q;
    OptionType type;
};

constexpr std::size_t kTable = 1 << 14;  // 16384 distinct quotes

std::vector<Quote> make_quotes() {
    qpbench::InputGen g(20261101);
    std::vector<Quote> v(kTable);
    for (std::size_t i = 0; i < kTable; ++i) {
        v[i] = Quote{g.uniform(60.0, 140.0), g.uniform(80.0, 120.0), g.uniform(0.02, 2.0),
                     g.uniform(-0.01, 0.06), g.uniform(0.08, 0.6), g.uniform(0.0, 0.04),
                     (i & 1U) != 0 ? OptionType::Put : OptionType::Call};
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    qpbench::Bench b(argc, argv, "equity-options");
    const std::vector<Quote> qs = make_quotes();
    const auto at = [&](std::size_t i) -> const Quote& { return qs[i & (kTable - 1)]; };
    double acc = 0.0;

    b.run("bs_price", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += eqopt::bs_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type);
    }, 100000, 1000000);

    b.run("bs_greeks", [&](std::size_t i) {
        const Quote& x = at(i);
        const auto g = eqopt::bs_greeks(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type);
        acc += g.delta + g.vega;
    }, 100000, 1000000);

    b.run("black76_price", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += eqopt::black76_price(x.S, x.K, x.T, x.r, x.sigma, x.type);
    }, 100000, 1000000);

    // Implied-vol table: quotes whose price is strictly inside the
    // no-arbitrage bounds (deep OTM short-dated prices round to the bound).
    std::vector<Quote> ivq;
    std::vector<double> prices;
    for (std::size_t i = 0; ivq.size() < kTable; ++i) {
        const Quote& x = qs[i % kTable];
        const double p = eqopt::bs_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type);
        try {
            (void)eqopt::implied_vol(p, x.S, x.K, x.T, x.r, x.q, x.type);
        } catch (const std::exception&) {
            continue;
        }
        ivq.push_back(x);
        prices.push_back(p);
    }
    b.run("implied_vol", [&](std::size_t i) {
        const Quote& x = ivq[i & (kTable - 1)];
        acc += eqopt::implied_vol(prices[i & (kTable - 1)], x.S, x.K, x.T, x.r, x.q, x.type);
    }, 20000, 300000);

    b.run("crr_price eu n=1000", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += eqopt::crr_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type,
                                eqopt::ExerciseStyle::European, 1000);
    }, 200, 3000);

    b.run("crr_price am n=1000", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += eqopt::crr_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type,
                                eqopt::ExerciseStyle::American, 1000);
    }, 200, 3000);

    b.run("mc_price 10k paths 1 thr", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += eqopt::mc_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type, 10000, true, true,
                               42 + i, 1).price;
    }, 50, 2000);

    b.run("mc_price 10k paths 4 thr", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += eqopt::mc_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type, 10000, true, true,
                               42 + i, 4).price;
    }, 50, 2000);

    b.run("mc_price 1M paths 4 thr", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += eqopt::mc_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type, 1000000, true,
                               true, 42 + i, 4).price;
    }, 3, 40);

#ifdef EQOPT_BENCH_NEW_API
    b.run("try_bs_price", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += eqopt::try_bs_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type).value_or(0.0);
    }, 100000, 1000000);

    b.run("try_bs_greeks", [&](std::size_t i) {
        const Quote& x = at(i);
        const auto g = eqopt::try_bs_greeks(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type);
        acc += g ? g->delta + g->vega : 0.0;
    }, 100000, 1000000);

    b.run("try_implied_vol", [&](std::size_t i) {
        const Quote& x = ivq[i & (kTable - 1)];
        acc += eqopt::try_implied_vol(prices[i & (kTable - 1)], x.S, x.K, x.T, x.r, x.q,
                                      x.type).value_or(0.0);
    }, 20000, 300000);

    std::vector<double> tree_ws(eqopt::crr_workspace_size(1000));
    b.run("try_crr_price am n=1000 (ws)", [&](std::size_t i) {
        const Quote& x = at(i);
        acc += eqopt::try_crr_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type,
                                    eqopt::ExerciseStyle::American, 1000, tree_ws)
                   .value_or(0.0);
    }, 200, 3000);

    eqopt::ThreadPool pool(4);
    eqopt::McWorkspace mc_ws;
    mc_ws.reserve(1000000, true);
    b.run("try_mc_price 10k paths 4 thr (pool)", [&](std::size_t i) {
        const Quote& x = at(i);
        const auto r = eqopt::try_mc_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type, 10000,
                                           true, true, 42 + i, 4, mc_ws, &pool);
        acc += r ? r->price : 0.0;
    }, 50, 2000);
    b.run("try_mc_price 1M paths 4 thr (pool)", [&](std::size_t i) {
        const Quote& x = at(i);
        const auto r = eqopt::try_mc_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type, 1000000,
                                           true, true, 42 + i, 4, mc_ws, &pool);
        acc += r ? r->price : 0.0;
    }, 3, 40);

    // Batch: 1024 quotes per call, structure of arrays.
    constexpr std::size_t kBatch = 1024;
    const std::size_t n_batches = kTable / kBatch;
    std::vector<double> S(kTable), K(kTable), T(kTable), r(kTable), sig(kTable), q(kTable);
    std::vector<OptionType> ty(kTable);
    for (std::size_t i = 0; i < kTable; ++i) {
        S[i] = qs[i].S; K[i] = qs[i].K; T[i] = qs[i].T; r[i] = qs[i].r;
        sig[i] = qs[i].sigma; q[i] = qs[i].q; ty[i] = qs[i].type;
    }
    std::vector<double> out(kBatch), o2(kBatch), o3(kBatch), o4(kBatch), o5(kBatch),
        o6(kBatch), o7(kBatch), o8(kBatch);
    auto slice = [&](const auto& v, std::size_t i) {
        return std::span(v).subspan((i % n_batches) * kBatch, kBatch);
    };
    b.run("bs_price_batch x1024", [&](std::size_t i) {
        const eqopt::BsBatchInputs in{slice(S, i), slice(K, i), slice(T, i),
                                      slice(r, i), slice(sig, i), slice(q, i), slice(ty, i)};
        (void)eqopt::bs_price_batch(in, out);
        acc += out[i & (kBatch - 1)];
    }, 200, 20000);
    b.run("bs_greeks_batch x1024", [&](std::size_t i) {
        const eqopt::BsBatchInputs in{slice(S, i), slice(K, i), slice(T, i),
                                      slice(r, i), slice(sig, i), slice(q, i), slice(ty, i)};
        const eqopt::BsGreeksBatchOutputs o{out, o2, o3, o4, o5, o6, o7, o8};
        (void)eqopt::bs_greeks_batch(in, o);
        acc += o2[i & (kBatch - 1)];
    }, 200, 20000);
#endif

    qpbench::g_sink = acc;
    return 0;
}
