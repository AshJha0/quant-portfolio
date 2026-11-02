// Per-call latency (p50/p90/p99/p99.9) of the Phase 6 core kernels next to
// the code they speed up. Uses the shared harness (qp/bench/latency.hpp),
// which records compiler, flags, CPU model and timer with every number.
//
//   bench_core [--out results.json] [--label NAME] [--quick] [--only TEXT]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "qp/bench/latency.hpp"
#include "qp/gemm.hpp"
#include "qp/philox.hpp"
#include "qp/quantile.hpp"
#include "qp/vmath.hpp"

int main(int argc, char** argv) {
    qpbench::Bench b(argc, argv, "core");
    double acc = 0.0;
    constexpr std::size_t kN = 1024;
    std::vector<double> x(kN), y(kN), xe(kN);
    {
        qpbench::InputGen g(1);
        for (std::size_t i = 0; i < kN; ++i) {
            x[i] = g.uniform(-6.0, 6.0);
            xe[i] = g.uniform(-50.0, 50.0);
        }
    }

    b.run(
        "std::erfc x1024 (scalar)",
        [&](std::size_t) {
            for (std::size_t i = 0; i < kN; ++i) y[i] = std::erfc(x[i]);
            acc += y[kN - 1];
        },
        200, 20000);
    b.run(
        "vm::erfc_batch x1024",
        [&](std::size_t) {
            qp::vm::erfc_batch(x, y);
            acc += y[kN - 1];
        },
        200, 20000);
    b.run(
        "std::exp x1024 (scalar)",
        [&](std::size_t) {
            for (std::size_t i = 0; i < kN; ++i) y[i] = std::exp(xe[i]);
            acc += y[kN - 1];
        },
        200, 20000);
    b.run(
        "vm::exp_batch x1024",
        [&](std::size_t) {
            qp::vm::exp_batch(xe, y);
            acc += y[kN - 1];
        },
        200, 20000);
    b.run(
        "vm::norm_cdf_batch x1024",
        [&](std::size_t) {
            qp::vm::norm_cdf_batch(x, y);
            acc += y[kN - 1];
        },
        200, 20000);

    std::mt19937_64 mt(7);
    b.run(
        "mt19937_64 x1024 words",
        [&](std::size_t) {
            std::uint64_t s = 0;
            for (std::size_t i = 0; i < kN; ++i) s ^= mt();
            acc += static_cast<double>(s & 1U);
        },
        200, 20000);
    b.run(
        "philox4x32-10 x1024 words",
        [&](std::size_t i) {
            qp::PhiloxStream ps(7, i);
            std::uint64_t s = 0;
            for (std::size_t k = 0; k < kN; ++k) s ^= ps.next_u64();
            acc += static_cast<double>(s & 1U);
        },
        200, 20000);

    std::vector<double> pnl(10000), work(pnl.size());
    {
        qpbench::InputGen g(2);
        for (auto& v : pnl) v = g.uniform(-1e6, 1e6);
    }
    b.run(
        "quantile by full sort n=10000",
        [&](std::size_t) {
            work = pnl;
            std::sort(work.begin(), work.end());
            acc += work[99];
        },
        100, 5000);
    b.run(
        "quantile_type7 (select) n=10000",
        [&](std::size_t) {
            work = pnl;
            acc += qp::quantile_type7(work, 0.01);
        },
        100, 5000);

    constexpr std::size_t kM = 128;
    std::vector<double> a(kM * kM), bm(kM * kM), c(kM * kM);
    {
        qpbench::InputGen g(3);
        for (auto& v : a) v = g.uniform(-1.0, 1.0);
        for (auto& v : bm) v = g.uniform(-1.0, 1.0);
    }
    b.run(
        "gemm naive 128^3",
        [&](std::size_t) {
            std::fill(c.begin(), c.end(), 0.0);
            for (std::size_t i = 0; i < kM; ++i)
                for (std::size_t j = 0; j < kM; ++j) {
                    double s = 0.0;
                    for (std::size_t p = 0; p < kM; ++p) s += a[i * kM + p] * bm[p * kM + j];
                    c[i * kM + j] = s;
                }
            acc += c[0];
        },
        20, 1000);
    b.run(
        "gemm_blocked 128^3",
        [&](std::size_t) {
            std::fill(c.begin(), c.end(), 0.0);
            qp::gemm_blocked(a.data(), bm.data(), c.data(), kM, kM, kM);
            acc += c[0];
        },
        20, 1000);

    qpbench::g_sink = acc;
    return 0;
}
