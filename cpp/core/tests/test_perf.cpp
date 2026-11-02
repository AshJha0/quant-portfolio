// Tests for the Phase 6 kernels: Philox (known-answer vectors), O(n)
// quantiles and blocked GEMM (bit for bit against the naive code), and the
// batch exp/erfc/normal cdf (documented tolerances against libm).

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include "qp/gemm.hpp"
#include "qp/philox.hpp"
#include "qp/quantile.hpp"
#include "qp/thread_pool.hpp"
#include "qp/vmath.hpp"

namespace {

// ------------------------------------------------------------------ Philox

// Random123 1.14 examples/kat_vectors, philox4x32 10 rounds.
TEST(Philox, KnownAnswerVectors) {
    EXPECT_EQ(qp::philox4x32<10>({0, 0, 0, 0}, {0, 0}),
              (qp::PhiloxCounter{0x6627e8d5U, 0xe169c58dU, 0xbc57ac4cU, 0x9b00dbd8U}));
    EXPECT_EQ(qp::philox4x32<10>({0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU},
                                 {0xffffffffU, 0xffffffffU}),
              (qp::PhiloxCounter{0x408f276dU, 0x41c83b0eU, 0xa20bc7c6U, 0x6d5451fdU}));
    EXPECT_EQ(qp::philox4x32<10>({0x243f6a88U, 0x85a308d3U, 0x13198a2eU, 0x03707344U},
                                 {0xa4093822U, 0x299f31d0U}),
              (qp::PhiloxCounter{0xd16cfe09U, 0x94fdccebU, 0x5001e420U, 0x24126ea1U}));
}

TEST(Philox, SeekMatchesSequentialAndStreamsDiffer) {
    qp::PhiloxStream a(42, 7);
    std::vector<std::uint64_t> seq(9);
    for (auto& w : seq) w = a.next_u64();
    for (std::uint64_t i = 0; i < seq.size(); ++i) {
        qp::PhiloxStream b(42, 7);
        b.seek(i);
        EXPECT_EQ(b.next_u64(), seq[i]) << i;
    }
    qp::PhiloxStream other(42, 8);
    EXPECT_NE(other.next_u64(), seq[0]);
}

TEST(Philox, UniformMeanAndRange) {
    qp::PhiloxStream s(1, 0);
    double sum = 0.0;
    constexpr int kN = 200000;
    for (int i = 0; i < kN; ++i) {
        const double u = s.uniform();
        ASSERT_GT(u, 0.0);
        ASSERT_LT(u, 1.0);
        sum += u;
    }
    // Mean of U(0,1) is 1/2 with sd 1/sqrt(12 n) = 6.5e-4; allow 5 sd.
    EXPECT_NEAR(sum / kN, 0.5, 5 * 6.5e-4);
}

// --------------------------------------------------------------- quantiles

std::vector<double> random_vec(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    std::vector<double> v(n);
    for (auto& x : v) x = static_cast<double>(static_cast<std::int64_t>(g() >> 11)) * 1e-12 - 4.5e3;
    return v;
}

TEST(Quantile, SelectAndType7MatchSortBitForBit) {
    for (std::size_t n : {1U, 2U, 3U, 10U, 101U, 1000U, 4097U}) {
        const auto x = random_vec(n, n);
        auto sorted = x;
        std::sort(sorted.begin(), sorted.end());
        for (double q : {0.0, 0.01, 0.025, 0.5, 0.975, 0.99, 1.0}) {
            const double h = static_cast<double>(n - 1) * q;
            const auto lo = static_cast<std::size_t>(std::floor(h));
            const std::size_t hi = std::min(lo + 1, n - 1);
            const double ref = sorted[lo] + (h - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
            auto work = x;
            EXPECT_EQ(std::bit_cast<std::uint64_t>(qp::quantile_type7(work, q)),
                      std::bit_cast<std::uint64_t>(ref))
                << n << " " << q;
            auto w2 = x;
            EXPECT_EQ(qp::select_kth(w2, lo), sorted[lo]);
        }
    }
}

// -------------------------------------------------------------------- GEMM

TEST(Gemm, BlockedEqualsNaiveBitForBit) {
    const std::size_t m = 70, k = 131, n = 67;  // not multiples of the tile
    const auto a = random_vec(m * k, 1), b = random_vec(k * n, 2);
    std::vector<double> c(m * n, 0.0), ref(m * n, 0.0);
    qp::gemm_blocked(a.data(), b.data(), c.data(), m, k, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t p = 0; p < k; ++p) {
                const double prod = a[i * k + p] * b[p * n + j];
                s = s + prod;
            }
            ref[i * n + j] = s;
        }
    EXPECT_EQ(c, ref);
}

TEST(Gemm, SyrkUpperEqualsRowStreamingCovarianceLoop) {
    const std::size_t t = 300, n = 77;
    const auto d = random_vec(t * n, 3);
    std::vector<double> c(n * n, 0.0), ref(n * n, 0.0);
    qp::syrk_upper_blocked(d.data(), c.data(), t, n);
    for (std::size_t r = 0; r < t; ++r)
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i; j < n; ++j) ref[i * n + j] += d[r * n + i] * d[r * n + j];
    EXPECT_EQ(c, ref);
}

// ------------------------------------------------------------ batch maths

double rel_err(double got, double ref) { return std::abs(got - ref) / std::abs(ref); }

TEST(VMath, ExpWithinTwoUlp) {
    std::mt19937_64 g(9);
    std::uniform_real_distribution<double> dist(-708.0, 709.0);
    std::vector<double> x(100000), y(x.size());
    for (auto& v : x) v = dist(g);
    x[0] = 0.0;
    x[1] = 1.0;
    x[2] = -1e-300;
    qp::vm::exp_batch(x, y);
    double worst = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) worst = std::max(worst, rel_err(y[i], std::exp(x[i])));
    EXPECT_LE(worst, 4.5e-16);  // 2 ulp at the top of a binade
    EXPECT_EQ(y[0], 1.0);
}

TEST(VMath, ExpUnderflowAndOverflow) {
    const std::vector<double> x{-800.0, -745.5, 710.5, 800.0};
    std::vector<double> y(x.size());
    qp::vm::exp_batch(x, y);
    EXPECT_EQ(y[0], 0.0);
    EXPECT_LE(y[1], 5e-324);
    EXPECT_TRUE(std::isinf(y[2]));
    EXPECT_TRUE(std::isinf(y[3]));
}

TEST(VMath, ErfcWithinDocumentedTolerance) {
    std::vector<double> x;
    for (double v = -6.0; v <= 26.5; v += 1.0 / 512.0) x.push_back(v);
    std::vector<double> y(x.size());
    qp::vm::erfc_batch(x, y);
    double worst = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) worst = std::max(worst, rel_err(y[i], std::erfc(x[i])));
    EXPECT_LE(worst, 2e-15);
    const std::vector<double> far{30.0, -30.0};
    std::vector<double> yf(2);
    qp::vm::erfc_batch(far, yf);
    EXPECT_EQ(yf[0], 0.0);
    EXPECT_EQ(yf[1], 2.0);
}

TEST(VMath, NormCdfAbsoluteAndRelative) {
    std::vector<double> x;
    for (double v = -37.0; v <= 9.0; v += 1.0 / 256.0) x.push_back(v);
    std::vector<double> y(x.size());
    qp::vm::norm_cdf_batch(x, y);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double ref = 0.5 * std::erfc(-x[i] * 0x1.6a09e667f3bcdp-1);  // same argument
        ASSERT_LE(std::abs(y[i] - ref), 4e-16) << x[i];
        ASSERT_LE(rel_err(y[i], ref), 2e-15) << x[i];
    }
}

// ------------------------------------------------- thread pool determinism

TEST(ThreadPool, PhiloxPathsIndependentOfScheduling) {
    // Each chunk draws its own Philox stream, so the per-chunk sums are the
    // same whatever thread runs them and in whatever order.
    constexpr std::size_t kChunks = 64;
    std::vector<double> serial(kChunks), pooled(kChunks);
    auto work = [](std::size_t c) {
        qp::PhiloxStream s(2026, c);
        double acc = 0.0;
        for (int i = 0; i < 1000; ++i) acc += s.uniform();
        return acc;
    };
    for (std::size_t c = 0; c < kChunks; ++c) serial[c] = work(c);
    qp::ThreadPool pool(3);
    auto task = [&](unsigned c) noexcept { pooled[c] = work(c); };
    pool.for_each_task(static_cast<unsigned>(kChunks), task);
    EXPECT_EQ(serial, pooled);
}

}  // namespace
