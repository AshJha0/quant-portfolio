// Unit tests for qp::core. Reference values come from closed forms or from
// mpmath at 50 digits (quoted in the comments); none compares the code with
// itself except the bit-identity checks, which compare two code paths.

#include <gtest/gtest.h>

#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "qp/brent.hpp"
#include "qp/expected.hpp"
#include "qp/iv_core.hpp"
#include "qp/matrix.hpp"
#include "qp/normal.hpp"
#include "qp/rng.hpp"
#include "qp/special.hpp"
#include "qp/thread_pool.hpp"

namespace {

// ------------------------------------------------------------------ normal

TEST(Normal, CdfKnownValues) {
    EXPECT_EQ(qp::norm_cdf(0.0), 0.5);
    // mpmath: ncdf(1.96) = 0.97500210485177952... erfc is not correctly
    // rounded in glibc or the MSVC CRT, so allow 2 ulp (ulp = 1.1e-16 here).
    EXPECT_NEAR(qp::norm_cdf(1.96), 0.9750021048517795, 2.3e-16);
    // Left tail keeps relative accuracy: Mills ratio bound
    // n(x)/|x| (1 - 1/x^2) < N(x) < n(x)/|x| for x < 0.
    const double x = -30.0;
    const double upper = qp::norm_pdf(x) / -x;
    EXPECT_LT(qp::norm_cdf(x), upper);
    EXPECT_GT(qp::norm_cdf(x), upper * (1.0 - 1.0 / (x * x)));
    EXPECT_NEAR(qp::norm_pdf(0.0), 0.3989422804014327, 1e-17);
}

TEST(Normal, RcpFormAgreesToArgumentRounding) {
    // The two forms differ by at most one rounding of the erfc argument
    // t = -x/sqrt 2. d ln erfc(t)/dt ~ 2|t| in the tail, so the relative
    // difference is bounded by (2 t^2 + 2) eps; 4x that is a safe margin.
    for (double x = -8.0; x <= 8.0; x += 0.37) {
        const double a = qp::norm_cdf(x);
        const double b = qp::norm_cdf_rcp(x);
        const double bound = 4.0 * (x * x + 2.0) * qp::kEps;
        EXPECT_LE(std::fabs(a - b) / a, bound) << x;
    }
}

TEST(Normal, PpfInvertsCdf) {
    for (double p : {1e-300, 1e-12, 0.001, 0.02425, 0.3, 0.5, 0.7, 0.97575, 0.999}) {
        const double z = qp::norm_ppf_unchecked<2>(p);
        EXPECT_NEAR(qp::norm_cdf(z) / p, 1.0, 1e-13) << p;
    }
    // Acklam alone is good to ~1.15e-9 relative.
    EXPECT_NEAR(qp::acklam_ppf(0.975), 1.959963984540054, 5e-9);
    // One Halley step: mpmath ndtri(0.975) = 1.9599639845400542.
    EXPECT_NEAR(qp::norm_ppf_unchecked<1>(0.975), 1.9599639845400542, 1e-14);
}

// --------------------------------------------------------------------- rng

TEST(Rng, SplitMixReferenceValue) {
    // splitmix64 with state 0 -> first output of the reference generator.
    EXPECT_EQ(qp::splitmix64(0), 0xe220a8397b1dcdafULL);
    static_assert(qp::splitmix64(0) == 0xe220a8397b1dcdafULL);
}

TEST(Rng, UniformOpenInterval) {
    EXPECT_EQ(qp::uniform_open53(0), 0x1.0p-54);
    // The top 53 bits all ones give (2^53 - 0.5) 2^-53, which is not a
    // double and rounds to 1.0. That word has probability 2^-53 per draw;
    // the engines inherit this from their original code and keep it so no
    // stream changes. Every other word maps into (0, 1).
    EXPECT_EQ(qp::uniform_open53(~std::uint64_t{0}), 1.0);
    EXPECT_LT(qp::uniform_open53(~std::uint64_t{0} - (std::uint64_t{1} << 11)), 1.0);
}

TEST(Rng, GammaMeanAndVariance) {
    // Gamma(k, 1): mean k, variance k. 200k draws: SE of the mean ~ sqrt(k/n).
    std::mt19937_64 eng(7);
    auto u = [&] { return qp::uniform_open53(eng()); };
    auto z = [&] { return qp::norm_ppf_unchecked<1>(u()); };
    for (double k : {0.4, 2.5}) {
        const int n = 200000;
        double s = 0.0, s2 = 0.0;
        for (int i = 0; i < n; ++i) {
            const double g = qp::gamma_marsaglia_tsang(k, u, z);
            s += g;
            s2 += g * g;
        }
        const double mean = s / n;
        const double var = s2 / n - mean * mean;
        EXPECT_NEAR(mean, k, 5.0 * std::sqrt(k / n)) << k;
        EXPECT_NEAR(var / k, 1.0, 0.05) << k;
    }
}

// ----------------------------------------------------------------- special

TEST(Special, LogGamma) {
    EXPECT_NEAR(qp::log_gamma_lanczos(5.0), std::log(24.0), 1e-14);
    // ln Gamma(0.5) = ln sqrt(pi).
    EXPECT_NEAR(qp::log_gamma_lanczos(0.5), 0.5 * std::log(qp::kPi), 1e-14);
    // Reflection branch: ln Gamma(0.1) = 2.252712651734206 (mpmath).
    EXPECT_NEAR(qp::log_gamma_lanczos(0.1), 2.252712651734206, 1e-13);
}

TEST(Special, IncompleteBetaAndGamma) {
    // I_x(1, 1) = x; I_x(a, b) = 1 - I_{1-x}(b, a).
    EXPECT_NEAR(qp::betainc_unchecked(1.0, 1.0, 0.3, 1e-16, 500), 0.3, 1e-15);
    EXPECT_NEAR(qp::betainc_unchecked(2.5, 4.0, 0.2, 1e-16, 500),
                1.0 - qp::betainc_unchecked(4.0, 2.5, 0.8, 1e-16, 500), 1e-14);
    // P(1, x) = 1 - exp(-x).
    EXPECT_NEAR(qp::gamma_p_series(1.0, 0.5, 1e-16), 1.0 - std::exp(-0.5), 1e-15);
    EXPECT_NEAR(qp::gamma_q_cf(1.0, 3.0, 1e-16), std::exp(-3.0), 1e-15);
}

TEST(Special, BinomialCdf) {
    // n = 4, p = 0.5: P(X <= 1) = 5/16.
    EXPECT_NEAR(qp::binomial_cdf_unchecked(1, 4, 0.5, 1e-16, 500), 5.0 / 16.0, 1e-15);
    // Underflow route: (1 - 0.5)^2000 = 0, P(X <= 1000) is just above 0.5.
    const double c = qp::binomial_cdf_unchecked(1000, 2000, 0.5, 1e-16, 500);
    EXPECT_GT(c, 0.5);
    EXPECT_LT(c, 0.51);
}

// ------------------------------------------------------------------ matrix

TEST(Matrix, ConstructionAndAccess) {
    qp::Matrix m(2, 3, 1.5);
    EXPECT_EQ(m.rows(), 2u);
    EXPECT_EQ(m.cols(), 3u);
    EXPECT_EQ(m(1, 2), 1.5);
    m(1, 2) = 4.0;
    EXPECT_EQ(m.row(1)[2], 4.0);
    EXPECT_FALSE(m.square());
    EXPECT_THROW(qp::Matrix(2, 2, std::vector<double>{1.0, 2.0, 3.0}), std::invalid_argument);
}

TEST(Matrix, CholeskyRaw) {
    // [[4, 2], [2, 3]] = L L^T with L = [[2, 0], [1, sqrt 2]].
    const double a[4] = {4.0, 2.0, 2.0, 3.0};
    double l[4];
    ASSERT_TRUE(qp::cholesky_raw(a, 2, l));
    EXPECT_EQ(l[0], 2.0);
    EXPECT_EQ(l[1], 0.0);
    EXPECT_EQ(l[2], 1.0);
    EXPECT_EQ(l[3], std::sqrt(2.0));
    const double b[4] = {1.0, 2.0, 2.0, 1.0};  // indefinite
    EXPECT_FALSE(qp::cholesky_raw(b, 2, l));
}

TEST(Matrix, CholeskyJitterAndRepair) {
    // Rank-one correlation: needs jitter, then factors.
    qp::Matrix c(2, 2, 1.0);
    const auto r = qp::linalg::cholesky(c, qp::linalg::RepairKind::None, "hint");
    EXPECT_GT(r.jitter, 0.0);
    EXPECT_FALSE(r.warning.empty());
    // Materially indefinite: refuses without a repair, factors with one.
    qp::Matrix bad(3, 3, 0.0);
    const double v[9] = {1, 0.9, -0.9, 0.9, 1, 0.9, -0.9, 0.9, 1};
    for (std::size_t i = 0; i < 9; ++i) bad.values()[i] = v[i];
    try {
        (void)qp::linalg::cholesky(bad, qp::linalg::RepairKind::None, "my hint");
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("my hint"), std::string::npos);
    }
    for (auto k : {qp::linalg::RepairKind::Clip, qp::linalg::RepairKind::Higham}) {
        const auto rr = qp::linalg::cholesky(bad, k, "");
        EXPECT_TRUE(rr.repaired);
    }
}

TEST(Matrix, SymmetricEigen) {
    qp::Matrix a(2, 2, 0.0);
    a(0, 0) = 2.0;
    a(0, 1) = 1.0;
    a(1, 0) = 1.0;
    a(1, 1) = 2.0;
    std::vector<double> values;
    qp::Matrix vectors;
    qp::linalg::symmetric_eigen(a, values, vectors);
    ASSERT_EQ(values.size(), 2u);
    EXPECT_NEAR(values[0], 1.0, 1e-15);
    EXPECT_NEAR(values[1], 3.0, 1e-15);
}

TEST(Matrix, MatvecRowsMatchesPlainLoopBitForBit) {
    // 11 rows exercise the vector block, the 4-row block and the remainder.
    const std::size_t rows = 11, cols = 7;
    std::mt19937_64 eng(3);
    std::vector<double> a(rows * cols), x(cols), y(rows);
    for (double& v : a) v = qp::uniform_open53(eng()) - 0.5;
    for (double& v : x) v = qp::uniform_open53(eng()) * 3.0;
    qp::matvec_rows(a.data(), rows, cols, x.data(), y.data());
    for (std::size_t i = 0; i < rows; ++i) {
        double acc = 0.0;
        for (std::size_t j = 0; j < cols; ++j) {
            const double prod = a[i * cols + j] * x[j];
            acc = acc + prod;
        }
        EXPECT_EQ(std::bit_cast<std::uint64_t>(y[i]), std::bit_cast<std::uint64_t>(acc)) << i;
    }
}

// ------------------------------------------------------------------- brent

TEST(Brent, FindsSqrt2) {
    const double r = qp::brentq([](double x) { return x * x - 2.0; }, 0.0, 2.0, 1e-15);
    EXPECT_NEAR(r, std::sqrt(2.0), 4e-15);
    EXPECT_THROW((void)qp::brentq([](double x) { return x * x + 1.0; }, 0.0, 2.0),
                 std::invalid_argument);
}

// ------------------------------------------------------------- thread pool

TEST(ThreadPool, RunsEveryTaskOnce) {
    qp::ThreadPool pool(3);
    std::vector<std::atomic<int>> hits(64);
    auto f = [&](unsigned t) noexcept { hits[t].fetch_add(1); };
    for (int rep = 0; rep < 20; ++rep) pool.for_each_task(64, f);
    for (auto& h : hits) EXPECT_EQ(h.load(), 20);
    EXPECT_EQ(pool.workers(), 3u);
}

TEST(ThreadPool, ResultIndependentOfPoolSize) {
    auto run = [](unsigned workers) {
        qp::ThreadPool pool(workers);
        std::vector<double> out(16);
        auto f = [&](unsigned t) noexcept {
            std::mt19937_64 eng(qp::splitmix64(t));
            double s = 0.0;
            for (int i = 0; i < 1000; ++i) s += qp::uniform_open53(eng());
            out[t] = s;
        };
        pool.for_each_task(16, f);
        return out;
    };
    EXPECT_EQ(run(0), run(4));
}

TEST(ThreadPool, NestedRunOnSamePoolRunsInlineAndDoesNotDeadlock) {
    qp::ThreadPool pool(3);
    std::vector<std::atomic<int>> hits(8 * 8);
    struct Ctx {
        qp::ThreadPool* pool;
        std::vector<std::atomic<int>>* hits;
    } ctx{&pool, &hits};
    auto outer = [&](unsigned i) noexcept {
        auto inner = [&](unsigned j) noexcept { (*ctx.hits)[i * 8 + j].fetch_add(1); };
        ctx.pool->for_each_task(8, inner);
    };
    pool.for_each_task(8, outer);
    for (auto& h : hits) EXPECT_EQ(h.load(), 1);
    // The pool is still usable after the nested job.
    std::atomic<int> n{0};
    auto g = [&](unsigned) noexcept { n.fetch_add(1); };
    pool.for_each_task(10, g);
    EXPECT_EQ(n.load(), 10);
}

TEST(ThreadPool, NestedRunOnDefaultPoolDoesNotDeadlock) {
    std::atomic<int> n{0};
    auto outer = [&](unsigned) noexcept {
        auto inner = [&](unsigned) noexcept { n.fetch_add(1); };
        qp::default_thread_pool().for_each_task(4, inner);
    };
    qp::default_thread_pool().for_each_task(4, outer);
    EXPECT_EQ(n.load(), 16);
}

TEST(ThreadPool, NestedRunOnAnotherPoolUsesItsWorkers) {
    qp::ThreadPool a(2);
    qp::ThreadPool b(2);
    std::atomic<int> n{0};
    auto outer = [&](unsigned) noexcept {
        auto inner = [&](unsigned) noexcept { n.fetch_add(1); };
        b.for_each_task(5, inner);
    };
    a.for_each_task(3, outer);
    EXPECT_EQ(n.load(), 15);
}

TEST(ThreadPool, SpinningPoolGivesSameResult) {
    qp::ThreadPool pool(3, std::chrono::microseconds(50));
    std::vector<std::atomic<int>> hits(32);
    auto f = [&](unsigned t) noexcept { hits[t].fetch_add(1); };
    for (int rep = 0; rep < 5; ++rep) pool.for_each_task(32, f);
    for (auto& h : hits) EXPECT_EQ(h.load(), 5);
}

// ---------------------------------------------------------------- expected

}  // namespace

namespace qp_test_err {
struct Err {
    int code;
};
[[noreturn]] inline void throw_error(const Err& e) {
    throw std::invalid_argument("code " + std::to_string(e.code));
}
}  // namespace qp_test_err

namespace {

TEST(Expected, ValueOrThrowUsesTheEngineThrowError) {
    using R = qp::Result<double, qp_test_err::Err>;
    EXPECT_EQ(qp::value_or_throw(R{2.5}), 2.5);
    EXPECT_THROW(qp::value_or_throw(R{std::unexpected(qp_test_err::Err{7})}),
                 std::invalid_argument);
    using V = qp::Result<void, qp_test_err::Err>;
    EXPECT_NO_THROW(qp::value_or_throw(V{}));
}

// ----------------------------------------------------------------- iv core

TEST(IvCore, RecoversBlackVol) {
    // Black call, F = 100, K = 110, T = 0.5, df = 0.98, sigma = 0.25.
    const double F = 100.0, K = 110.0, T = 0.5, df = 0.98, sigma = 0.25;
    const double s = sigma * std::sqrt(T);
    const double d1 = std::log(F / K) / s + 0.5 * s;
    const double price = df * (F * qp::norm_cdf(d1) - K * qp::norm_cdf(d1 - s));
    int evals = 0;
    double out = 0.0;
    ASSERT_EQ(qp::iv_core::black_implied_vol_nx(price, F, K, T, df, true, evals, out),
              qp::iv_core::IvStatus::Ok);
    EXPECT_NEAR(out, sigma, 1e-13);
    EXPECT_LE(evals, 5);
}

}  // namespace
