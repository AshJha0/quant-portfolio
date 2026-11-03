// W4(b): per-path Philox streams and AS 241 normals in the linear MC kernel.
//
// * AS 241 against mpmath (bisection on mp.ncdf at 50 digits);
// * the batched fill equals scalar draws, bit for bit;
// * the threaded kernel equals path-by-path evaluation, bit for bit, so the
//   result does not depend on the pool size;
// * the stream is pinned to the values the Rust twin pins
//   (rust/equity-var-engine/tests/test_w4b.rs).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../src/path_rng.hpp"
#include "eqvar/kernels.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/monte_carlo.hpp"
#include "eqvar/parametric.hpp"

using namespace eqvar;

namespace {
std::uint64_t bits(double x) {
    std::uint64_t u = 0;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

struct Book {
    std::vector<double> w;
    Matrix cov;
};

Book book(std::size_t n) {
    std::vector<double> vols(n);
    Matrix corr(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        vols[i] = 0.01 + 0.0005 * static_cast<double>(i);
        for (std::size_t j = 0; j < n; ++j) corr(i, j) = i == j ? 1.0 : 0.3;
    }
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) {
        w[i] = 1e6 * (1.0 + static_cast<double>(i % 3)) * (i % 4 == 0 ? -1.0 : 1.0);
    }
    return {w, covariance_from_vols(vols, corr)};
}
}  // namespace

TEST(W4b, As241MatchesMpmath) {
    const double refs[][2] = {
        {1e-300, -37.047096299361199237},
        {1e-100, -21.273453560965324294},
        {1e-20, -9.2623400897984075796},
        {1e-12, -7.0344838253011319326},
        {1e-08, -5.6120012441747887279},
        {1e-05, -4.2648907939228246102},
        {0.001, -3.0902323061678135354},
        {0.0125, -2.2414027276049453536},
        {0.025, -1.9599639845400542118},
        {0.05, -1.644853626951472688},
        {0.075, -1.4395314709384559349},
        {0.1, -1.2815515655446004353},
        {0.2, -0.84162123357291416552},
        {0.3, -0.52440051270804081597},
        {0.4, -0.25334710313579974132},
        {0.45, -0.12566134685507400616},
        {0.49, -0.025068908258711058033},
        {0.5, 0.0},
        {0.51, 0.025068908258711058033},
        {0.6, 0.25334710313579974132},
        {0.7, 0.52440051270804065631},
        {0.75, 0.6744897501960817432},
        {0.8, 0.8416212335729143638},
        {0.9, 1.2815515655446005935},
        {0.95, 1.6448536269514722843},
        {0.975, 1.9599639845400538556},
        {0.99, 2.3263478740408407676},
        {0.999, 3.0902323061678132778},
        {0.99999, 4.2648907939238407699},
        {0.9999999999990905, 7.0477002566644087254},
        {0.9999999999999999, 8.2095361516013868556},
    };
    for (const auto& r : refs) {
        const double got = detail::norm_ppf_as241(r[0]);
        EXPECT_LE(std::abs(got - r[1]), 3e-15 * std::max(std::abs(r[1]), 1e-300)) << r[0];
    }
}

TEST(W4b, BatchedFillEqualsScalarDraws) {
    for (std::size_t len :
         {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{31}, std::size_t{32}}) {
        detail::PathRng a(0xDEADBEEF1234ULL, 5);
        detail::PathRng b(0xDEADBEEF1234ULL, 5);
        double z[32];
        a.fill_normals(z, len);
        for (std::size_t k = 0; k < len; ++k) EXPECT_EQ(bits(z[k]), bits(b.normal())) << len;
        EXPECT_EQ(bits(a.uniform()), bits(b.uniform()));
    }
}

TEST(W4b, StreamPinnedToRustValues) {
    detail::PathRng r(42, 3);
    const double want_z[4] = {0.4416806772623007, -0.3077768475777059, 0.4693154650693704,
                              -1.8602711545492243};
    for (double wz : want_z) EXPECT_EQ(r.normal(), wz);
    EXPECT_EQ(detail::PathRng(42, 3).chi_squared(5.0), 5.767996648372349);
    const Book b = book(50);
    const double v = monte_carlo_var(b.w, b.cov, 0.01, 10'000, Dist::Normal, 6.0, 2026).var;
    const double vt = monte_carlo_var(b.w, b.cov, 0.01, 10'000, Dist::StudentT, 5.0, 2026).var;
    // Same stream and operation order as Rust; libm may differ in the tails,
    // so the cross-language bound is 1e-12 relative (W4 acceptance).
    EXPECT_LE(std::abs(v / 1616611.6346006482 - 1.0), 1e-12) << v;
    EXPECT_LE(std::abs(vt / 1679342.0463568298 - 1.0), 1e-12) << vt;
}

TEST(W4b, ThreadedKernelEqualsPathByPath) {
    const Book b = book(20);
    for (Dist d : {Dist::Normal, Dist::StudentT}) {
        const double df = 5.0;
        Matrix scale = b.cov;
        if (d == Dist::StudentT) {
            for (std::size_t i = 0; i < 400; ++i) scale.data()[i] *= (df - 2.0) / df;
        }
        const Matrix chol = cholesky(scale, Repair::None).lower;
        McVarWorkspace ws;
        ws.reserve(20, 20'000);
        const auto r = try_monte_carlo_var_factored(b.w, {chol.data(), chol.size()}, 0.01, 20'000,
                                                    d, df, 77, 10, ws);
        ASSERT_TRUE(r.has_value());
        std::vector<double> v(20, 0.0);
        for (std::size_t i = 0; i < 20; ++i)
            for (std::size_t j = 0; j <= i; ++j) v[j] += chol(i, j) * b.w[i];
        for (std::size_t p = 0; p < 20'000; p += 997) {
            const double one = detail::linear_path(v.data(), 20, d == Dist::StudentT, df, 77, p);
            EXPECT_EQ(bits(ws.pnl[p]), bits(one)) << p;
        }
    }
}

TEST(W4b, RebaselineWithin3SeOfClosedForm) {
    const Book b = book(50);
    const MonteCarloResult r = monte_carlo_var(b.w, b.cov, 0.01, 10'000, Dist::Normal, 6.0, 2026);
    const double exact = parametric_var(b.w, b.cov, 0.01);
    EXPECT_LE(std::abs(r.var - exact), 3.0 * r.var_se) << r.var << " " << exact;
}

TEST(W4b, ThreadedBootstrapEqualsSequential) {
    const Book b = book(50);
    const Matrix chol = cholesky(b.cov, Repair::None).lower;
    McVarWorkspace with_pool;
    with_pool.reserve(50, 10'000);
    McVarWorkspace without_pool;
    without_pool.reserve(50, 10'000);
    without_pool.boot.clear();  // no slots: sequential bootstrap
    const auto a = try_monte_carlo_var_factored(b.w, {chol.data(), chol.size()}, 0.01, 10'000,
                                                Dist::Normal, 6.0, 5, 64, with_pool);
    const auto s = try_monte_carlo_var_factored(b.w, {chol.data(), chol.size()}, 0.01, 10'000,
                                                Dist::Normal, 6.0, 5, 64, without_pool);
    ASSERT_TRUE(a.has_value() && s.has_value());
    EXPECT_EQ(bits(a->var_se), bits(s->var_se));
    EXPECT_EQ(bits(a->var), bits(s->var));
    EXPECT_GT(a->var_se, 0.0);
}
