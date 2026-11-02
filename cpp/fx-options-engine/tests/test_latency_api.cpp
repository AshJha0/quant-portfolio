// The noexcept kernel API, the batch pricer and the thread pool: each must
// reproduce the throwing scalar API exactly (bit for bit), and chunked MC
// results must not depend on which pool runs the chunks.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "fxopt/batch.hpp"
#include "fxopt/binomial.hpp"
#include "fxopt/black76.hpp"
#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/greeks.hpp"
#include "fxopt/implied_vol.hpp"
#include "fxopt/monte_carlo.hpp"
#include "fxopt/thread_pool.hpp"

using namespace fxopt;

namespace {

static_assert(noexcept(try_gk_price(1.0, 1.0, 1.0, 0.0, 0.0, 0.2, OptionType::Call)));
static_assert(noexcept(try_d1(1.0, 1.0, 1.0, 0.0, 0.0, 0.2)));
static_assert(noexcept(try_analytic_greeks(1.0, 1.0, 1.0, 0.0, 0.0, 0.2, OptionType::Call)));
static_assert(noexcept(try_black76_price(1.0, 1.0, 1.0, 0.0, 0.2, OptionType::Call)));
static_assert(noexcept(try_implied_vol(0.1, 1.0, 1.0, 1.0, 0.0, 0.0, OptionType::Call)));
static_assert(noexcept(try_binomial_price(1.0, 1.0, 1.0, 0.0, 0.0, 0.2, OptionType::Call, 10,
                                          Exercise::American, std::span<double>{})));
static_assert(noexcept(gk_price_batch(std::declval<const GkBatchInputs&>(), std::span<double>{})));
static_assert(noexcept(gk_greeks_batch(std::declval<const GkBatchInputs&>(),
                                       std::declval<const GkGreeksBatchOutputs&>())));
static_assert(noexcept(try_mc_price(1.0, 1.0, 1.0, 0.0, 0.0, 0.2, OptionType::Call, 100, Seed{1},
                                    true, true, 1, std::declval<McWorkspace&>(), nullptr)));
static_assert(std::is_trivially_copyable_v<Error>);
static_assert(std::is_trivially_copyable_v<McEstimate>);

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

struct Q {
    double S, K, T, rd, rf, sigma;
    OptionType type;
};

std::vector<Q> grid() {
    std::vector<Q> v;
    for (double S : {0.6, 1.08, 1.1, 150.0})
        for (double K : {0.5, 1.0, 1.1, 160.0})
            for (double T : {0.0, 1.0 / 365.0, 0.5, 3.0})
                for (double sigma : {0.0, 0.05, 0.3, 1.5})
                    for (double rd : {-0.01, 0.04})
                        for (OptionType t : {OptionType::Call, OptionType::Put})
                            v.push_back({S, K, T, rd, 0.02, sigma, t});
    return v;
}

bool greeks_domain(const Q& x) { return x.T > 0 && x.sigma > 0; }

}  // namespace

TEST(NoexceptApi, GkPriceEqualsThrowingApiBitForBit) {
    for (const Q& x : grid()) {
        const auto r = try_gk_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type);
        ASSERT_TRUE(r.has_value());
        EXPECT_TRUE(same_bits(*r, gk_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type)));
    }
}

TEST(NoexceptApi, ErrorsCarryTheThrowingMessage) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto r = try_gk_price(-1.0, 1.0, 1.0, 0.0, 0.0, 0.2, OptionType::Call);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, Errc::SpotNonPositive);
    try {
        (void)gk_price(-1.0, 1.0, 1.0, 0.0, 0.0, 0.2, OptionType::Call);
        FAIL() << "expected throw";
    } catch (const std::invalid_argument& e) {
        EXPECT_EQ(message(r.error()), e.what());
    }
    EXPECT_EQ(try_gk_price(1.0, 1.0, 1.0, nan, 0.0, 0.2, OptionType::Call).error().code,
              Errc::NonFinite);
    EXPECT_EQ(try_analytic_greeks(1.0, 1.0, 0.0, 0.0, 0.0, 0.2, OptionType::Call).error().code,
              Errc::GreeksDomain);
    EXPECT_EQ(try_implied_vol(1e9, 1.0, 1.0, 1.0, 0.0, 0.0, OptionType::Call).error().code,
              Errc::IvOutsideBounds);
}

TEST(NoexceptApi, GreeksAndBlack76EqualThrowingApi) {
    for (const Q& x : grid()) {
        if (!greeks_domain(x)) continue;
        const GreeksResult a = *try_analytic_greeks(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type);
        const GreeksResult b = analytic_greeks(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type);
        EXPECT_TRUE(same_bits(a.price, b.price) && same_bits(a.delta_spot, b.delta_spot) &&
                    same_bits(a.delta_forward, b.delta_forward) && same_bits(a.gamma, b.gamma) &&
                    same_bits(a.vega, b.vega) && same_bits(a.theta, b.theta) &&
                    same_bits(a.rho_domestic, b.rho_domestic) &&
                    same_bits(a.rho_foreign, b.rho_foreign) && same_bits(a.vanna, b.vanna) &&
                    same_bits(a.volga, b.volga));
    }
    for (const Q& x : grid()) {
        EXPECT_TRUE(same_bits(*try_black76_price(x.S, x.K, x.T, x.rd, x.sigma, x.type),
                              black76_price(x.S, x.K, x.T, x.rd, x.sigma, x.type)));
    }
}

TEST(NoexceptApi, ImpliedVolEqualsThrowingApi) {
    for (const Q& x : grid()) {
        if (!greeks_domain(x)) continue;
        const double p = gk_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type);
        const auto a = try_implied_vol(p, x.S, x.K, x.T, x.rd, x.rf, x.type);
        bool threw = false;
        double b = 0.0;
        try {
            b = implied_vol(p, x.S, x.K, x.T, x.rd, x.rf, x.type);
        } catch (const std::exception&) {
            threw = true;
        }
        ASSERT_EQ(a.has_value(), !threw);
        if (a) {
            EXPECT_TRUE(same_bits(*a, b));
        }
    }
}

TEST(NoexceptApi, BinomialWorkspaceEqualsThrowingApiAndIsChecked) {
    std::vector<double> ws(binomial_workspace_size(200));
    for (const Q& x : grid()) {
        for (Exercise ex : {Exercise::European, Exercise::American}) {
            bool threw = false;
            double b = 0.0;
            try {
                b = binomial_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type, 200, ex);
            } catch (const std::exception&) {
                threw = true;
            }
            const auto a =
                try_binomial_price(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type, 200, ex, ws);
            ASSERT_EQ(a.has_value(), !threw);
            if (a) {
                EXPECT_TRUE(same_bits(*a, b));
            }
        }
    }
    std::vector<double> small(binomial_workspace_size(200) - 1);
    EXPECT_EQ(try_binomial_price(1.1, 1.1, 1, 0.03, 0.01, 0.2, OptionType::Put, 200,
                                 Exercise::American, small)
                  .error()
                  .code,
              Errc::WorkspaceTooSmall);
}

TEST(BatchApi, PricesEqualScalarApiBitForBit) {
    const auto g = grid();
    std::vector<double> S, K, T, rd, rf, sig, out(g.size());
    std::vector<OptionType> ty;
    for (const Q& x : g) {
        S.push_back(x.S);
        K.push_back(x.K);
        T.push_back(x.T);
        rd.push_back(x.rd);
        rf.push_back(x.rf);
        sig.push_back(x.sigma);
        ty.push_back(x.type);
    }
    ASSERT_TRUE(gk_price_batch({S, K, T, rd, rf, sig, ty}, out).has_value());
    for (std::size_t i = 0; i < g.size(); ++i) {
        EXPECT_TRUE(same_bits(out[i], gk_price(S[i], K[i], T[i], rd[i], rf[i], sig[i], ty[i])))
            << i;
    }
}

TEST(BatchApi, GreeksEqualScalarApiBitForBit) {
    std::vector<double> S, K, T, rd, rf, sig;
    std::vector<OptionType> ty;
    for (const Q& x : grid()) {
        if (!greeks_domain(x)) continue;
        S.push_back(x.S);
        K.push_back(x.K);
        T.push_back(x.T);
        rd.push_back(x.rd);
        rf.push_back(x.rf);
        sig.push_back(x.sigma);
        ty.push_back(x.type);
    }
    const std::size_t n = S.size();
    std::vector<double> o[10];
    for (auto& v : o) v.resize(n);
    const GkGreeksBatchOutputs out{o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9]};
    ASSERT_TRUE(gk_greeks_batch({S, K, T, rd, rf, sig, ty}, out).has_value());
    for (std::size_t i = 0; i < n; ++i) {
        const GreeksResult g = analytic_greeks(S[i], K[i], T[i], rd[i], rf[i], sig[i], ty[i]);
        const double want[10] = {g.price, g.delta_spot,   g.delta_forward, g.gamma, g.vega,
                                 g.theta, g.rho_domestic, g.rho_foreign,   g.vanna, g.volga};
        for (int f = 0; f < 10; ++f) EXPECT_TRUE(same_bits(o[f][i], want[f])) << i << " " << f;
    }
}

TEST(BatchApi, ReportsFirstBadIndexAndWritesNothing) {
    std::vector<double> S{1.1, 1.1, -1, 1.1}, K(4, 1.1), T(4, 1), rd(4, 0), rf(4, 0), sig(4, 0.2),
        out(4, 7.0);
    std::vector<OptionType> ty(4, OptionType::Call);
    S[3] = std::numeric_limits<double>::infinity();
    const auto res = gk_price_batch({S, K, T, rd, rf, sig, ty}, out);
    ASSERT_FALSE(res.has_value());
    EXPECT_EQ(res.error().index, 2U);
    EXPECT_EQ(res.error().error.code, Errc::SpotNonPositive);
    for (double v : out) EXPECT_EQ(v, 7.0);
    std::vector<double> short_out(3);
    EXPECT_EQ(gk_price_batch({S, K, T, rd, rf, sig, ty}, short_out).error().error.code,
              Errc::SizeMismatch);
}

// The fast batch is not bit-exact; the documented bound is
// |fast - exact| <= 1e-13 * max(1, S, K). Elements the fast formula does not
// cover (T = 0, sigma = 0) take the scalar path and must match bit for bit.
TEST(BatchApi, FastPricesWithinDocumentedBound) {
    std::vector<double> S, K, T, rd, rf, sig;
    std::vector<OptionType> ty;
    auto add = [&](double s, double k, double t, double d, double f, double v, OptionType o) {
        S.push_back(s);
        K.push_back(k);
        T.push_back(t);
        rd.push_back(d);
        rf.push_back(f);
        sig.push_back(v);
        ty.push_back(o);
    };
    for (const Q& x : grid()) add(x.S, x.K, x.T, x.rd, x.rf, x.sigma, x.type);
    std::mt19937_64 g(2026);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < 100000; ++i) {
        add(1.1 * std::exp(2.0 * u(g) - 1.0), 1.1, 1e-3 + 5.0 * u(g), 0.1 * u(g) - 0.02,
            0.1 * u(g) - 0.02, 0.01 + 2.0 * u(g),
            u(g) < 0.5 ? OptionType::Call : OptionType::Put);
    }
    const std::size_t n = S.size();
    std::vector<double> fast(n), exact(n);
    ASSERT_TRUE(gk_price_batch({S, K, T, rd, rf, sig, ty}, exact).has_value());
    ASSERT_TRUE(gk_price_batch_fast({S, K, T, rd, rf, sig, ty}, fast).has_value());
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        worst = std::max(worst, std::abs(fast[i] - exact[i]) / std::max({1.0, S[i], K[i]}));
        if (T[i] == 0.0 || sig[i] == 0.0) {
            EXPECT_TRUE(same_bits(fast[i], exact[i])) << i;
        }
    }
    EXPECT_LE(worst, 1e-13);
    RecordProperty("worst_scaled_abs_err", std::to_string(worst));
}

TEST(BatchApi, FastReportsTheSameErrorsAsExact) {
    std::vector<double> S{1.1, 1.1, -1, 1.1}, K(4, 1.1), T(4, 1), rd(4, 0), rf(4, 0), sig(4, 0.2),
        out(4, 7.0);
    std::vector<OptionType> ty(4, OptionType::Call);
    const auto res = gk_price_batch_fast({S, K, T, rd, rf, sig, ty}, out);
    ASSERT_FALSE(res.has_value());
    EXPECT_EQ(res.error().index, 2U);
    EXPECT_EQ(res.error().error.code, Errc::SpotNonPositive);
    for (double v : out) EXPECT_EQ(v, 7.0);
    std::vector<double> short_out(3);
    EXPECT_EQ(gk_price_batch_fast({S, K, T, rd, rf, sig, ty}, short_out).error().error.code,
              Errc::SizeMismatch);
    EXPECT_TRUE(gk_price_batch_fast({{}, {}, {}, {}, {}, {}, {}}, {}).has_value());
}

TEST(ThreadPoolMc, SingleStreamEqualsLegacyApi) {
    McWorkspace ws;
    ws.reserve(20001, true);
    for (bool av : {false, true}) {
        for (bool cv : {false, true}) {
            const MCResult legacy =
                mc_price(1.1, 1.15, 1, 0.04, 0.02, 0.25, OptionType::Call, 20001, Seed{7}, av, cv);
            const auto r = try_mc_price(1.1, 1.15, 1, 0.04, 0.02, 0.25, OptionType::Call, 20001,
                                        Seed{7}, av, cv, 1, ws, nullptr);
            ASSERT_TRUE(r.has_value());
            EXPECT_TRUE(same_bits(r->price, legacy.price) &&
                        same_bits(r->std_error, legacy.std_error));
            EXPECT_EQ(legacy.method, r->method);
            EXPECT_EQ(legacy.n_paths, r->n_paths);
        }
    }
}

TEST(ThreadPoolMc, ChunkedResultIndependentOfPoolAndRuns) {
    // For a fixed (seed, threads) the chunks and their streams are fixed, so
    // any pool (or none) must give the same bits, every run.
    McWorkspace ws;
    ws.reserve(20001, true);
    ThreadPool p0(0), p1(1), p3(3), p7(7);
    for (unsigned threads : {2U, 4U, 9U}) {
        for (bool av : {false, true}) {
            const auto ref = try_mc_price(1.1, 1.15, 1, 0.04, 0.02, 0.25, OptionType::Put, 20001,
                                          Seed{9}, av, true, threads, ws, nullptr);
            ASSERT_TRUE(ref.has_value());
            for (ThreadPool* p : {&p0, &p1, &p3, &p7}) {
                for (int rep = 0; rep < 3; ++rep) {
                    const auto r = try_mc_price(1.1, 1.15, 1, 0.04, 0.02, 0.25, OptionType::Put,
                                                20001, Seed{9}, av, true, threads, ws, p);
                    ASSERT_TRUE(r.has_value());
                    EXPECT_TRUE(same_bits(r->price, ref->price) &&
                                same_bits(r->std_error, ref->std_error));
                }
            }
            // Statistically consistent with the single stream: within 5
            // combined standard errors (both are unbiased estimators).
            const auto single = try_mc_price(1.1, 1.15, 1, 0.04, 0.02, 0.25, OptionType::Put, 20001,
                                             Seed{9}, av, true, 1, ws, nullptr);
            EXPECT_LT(std::fabs(single->price - ref->price),
                      5.0 * std::hypot(single->std_error, ref->std_error));
        }
    }
}

TEST(ThreadPoolMc, WorkspaceAndThreadErrors) {
    McWorkspace ws;
    ws.reserve(100, false);
    EXPECT_EQ(try_mc_price(1, 1, 1, 0, 0, 0.2, OptionType::Call, 1000, Seed{1}, true, false, 1, ws,
                           nullptr)
                  .error()
                  .code,
              Errc::WorkspaceTooSmall);
    EXPECT_EQ(
        try_mc_price(1, 1, 1, 0, 0, 0.2, OptionType::Call, 100, Seed{1}, true, true, 1, ws, nullptr)
            .error()
            .code,
        Errc::WorkspaceTooSmall);
    EXPECT_EQ(try_mc_price(1, 1, 1, 0, 0, 0.2, OptionType::Call, 100, Seed{1}, true, false, 0, ws,
                           nullptr)
                  .error()
                  .code,
              Errc::BadThreads);
}

TEST(ThreadPool, RunsEveryTaskOnceUnderRepeatedUse) {
    ThreadPool pool(5);
    std::vector<int> hits(64, 0);
    for (int rep = 0; rep < 500; ++rep) {
        auto f = [&](unsigned t) noexcept { hits[t] += 1; };
        pool.for_each_task(64, f);
    }
    for (int h : hits) EXPECT_EQ(h, 500);
}
