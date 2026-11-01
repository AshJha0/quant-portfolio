// The noexcept kernel API, the batch pricer and the thread pool: each must
// reproduce the throwing scalar API exactly (bit for bit), and MC results
// must not depend on which pool runs the chunks.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "eqopt/batch.hpp"
#include "eqopt/binomial.hpp"
#include "eqopt/black76.hpp"
#include "eqopt/black_scholes.hpp"
#include "eqopt/greeks.hpp"
#include "eqopt/implied_vol.hpp"
#include "eqopt/monte_carlo.hpp"
#include "eqopt/thread_pool.hpp"

using namespace eqopt;

namespace {

// Kernel entry points are noexcept: a failure is a value, never a throw.
static_assert(noexcept(try_bs_price(1.0, 1.0, 1.0, 0.0, 0.2)));
static_assert(noexcept(try_bs_greeks(1.0, 1.0, 1.0, 0.0, 0.2)));
static_assert(noexcept(try_d1_d2(1.0, 1.0, 1.0, 0.0, 0.2)));
static_assert(noexcept(try_black76_price(1.0, 1.0, 1.0, 0.0, 0.2)));
static_assert(noexcept(try_black76_greeks(1.0, 1.0, 1.0, 0.0, 0.2)));
static_assert(noexcept(try_implied_vol(0.1, 1.0, 1.0, 1.0, 0.0)));
static_assert(noexcept(try_crr_price(1.0, 1.0, 1.0, 0.0, 0.2, 0.0, OptionType::Call,
                                     ExerciseStyle::American, 10, std::span<double>{})));
static_assert(noexcept(bs_price_batch(std::declval<const BsBatchInputs&>(),
                                      std::span<double>{})));
static_assert(noexcept(bs_greeks_batch(std::declval<const BsBatchInputs&>(),
                                       std::declval<const BsGreeksBatchOutputs&>())));
static_assert(noexcept(try_mc_price(1.0, 1.0, 1.0, 0.0, 0.2, 0.0, OptionType::Call, 100,
                                    true, true, 1, 1, std::declval<McWorkspace&>(),
                                    nullptr)));
static_assert(std::is_trivially_copyable_v<Error>);

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

struct Q {
    double S, K, T, r, sigma, q;
    OptionType type;
};

std::vector<Q> grid() {
    std::vector<Q> v;
    for (double S : {0.0, 50.0, 99.5, 100.0, 180.0})
        for (double K : {0.0, 80.0, 100.0, 140.0})
            for (double T : {0.0, 1.0 / 365.0, 0.5, 3.0})
                for (double sigma : {0.0, 0.05, 0.3, 1.5})
                    for (double r : {-0.01, 0.04})
                        for (OptionType t : {OptionType::Call, OptionType::Put})
                            v.push_back({S, K, T, r, sigma, 0.015, t});
    return v;
}

}  // namespace

TEST(NoexceptApi, BsPriceEqualsThrowingApiBitForBit) {
    for (const Q& x : grid()) {
        const auto r = try_bs_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type);
        ASSERT_TRUE(r.has_value());
        EXPECT_TRUE(same_bits(*r, bs_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type)));
    }
}

TEST(NoexceptApi, ErrorsCarryTheThrowingMessage) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto r = try_bs_price(-1.0, 100.0, 1.0, 0.0, 0.2);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, Errc::NegativeOrNonFinite);
    try {
        (void)bs_price(-1.0, 100.0, 1.0, 0.0, 0.2);
        FAIL() << "expected throw";
    } catch (const std::invalid_argument& e) {
        EXPECT_EQ(message(r.error()), e.what());
    }
    EXPECT_EQ(try_bs_price(100.0, 100.0, 1.0, nan, 0.2).error().code, Errc::NonFiniteRate);
    EXPECT_EQ(try_bs_greeks(100.0, 100.0, 0.0, 0.0, 0.2).error().code,
              Errc::NotStrictlyPositive);
    EXPECT_EQ(try_implied_vol(1e9, 100.0, 100.0, 1.0, 0.0).error().code,
              Errc::AboveUpperBound);
    EXPECT_EQ(try_bs_price(1e300, 1.0, 10.0, 0.0, 0.2, -80.0).error().code,
              Errc::NonFiniteResult);
}

TEST(NoexceptApi, GreeksAndBlack76EqualThrowingApi) {
    for (const Q& x : grid()) {
        if (x.S <= 0 || x.K <= 0 || x.T <= 0 || x.sigma <= 0) continue;
        const BSGreeks a = *try_bs_greeks(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type);
        const BSGreeks b = bs_greeks(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type);
        EXPECT_TRUE(same_bits(a.price, b.price) && same_bits(a.delta, b.delta) &&
                    same_bits(a.gamma, b.gamma) && same_bits(a.vega, b.vega) &&
                    same_bits(a.theta, b.theta) && same_bits(a.rho, b.rho) &&
                    same_bits(a.vanna, b.vanna) && same_bits(a.volga, b.volga));
        const Black76Greeks c = *try_black76_greeks(x.S, x.K, x.T, x.r, x.sigma, x.type);
        const Black76Greeks d = black76_greeks(x.S, x.K, x.T, x.r, x.sigma, x.type);
        EXPECT_TRUE(same_bits(c.price, d.price) && same_bits(c.theta, d.theta));
        EXPECT_TRUE(same_bits(*try_black76_price(x.S, x.K, x.T, x.r, x.sigma, x.type),
                              black76_price(x.S, x.K, x.T, x.r, x.sigma, x.type)));
    }
}

TEST(NoexceptApi, ImpliedVolEqualsThrowingApi) {
    for (const Q& x : grid()) {
        if (x.S <= 0 || x.K <= 0 || x.T <= 0 || x.sigma <= 0) continue;
        const double p = bs_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type);
        const auto a = try_implied_vol(p, x.S, x.K, x.T, x.r, x.q, x.type);
        bool threw = false;
        double b = 0.0;
        try {
            b = implied_vol(p, x.S, x.K, x.T, x.r, x.q, x.type);
        } catch (const std::exception&) {
            threw = true;
        }
        ASSERT_EQ(a.has_value(), !threw);
        if (a) {
            EXPECT_TRUE(same_bits(*a, b));
        }
    }
}

TEST(NoexceptApi, CrrWorkspaceEqualsThrowingApiAndIsChecked) {
    std::vector<double> ws(crr_workspace_size(200));
    for (const Q& x : grid()) {
        for (ExerciseStyle ex : {ExerciseStyle::European, ExerciseStyle::American}) {
            const auto a = try_crr_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type, ex, 200, ws);
            ASSERT_TRUE(a.has_value());
            EXPECT_TRUE(same_bits(*a, crr_price(x.S, x.K, x.T, x.r, x.sigma, x.q, x.type, ex,
                                                200)));
        }
    }
    std::vector<double> small(crr_workspace_size(200) - 1);
    EXPECT_EQ(try_crr_price(100, 100, 1, 0.03, 0.2, 0, OptionType::Put,
                            ExerciseStyle::American, 200, small)
                  .error()
                  .code,
              Errc::WorkspaceTooSmall);
}

TEST(BatchApi, PricesEqualScalarApiBitForBit) {
    const auto g = grid();
    std::vector<double> S, K, T, r, sig, q, out(g.size());
    std::vector<OptionType> ty;
    for (const Q& x : g) {
        S.push_back(x.S); K.push_back(x.K); T.push_back(x.T); r.push_back(x.r);
        sig.push_back(x.sigma); q.push_back(x.q); ty.push_back(x.type);
    }
    ASSERT_TRUE(bs_price_batch({S, K, T, r, sig, q, ty}, out).has_value());
    for (std::size_t i = 0; i < g.size(); ++i) {
        EXPECT_TRUE(same_bits(out[i], bs_price(S[i], K[i], T[i], r[i], sig[i], q[i], ty[i])))
            << i;
    }
}

TEST(BatchApi, GreeksEqualScalarApiBitForBit) {
    std::vector<double> S, K, T, r, sig, q;
    std::vector<OptionType> ty;
    for (const Q& x : grid()) {
        if (x.S <= 0 || x.K <= 0 || x.T <= 0 || x.sigma <= 0) continue;
        S.push_back(x.S); K.push_back(x.K); T.push_back(x.T); r.push_back(x.r);
        sig.push_back(x.sigma); q.push_back(x.q); ty.push_back(x.type);
    }
    const std::size_t n = S.size();
    std::vector<double> o[8];
    for (auto& v : o) v.resize(n);
    const BsGreeksBatchOutputs out{o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]};
    ASSERT_TRUE(bs_greeks_batch({S, K, T, r, sig, q, ty}, out).has_value());
    for (std::size_t i = 0; i < n; ++i) {
        const BSGreeks g = bs_greeks(S[i], K[i], T[i], r[i], sig[i], q[i], ty[i]);
        const double want[8] = {g.price, g.delta, g.gamma, g.vega,
                                g.theta, g.rho,   g.vanna, g.volga};
        for (int f = 0; f < 8; ++f) EXPECT_TRUE(same_bits(o[f][i], want[f])) << i << " " << f;
    }
}

TEST(BatchApi, ReportsFirstBadIndexAndWritesNothing) {
    std::vector<double> S{100, 100, -1, 100}, K(4, 100), T(4, 1), r(4, 0), sig(4, 0.2),
        q(4, 0), out(4, 7.0);
    std::vector<OptionType> ty(4, OptionType::Call);
    S[3] = std::numeric_limits<double>::infinity();
    const auto res = bs_price_batch({S, K, T, r, sig, q, ty}, out);
    ASSERT_FALSE(res.has_value());
    EXPECT_EQ(res.error().index, 2U);
    EXPECT_EQ(res.error().error.code, Errc::NegativeOrNonFinite);
    for (double v : out) EXPECT_EQ(v, 7.0);
    std::vector<double> short_out(3);
    EXPECT_EQ(bs_price_batch({S, K, T, r, sig, q, ty}, short_out).error().error.code,
              Errc::SizeMismatch);
}

TEST(ThreadPoolMc, ResultIndependentOfPoolSizeAndEqualsLegacyApi) {
    // For a fixed (seed, threads) the chunks and their RNG streams are fixed,
    // so any pool (or none) must give the same bits, every run.
    McWorkspace ws;
    ws.reserve(20001, true);
    ThreadPool p0(0), p1(1), p3(3), p7(7);
    for (unsigned threads : {1U, 2U, 4U, 9U}) {
        for (bool av : {false, true}) {
            for (bool cv : {false, true}) {
                const MCResult legacy = mc_price(100, 105, 1, 0.04, 0.25, 0.015,
                                                 OptionType::Call, 20001, av, cv, 7, threads);
                for (ThreadPool* p : {static_cast<ThreadPool*>(nullptr), &p0, &p1, &p3, &p7}) {
                    for (int rep = 0; rep < 3; ++rep) {
                        const auto r = try_mc_price(100, 105, 1, 0.04, 0.25, 0.015,
                                                    OptionType::Call, 20001, av, cv, 7,
                                                    threads, ws, p);
                        ASSERT_TRUE(r.has_value());
                        EXPECT_TRUE(same_bits(r->price, legacy.price) &&
                                    same_bits(r->std_error, legacy.std_error));
                    }
                }
            }
        }
    }
}

TEST(ThreadPoolMc, WorkspaceTooSmallIsAnError) {
    McWorkspace ws;
    ws.reserve(100, false);
    EXPECT_EQ(try_mc_price(100, 100, 1, 0.0, 0.2, 0.0, OptionType::Call, 1000, true, false, 1,
                           1, ws, nullptr)
                  .error()
                  .code,
              Errc::WorkspaceTooSmall);
    EXPECT_EQ(try_mc_price(100, 100, 1, 0.0, 0.2, 0.0, OptionType::Call, 100, true, true, 1,
                           1, ws, nullptr)
                  .error()
                  .code,
              Errc::WorkspaceTooSmall);
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
