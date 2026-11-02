// Regression tests: overflow-safe edges, Brent failure, PA fold bracket,
// strong-typed Monte Carlo parameters and the shared implied-vol core.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "../src/brent.hpp"
#include "fxopt/binomial.hpp"
#include "fxopt/deltas.hpp"
#include "fxopt/forwards.hpp"
#include "fxopt/garman_kohlhagen.hpp"
#include "fxopt/implied_vol.hpp"
#include "fxopt/iv_core.hpp"
#include "fxopt/monte_carlo.hpp"

using namespace fxopt;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

// mc_price(..., 100000, true, true, 42) used to compile with seed = 1,
// antithetic = true and control_variate = (42 -> true).
template <class... A>
concept CanCallMc = requires(A... a) { mc_price(a...); };
static_assert(
    !CanCallMc<double, double, double, double, double, double, OptionType, int, bool, bool, int>);
static_assert(
    !CanCallMc<double, double, double, double, double, double, OptionType, int, std::uint64_t>);
static_assert(
    CanCallMc<double, double, double, double, double, double, OptionType, int, Seed, bool, bool>);

}  // namespace

TEST(NumericalSafety, SigmaZeroWithOverflowingForwardIsFinite) {
    EXPECT_EQ(gk_price(1.0, 1.0, 1000.0, 0.8, 0.0, 0.0, OptionType::Call), 1.0);
    EXPECT_EQ(gk_price(1.0, 1.0, 1000.0, 0.8, 0.0, 0.0, OptionType::Put), 0.0);
    EXPECT_EQ(
        binomial_price(1.0, 1.0, 1000.0, 0.8, 0.0, 0.0, OptionType::Call, 100, Exercise::American),
        1.0);
}

TEST(NumericalSafety, NonFiniteResultsThrow) {
    EXPECT_THROW(gk_price(1.0, 1.0, 1000.0, -0.8, 0.0, 0.2, OptionType::Put),
                 std::invalid_argument);
    EXPECT_THROW(cip_forward(1.0, 1000.0, 0.8, 0.0), std::invalid_argument);
    EXPECT_THROW(forward_points(1.1, 1.0, 0.03, 0.01, std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
    EXPECT_THROW(spot_to_forward_delta(std::numeric_limits<double>::quiet_NaN(), 1.0, 0.01),
                 std::invalid_argument);
    EXPECT_THROW(forward_to_spot_delta(std::numeric_limits<double>::infinity(), 1.0, 0.01),
                 std::invalid_argument);
    EXPECT_THROW(mc_price(1.0, 1.0, 1000.0, 0.8, 0.0, 0.2, OptionType::Call, 1000, Seed{1}),
                 std::invalid_argument);
}

TEST(NumericalSafety, BrentNonConvergenceThrows) {
    // A steep sign change that 3 iterations cannot resolve.
    const auto f = [](double x) { return std::tanh(1000.0 * (x - 0.123456789)); };
    EXPECT_THROW(detail::brentq(f, -1e6, 1e6, 1e-14, 3), std::runtime_error);
    EXPECT_NEAR(detail::brentq(f, -1e6, 1e6, 1e-14, 400), 0.123456789, 1e-12);
}

TEST(NumericalSafety, PaFoldBracketValidForAnyTotalVol) {
    // The fixed [-20, 20] bracket failed above v = 20.
    for (double v : {1e-6, 0.1, 1.0, 19.0, 21.0, 36.0, 40.0, 100.0}) {
        const auto [lo, hi] = pa_fold_bracket(v);
        const auto h = [v](double x) {
            return log_norm_cdf(x) + std::log(v) + 0.5 * x * x + 0.9189385332046728;
        };
        EXPECT_LT(h(lo), 0.0) << v;
        EXPECT_GT(h(hi), 0.0) << v;
    }
    // A PA call strike solve at sigma sqrt(T) = 25 succeeds.
    const double k = strike_from_delta(1e-3, 1.1, 1.0, 0.03, 0.01, 25.0, OptionType::Call,
                                       DeltaConvention::ForwardPa);
    EXPECT_TRUE(std::isfinite(k));
}

TEST(NumericalSafety, McNeedsIndependentSamples) {
    EXPECT_THROW(
        mc_price(1.1, 1.1, 1.0, 0.03, 0.01, 0.1, OptionType::Call, 2, Seed{1}, true, false),
        std::invalid_argument);
    EXPECT_THROW(mc_price(1.1, 1.1, 1.0, 0.03, 0.01, 0.1, OptionType::Call, 4, Seed{1}, true, true),
                 std::invalid_argument);
    const McConfig cfg{6, Seed{1}, true, true};
    EXPECT_GT(mc_price(1.1, 1.1, 1.0, 0.03, 0.01, 0.1, OptionType::Call, cfg).std_error, 0.0);
}

TEST(NumericalSafety, AntitheticPlusCvBeatsCvAloneAcrossSeeds) {
    // Fitting beta on raw draws and pairing afterwards gave a 1M-path SE of
    // 2.95e-5 here (README); the pair-average fit gives about 9.4e-6.
    for (std::uint64_t seed = 1; seed <= 7; ++seed) {
        const auto both = mc_price(1.10, 1.10, 1.0, 0.04, 0.02, 0.10, OptionType::Call, 100000,
                                   Seed{seed}, true, true);
        const auto cv = mc_price(1.10, 1.10, 1.0, 0.04, 0.02, 0.10, OptionType::Call, 100000,
                                 Seed{seed}, false, true);
        EXPECT_LE(both.std_error, cv.std_error) << seed;
        EXPECT_LT(both.std_error, 0.6 * 1.39e-4) << seed;
        const double exact = gk_price(1.10, 1.10, 1.0, 0.04, 0.02, 0.10, OptionType::Call);
        EXPECT_LT(std::fabs(both.price - exact), 4.0 * both.std_error);
    }
}

TEST(ImpliedVolCore, NoPrematureExitOnTinyOtmPrice) {
    // The old solver stopped on |price residual| < 1e-14 and returned this
    // vol only to about 2.5e-7.
    for (double sigma : {0.10, 0.14}) {
        const double put = gk_price(100.0, 90.0, 0.02, 0.03, 0.0, sigma, OptionType::Put);
        EXPECT_NEAR(implied_vol(put, 100.0, 90.0, 0.02, 0.03, 0.0, OptionType::Put), sigma, 1e-12);
    }
}

TEST(ImpliedVolCore, BoundsContract) {
    const double S = 1.1, K = 1.0, T = 0.5, rd = 0.04, rf = 0.02;
    const double lower = gk_price(S, K, T, rd, rf, 0.0, OptionType::Call);
    EXPECT_EQ(implied_vol(lower, S, K, T, rd, rf, OptionType::Call), 0.0);
    EXPECT_THROW(implied_vol(S * std::exp(-rf * T), S, K, T, rd, rf, OptionType::Call),
                 std::invalid_argument);
    EXPECT_THROW(implied_vol(lower - 1e-9, S, K, T, rd, rf, OptionType::Call),
                 std::invalid_argument);
}

TEST(ImpliedVolCore, AtMostTenEvaluations) {
    const double df = 0.97;
    for (double lk : {-3.0, -1.0, -0.3, -0.05, 0.0, 0.05, 0.3, 1.0, 3.0}) {
        for (double t : {1e-4, 0.02, 0.25, 1.0, 10.0}) {
            for (double sigma : {0.01, 0.1, 0.5, 2.0}) {
                const double k = std::exp(lk);
                const bool is_call = k >= 1.0;
                const double r = -std::log(df) / t;
                const double price =
                    gk_price(1.0, k, t, r, r, sigma, is_call ? OptionType::Call : OptionType::Put);
                if (!(price > 1e-300)) continue;
                int evals = 0;
                iv_core::black_implied_vol(price, 1.0, k, t, df, is_call, evals);
                EXPECT_LE(evals, 10) << lk << " " << t << " " << sigma;
            }
        }
    }
    EXPECT_NEAR(gk_price(1.0, 1.0, 1.0, 0.0, 0.0, 0.2, OptionType::Call), 0.0796557, 1e-6);
    (void)kEps;
}
