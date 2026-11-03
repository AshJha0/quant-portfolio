// Historical VaR rule: k-th worst and type-7 interpolated against the mpmath
// oracle (python/equity/03-var-es-engine/tests/golden/var_oracle.py).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "eqvar/historical.hpp"
#include "oracle_vectors.hpp"

using namespace eqvar;

namespace {
std::vector<double> conv_sample() {
    std::vector<double> p(250);
    for (int i = 0; i < 250; ++i) {
        const double x = 1000.0 * std::sin(0.37 * i + 0.1) * (1.0 + 0.5 * std::cos(0.11 * i));
        p[i] = std::floor(x * 100.0 + 0.5) / 100.0;
    }
    return p;
}
struct Row {
    double a, kth, interp;
};
const Row kRows[] = {{0.004, oracle::hist_kth_var_a0004, oracle::hist_interp_var_a0004},
                     {0.01, oracle::hist_kth_var_a001, oracle::hist_interp_var_a001},
                     {0.025, oracle::hist_kth_var_a0025, oracle::hist_interp_var_a0025},
                     {0.05, oracle::hist_kth_var_a005, oracle::hist_interp_var_a005},
                     {0.1, oracle::hist_kth_var_a01, oracle::hist_interp_var_a01}};
}  // namespace

TEST(HistRule, BothRulesMatchOracle) {
    const std::vector<double> p = conv_sample();
    for (const auto& r : kRows) {
        EXPECT_EQ(historical_var(p, r.a, HistRule::kKthWorst), r.kth) << r.a;
        const double v = historical_var(p, r.a, HistRule::kInterpolated);
        EXPECT_LE(std::abs(v - r.interp), 1e-12 * std::abs(r.interp)) << r.a << " " << v;
    }
}

TEST(HistRule, Mar99InterpolatesThirdAndFourthWorst) {
    std::vector<double> l = conv_sample();
    for (double& x : l) x = -x;
    std::sort(l.begin(), l.end(), std::greater<>());
    const double v = interpolated_var(conv_sample(), 0.01);
    EXPECT_NEAR(v, l[2] + 0.49 * (l[3] - l[2]), 1e-12 * l[2]);
}

TEST(HistRule, RejectsEmptyAndNonFinite) {
    EXPECT_THROW((void)interpolated_var(std::vector<double>{}, 0.01), std::invalid_argument);
    EXPECT_THROW((void)interpolated_var(std::vector<double>{1.0, NAN}, 0.01),
                 std::invalid_argument);
}
