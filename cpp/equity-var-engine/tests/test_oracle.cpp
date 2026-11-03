// W5: the C++ engine against the independent oracle.
//
// Expected values come from oracle_vectors.hpp, generated from
// python/equity/03-var-es-engine/tests/golden/var_oracle.py (mpmath at 50
// digits and exact rationals; it imports nothing from eq_var or fx_var):
//
//   cd python/equity/03-var-es-engine
//   python tests/golden/var_oracle.py           # -> oracle_vectors.json (both projects)
//   python tests/golden/emit_oracle_consts.py   # -> this header and the Rust module
//
// Closed forms must agree to 1e-10 relative; the historical VaR order
// statistic exactly; Monte Carlo VaR of a Gaussian book within 3 standard
// errors of the exact quantile.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "eqvar/backtest.hpp"
#include "eqvar/expected_shortfall.hpp"
#include "eqvar/historical.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/monte_carlo.hpp"
#include "eqvar/parametric.hpp"
#include "oracle_vectors.hpp"

using namespace eqvar;

namespace {

constexpr double kRel = 1e-10;

void close(double actual, double expected, const std::string& what, double rel = kRel) {
    ASSERT_TRUE(std::isfinite(actual)) << what;
    EXPECT_LE(std::fabs(actual - expected), rel * std::fabs(expected))
        << what << ": engine " << actual << " oracle " << expected << " rel "
        << std::fabs(actual / expected - 1.0);
}

std::vector<double> conv_sample() {
    std::vector<double> p(250);
    for (int i = 0; i < 250; ++i) {
        const double x = 1000.0 * std::sin(0.37 * i + 0.1) * (1.0 + 0.5 * std::cos(0.11 * i));
        p[i] = std::floor(x * 100.0 + 0.5) / 100.0;
    }
    return p;
}

constexpr double kSig = 12345.678, kMean = -37.5;  // oracle inputs.sigma_mean_odd

struct AlphaRow {
    double a;
    double n_var_unit, n_es_unit, n_var_odd, n_es_odd;
};

struct TRow {
    double a, df;
    double var_unit, es_unit, var_odd, es_odd;
};

#define QP_T(A, AT, DF)                                                       \
    TRow {                                                                    \
        A, DF, oracle::t##DF##_var_unit_a##AT, oracle::t##DF##_es_unit_a##AT, \
            oracle::t##DF##_var_odd_a##AT, oracle::t##DF##_es_odd_a##AT       \
    }
#define QP_T_ALL(A, AT) \
    QP_T(A, AT, 3), QP_T(A, AT, 4), QP_T(A, AT, 6), QP_T(A, AT, 10), QP_T(A, AT, 30)

}  // namespace

TEST(Oracle, NormalVarEs) {
    const AlphaRow rows[] = {
        {0.001, oracle::normal_var_unit_a0001, oracle::normal_es_unit_a0001,
         oracle::normal_var_odd_a0001, oracle::normal_es_odd_a0001},
        {0.01, oracle::normal_var_unit_a001, oracle::normal_es_unit_a001,
         oracle::normal_var_odd_a001, oracle::normal_es_odd_a001},
        {0.025, oracle::normal_var_unit_a0025, oracle::normal_es_unit_a0025,
         oracle::normal_var_odd_a0025, oracle::normal_es_odd_a0025},
        {0.05, oracle::normal_var_unit_a005, oracle::normal_es_unit_a005,
         oracle::normal_var_odd_a005, oracle::normal_es_odd_a005},
    };
    const Matrix one(1, 1, std::vector<double>{1.0});
    const std::vector<double> w1{1.0}, ws{kSig};
    for (const auto& r : rows) {
        const std::string a = std::to_string(r.a);
        close(parametric_var(w1, one, r.a), r.n_var_unit, "N VaR unit " + a);
        close(parametric_var(ws, one, r.a, Dist::Normal, 6.0, kMean), r.n_var_odd,
              "N VaR odd " + a);
        close(normal_es(1.0, r.a), r.n_es_unit, "N ES unit " + a);
        close(normal_es(kSig, r.a, kMean), r.n_es_odd, "N ES odd " + a);
    }
}

TEST(Oracle, StudentTVarEs) {
    const TRow rows[] = {QP_T_ALL(0.001, 0001), QP_T_ALL(0.01, 001), QP_T_ALL(0.025, 0025),
                         QP_T_ALL(0.05, 005)};
    const Matrix one(1, 1, std::vector<double>{1.0});
    const std::vector<double> w1{1.0}, ws{kSig};
    for (const auto& r : rows) {
        const std::string tag = "a=" + std::to_string(r.a) + " df=" + std::to_string(r.df);
        close(parametric_var(w1, one, r.a, Dist::StudentT, r.df), r.var_unit, "t VaR unit " + tag);
        close(parametric_var(ws, one, r.a, Dist::StudentT, r.df, kMean), r.var_odd,
              "t VaR odd " + tag);
        close(student_t_es(1.0, r.a, r.df), r.es_unit, "t ES unit " + tag);
        close(student_t_es(kSig, r.a, r.df, kMean), r.es_odd, "t ES odd " + tag);
    }
}

TEST(Oracle, CornishFisher) {
    struct Row {
        double s, k, a, valid, var;
    };
    const Row rows[] = {
        {-0.5, 1.0, 0.01, oracle::cf_sm05_k1_a001_valid, oracle::cf_sm05_k1_a001_var_odd},
        {0.122, -0.427, 0.01, oracle::cf_s0122_km0427_a001_valid,
         oracle::cf_s0122_km0427_a001_var_odd},
        {0.2, 0.5, 0.025, oracle::cf_s02_k05_a0025_valid, oracle::cf_s02_k05_a0025_var_odd},
        {-0.3, 2.0, 0.05, oracle::cf_sm03_k2_a005_valid, oracle::cf_sm03_k2_a005_var_odd},
        {0.0, 0.0, 0.01, oracle::cf_s0_k0_a001_valid, oracle::cf_s0_k0_a001_var_odd},
    };
    for (const auto& r : rows) {
        EXPECT_EQ(cornish_fisher_valid(r.s, r.k, r.a), r.valid == 1.0);
        close(cornish_fisher_var(kSig, r.a, r.s, r.k, kMean), r.var,
              "CF " + std::to_string(r.s) + " " + std::to_string(r.k));
    }
}

TEST(Oracle, CaseBParametric) {
    const std::vector<double> w{1.0e6, -5.0e5, 2.0e5};
    const double v[3] = {0.01, 0.015, 0.02};
    const double c[3][3] = {{1.0, 0.5, 0.25}, {0.5, 1.0, 0.3}, {0.25, 0.3, 1.0}};
    Matrix cov(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) cov(i, j) = v[i] * c[i][j] * v[j];
    const double sigma = portfolio_sigma(w, cov);
    close(sigma, oracle::b_sigma, "B sigma");
    close(parametric_var(w, cov, 0.01), oracle::b_normal_var_a001, "B N VaR 1%");
    close(parametric_var(w, cov, 0.025), oracle::b_normal_var_a0025, "B N VaR 2.5%");
    close(parametric_var(w, cov, 0.01, Dist::Normal, 6.0, 0.0, 10), oracle::b_normal_var_10d_a001,
          "B N VaR 1% 10d");
    close(parametric_var(w, cov, 0.025, Dist::Normal, 6.0, 0.0, 10), oracle::b_normal_var_10d_a0025,
          "B N VaR 2.5% 10d");
    close(normal_es(sigma, 0.01), oracle::b_normal_es_a001, "B N ES 1%");
    close(normal_es(sigma, 0.025), oracle::b_normal_es_a0025, "B N ES 2.5%");
    close(parametric_var(w, cov, 0.01, Dist::StudentT, 5.0), oracle::b_t5_var_a001, "B t5 VaR 1%");
    close(parametric_var(w, cov, 0.025, Dist::StudentT, 5.0), oracle::b_t5_var_a0025,
          "B t5 VaR 2.5%");
    close(student_t_es(sigma, 0.01, 5.0), oracle::b_t5_es_a001, "B t5 ES 1%");
    close(student_t_es(sigma, 0.025, 5.0), oracle::b_t5_es_a0025, "B t5 ES 2.5%");
}

TEST(Oracle, HistoricalKthWorstExact) {
    const std::vector<double> p = conv_sample();
    struct Row {
        double a, var, es;
    };
    const Row rows[] = {{0.004, oracle::hist_kth_var_a0004, oracle::hist_kth_es_a0004},
                        {0.01, oracle::hist_kth_var_a001, oracle::hist_kth_es_a001},
                        {0.025, oracle::hist_kth_var_a0025, oracle::hist_kth_es_a0025},
                        {0.05, oracle::hist_kth_var_a005, oracle::hist_kth_es_a005},
                        {0.1, oracle::hist_kth_var_a01, oracle::hist_kth_es_a01}};
    for (const auto& r : rows) {
        const TailRisk t = var_es_order_statistic(p, r.a);
        EXPECT_EQ(t.var, r.var) << r.a;  // an order statistic: exact
        EXPECT_EQ(historical_var(p, r.a), r.var) << r.a;
        close(t.es, r.es, "hist ES " + std::to_string(r.a), 1e-12);
    }
    EXPECT_EQ(oracle::hist_kth_rank_a001, 3.0);  // MAR99: 3rd worst of 250
}

TEST(Oracle, KupiecChristoffersenBasel) {
    struct K {
        int n, x;
        double a, lr, p;
    };
#define QP_K(N, X, A, AT) \
    K { N, X, A, oracle::kupiec_lr_n##N##_x##X##_a##AT, oracle::kupiec_p_n##N##_x##X##_a##AT }
    const K ks[] = {
        QP_K(250, 0, 0.01, 001),    QP_K(250, 1, 0.01, 001),  QP_K(250, 2, 0.01, 001),
        QP_K(250, 3, 0.01, 001),    QP_K(250, 4, 0.01, 001),  QP_K(250, 5, 0.01, 001),
        QP_K(250, 6, 0.01, 001),    QP_K(250, 7, 0.01, 001),  QP_K(250, 8, 0.01, 001),
        QP_K(250, 9, 0.01, 001),    QP_K(250, 10, 0.01, 001), QP_K(250, 11, 0.01, 001),
        QP_K(250, 12, 0.01, 001),   QP_K(500, 9, 0.01, 001),  QP_K(1000, 15, 0.01, 001),
        QP_K(250, 12, 0.025, 0025), QP_K(250, 5, 0.05, 005),  QP_K(1000, 50, 0.05, 005)};
#undef QP_K
    for (const auto& k : ks) {
        const KupiecResult r =
            kupiec_pof(KupiecInput{.n_obs = k.n, .n_exceptions = k.x, .alpha = k.a});
        const std::string tag = std::to_string(k.n) + "/" + std::to_string(k.x);
        if (k.lr < 1e-300) {
            EXPECT_LT(std::fabs(r.lr), 1e-12) << tag;
        } else {
            close(r.lr, k.lr, "kupiec LR " + tag);
        }
        close(r.pvalue, k.p, "kupiec p " + tag);
    }
    struct C {
        int mod, rem;
        double ind_lr, ind_p, cc_lr, cc_p;
    };
    const C cs[] = {{40, 0, oracle::chr_eq_ind_lr, oracle::chr_eq_ind_p, oracle::chr_eq_cc_lr,
                     oracle::chr_eq_cc_p},
                    {37, 5, oracle::chr_fx_ind_lr, oracle::chr_fx_ind_p, oracle::chr_fx_cc_lr,
                     oracle::chr_fx_cc_p}};
    for (const auto& c : cs) {
        std::vector<std::uint8_t> ex(250, 0);
        for (int t = 0; t < 250; ++t) ex[t] = (t % c.mod == c.rem || t == 100 || t == 101) ? 1 : 0;
        const auto ind = christoffersen_independence(ex);
        const auto cc = christoffersen_cc(ex, 0.01);
        close(ind.lr, c.ind_lr, "chr ind LR");
        close(ind.pvalue, c.ind_p, "chr ind p");
        close(cc.lr, c.cc_lr, "chr cc LR");
        close(cc.pvalue, c.cc_p, "chr cc p");
    }
    struct B {
        int n, x;
        double a, cdf;
    };
#define QP_B(N, X, A, AT) \
    B { N, X, A, oracle::basel_cdf_n##N##_x##X##_a##AT }
    const B bs[] = {
        QP_B(250, 0, 0.01, 001),  QP_B(250, 1, 0.01, 001),   QP_B(250, 2, 0.01, 001),
        QP_B(250, 3, 0.01, 001),  QP_B(250, 4, 0.01, 001),   QP_B(250, 5, 0.01, 001),
        QP_B(250, 6, 0.01, 001),  QP_B(250, 7, 0.01, 001),   QP_B(250, 8, 0.01, 001),
        QP_B(250, 9, 0.01, 001),  QP_B(250, 10, 0.01, 001),  QP_B(500, 8, 0.01, 001),
        QP_B(500, 12, 0.01, 001), QP_B(1000, 15, 0.01, 001), QP_B(250, 12, 0.025, 0025)};
#undef QP_B
    for (const auto& b : bs) {
        const BaselResult r =
            basel_traffic_light(BaselInput{.n_obs = b.n, .n_exceptions = b.x, .alpha = b.a});
        close(r.cumulative_prob, b.cdf, "basel " + std::to_string(b.n) + "/" + std::to_string(b.x));
    }
}

// Monte Carlo VaR of the Gaussian case-B book converges to the exact
// quantile: |MC - exact| <= 3 SE (bootstrap SE of the same run).
TEST(Oracle, MonteCarloGaussianBookWithin3Se) {
    const std::vector<double> w{1.0e6, -5.0e5, 2.0e5};
    const double v[3] = {0.01, 0.015, 0.02};
    const double c[3][3] = {{1.0, 0.5, 0.25}, {0.5, 1.0, 0.3}, {0.25, 0.3, 1.0}};
    Matrix cov(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) cov(i, j) = v[i] * c[i][j] * v[j];
    for (std::uint64_t seed : {1ULL, 2ULL, 3ULL}) {
        const MonteCarloResult r =
            monte_carlo_var(w, cov, 0.01, 200000, Dist::Normal, 6.0, seed, 200);
        EXPECT_LE(std::fabs(r.var - oracle::b_normal_var_a001), 3.0 * r.var_se)
            << "seed " << seed << " mc " << r.var << " se " << r.var_se;
        const MonteCarloResult t =
            monte_carlo_var(w, cov, 0.01, 200000, Dist::StudentT, 5.0, seed, 200);
        EXPECT_LE(std::fabs(t.var - oracle::b_t5_var_a001), 3.0 * t.var_se)
            << "t5 seed " << seed << " mc " << t.var << " se " << t.var_se;
    }
}
