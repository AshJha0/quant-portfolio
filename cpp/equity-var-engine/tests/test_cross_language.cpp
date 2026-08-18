// Cross-language golden tests against the Python reference implementation.
//
// Provenance: every expected constant below was produced by the Python
// research stack (the conceptual twin of this engine):
//
//   cd /home/claude/quant-portfolio/python/equity/03-var-es-engine
//   PYTHONPATH=src python3 /tmp/gen_golden.py     (numpy 2.x, scipy 1.17.1)
//
// generated on 2026-08-18 from eq_var (historical_var / parametric_var /
// expected_shortfall / backtesting modules).  Inputs are closed-form
// (sin/cos-based) deterministic series so both languages regenerate them
// independently; agreement is required to 1e-9 relative (the residual is
// libm-vs-numpy ulp noise in sin/cos, far below any risk tolerance).
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "eqvar/backtest.hpp"
#include "eqvar/expected_shortfall.hpp"
#include "eqvar/historical.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/parametric.hpp"

using namespace eqvar;

namespace {

// Relative tolerance for cross-language agreement.
void expect_rel(double actual, double expected, const char* what) {
    EXPECT_NEAR(actual, expected, 1e-9 * std::abs(expected) + 1e-12) << what;
}

// Case A input: pnl[t] = 100 sin(3t + 1) + 0.5 t cos(t), t = 0..99.
std::vector<double> case_a_pnl() {
    std::vector<double> pnl(100);
    for (int t = 0; t < 100; ++t) {
        const double td = static_cast<double>(t);
        pnl[t] = 100.0 * std::sin(3.0 * td + 1.0) + 0.5 * td * std::cos(td);
    }
    return pnl;
}

// Case C input: r[t, j] = 0.01 sin(t + j) + 0.005 cos(2t - j), t=0..59, j=0..2.
Matrix case_c_returns() {
    Matrix r(60, 3);
    for (int t = 0; t < 60; ++t) {
        for (int j = 0; j < 3; ++j) {
            const double td = static_cast<double>(t), jd = static_cast<double>(j);
            r(t, j) = 0.01 * std::sin(td + jd) + 0.005 * std::cos(2.0 * td - jd);
        }
    }
    return r;
}

}  // namespace

TEST(CrossLanguage, CaseA_HistoricalFamily) {
    const std::vector<double> pnl = case_a_pnl();
    expect_rel(historical_var(pnl, 0.01), 1.224129222375264e+02, "A hist VaR 1%");
    expect_rel(historical_var(pnl, 0.05), 1.045522835374927e+02, "A hist VaR 5%");
    expect_rel(expected_shortfall(pnl, 0.05), 1.226373207703405e+02, "A ES 5%");
    expect_rel(expected_shortfall(pnl, 0.01), 1.417568107549531e+02, "A ES 1%");
    expect_rel(age_weighted_var(pnl, 0.05, 0.98), 1.082348601407293e+02, "A BRW VaR 5%");
    expect_rel(filtered_historical_var(pnl, 0.05, 0.94), 1.093777910164513e+02, "A FHS VaR 5%");
}

TEST(CrossLanguage, CaseA_BrwWeights) {
    const std::vector<double> w = brw_weights(5, 0.98);
    const std::vector<double> expected = {
        1.920016255921656e-01, 1.959200261144547e-01, 1.999183939943416e-01,
        2.039983612187159e-01, 2.081615930803223e-01,
    };
    ASSERT_EQ(w.size(), expected.size());
    for (std::size_t i = 0; i < w.size(); ++i) {
        expect_rel(w[i], expected[i], "A brw weight");
    }
}

TEST(CrossLanguage, CaseB_ParametricFamily) {
    // w = [1e6, -5e5, 2e5]; vols = [1%, 1.5%, 2%];
    // corr = [[1, .5, .25], [.5, 1, .3], [.25, .3, 1]].
    const std::vector<double> w = {1.0e6, -5.0e5, 2.0e5};
    const std::vector<double> vols = {0.01, 0.015, 0.02};
    const Matrix corr(3, 3, {1.0, 0.5, 0.25, 0.5, 1.0, 0.3, 0.25, 0.3, 1.0});
    const Matrix cov = covariance_from_vols(vols, corr);
    const double sigma = portfolio_sigma(w, cov);
    expect_rel(sigma, 9.962429422585637e+03, "B sigma");
    expect_rel(parametric_var(w, cov, 0.01), 2.317607650751402e+04, "B VaR normal 1%");
    expect_rel(parametric_var(w, cov, 0.01, Dist::StudentT, 6.0), 2.556337478743866e+04,
               "B VaR t6 1%");
    expect_rel(parametric_var(w, cov, 0.01, Dist::Normal, 6.0, 0.0, 10),
               7.328918899006478e+04, "B VaR normal 1% 10d");
    expect_rel(normal_es(sigma, 0.025), 2.329019532123021e+04, "B ES normal 2.5%");
    expect_rel(student_t_es(sigma, 0.025, 6.0), 2.648647588170394e+04, "B ES t6 2.5%");
    expect_rel(cornish_fisher_var(sigma, 0.01, -0.5, 1.0), 2.823062626871169e+04,
               "B CF VaR 1%");
}

TEST(CrossLanguage, CaseC_CovarianceEstimators) {
    const Matrix r = case_c_returns();
    const Matrix s = sample_covariance(r);
    const Matrix e = ewma_covariance(r, 0.94);
    expect_rel(s(0, 0), 6.197010057346647e-05, "C sample cov 00");
    expect_rel(s(0, 1), 3.367477490492125e-05, "C sample cov 01");
    expect_rel(s(1, 2), 3.643904950084309e-05, "C sample cov 12");
    expect_rel(e(0, 0), 6.085653794066718e-05, "C ewma cov 00");
    expect_rel(e(0, 1), 3.159039791408869e-05, "C ewma cov 01");
    expect_rel(e(2, 2), 6.695548359317170e-05, "C ewma cov 22");

    const std::vector<double> w = {2.0e6, -1.0e6, 5.0e5};
    expect_rel(parametric_var(w, s, 0.01), 2.403056180968714e+04, "C VaR normal 1%");
    expect_rel(parametric_var(w, s, 0.05, Dist::StudentT, 8.0), 1.663517215790393e+04,
               "C VaR t8 5%");
}

TEST(CrossLanguage, CaseC_BacktestStatistics) {
    const KupiecResult k7 = kupiec_pof(250, 7, 0.01);
    expect_rel(k7.lr, 5.496990447792683e+00, "C kupiec LR (250, 7)");
    expect_rel(k7.pvalue, 1.904923089052653e-02, "C kupiec p (250, 7)");
    const KupiecResult k0 = kupiec_pof(250, 0, 0.01);
    expect_rel(k0.lr, 5.025167926750726e+00, "C kupiec LR (250, 0)");
    expect_rel(k0.pvalue, 2.498150305344973e-02, "C kupiec p (250, 0)");

    // Exception pattern: t % 40 == 0, plus days 100 and 101 (9 exceptions).
    std::vector<std::uint8_t> ex(250, 0);
    for (int t = 0; t < 250; t += 40) ex[t] = 1;
    ex[100] = 1;
    ex[101] = 1;
    int count = 0;
    for (std::uint8_t v : ex) count += v;
    ASSERT_EQ(count, 9);
    const ChristoffersenResult ind = christoffersen_independence(ex);
    expect_rel(ind.lr, 1.189356349111719e+00, "C christoffersen ind LR");
    expect_rel(ind.pvalue, 2.754594267438438e-01, "C christoffersen ind p");
    const ConditionalCoverageResult cc = christoffersen_cc(ex, 0.01);
    expect_rel(cc.lr, 1.141838698170947e+01, "C christoffersen cc LR");
    expect_rel(cc.pvalue, 3.315345323267836e-03, "C christoffersen cc p");
}
