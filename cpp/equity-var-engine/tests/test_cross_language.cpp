// Cross-language golden tests against the Python reference implementation.
//
// Provenance: every expected constant comes from `golden_vectors.hpp`, a
// GENERATED header emitted by `tools/gen_golden_header.py` from the JSON
// written by the Python research stack (the conceptual twin of this
// engine):
//
//   cd python/equity/03-var-es-engine
//   PYTHONPATH=src python3 tests/golden/generate_golden.py   # -> golden_vectors.json
//   cd ../../../cpp/equity-var-engine
//   python3 tools/gen_golden_header.py                       # -> tests/golden_vectors.hpp
//   python3 tools/gen_golden_header.py --check               # provenance check
//
// The JSON is itself locked by python/equity/03-var-es-engine/tests/
// test_golden_vectors.py, so a Python-side drift fails there first.
// Inputs are closed-form (sin/cos-based) deterministic series so both
// languages regenerate them independently; agreement is required to 1e-9
// relative + 1e-12 absolute (the residual is libm-vs-numpy ulp noise in
// sin/cos, far below any risk tolerance).
#include <gtest/gtest.h>

#include <cmath>
#include <iterator>
#include <vector>

#include "eqvar/backtest.hpp"
#include "eqvar/expected_shortfall.hpp"
#include "eqvar/historical.hpp"
#include "eqvar/matrix.hpp"
#include "eqvar/parametric.hpp"
#include "golden_vectors.hpp"

using namespace eqvar;

namespace {

// Relative tolerance for cross-language agreement.
void expect_rel(double actual, double expected, const char* what) {
    EXPECT_NEAR(actual, expected, golden::tol_rel * std::abs(expected) + golden::tol_abs)
        << what;
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
    expect_rel(historical_var(pnl, 0.01), golden::a_hist_var_1pct, "A hist VaR 1%");
    expect_rel(historical_var(pnl, 0.05), golden::a_hist_var_5pct, "A hist VaR 5%");
    expect_rel(expected_shortfall(pnl, 0.05), golden::a_es_5pct, "A ES 5%");
    expect_rel(expected_shortfall(pnl, 0.01), golden::a_es_1pct, "A ES 1%");
    expect_rel(age_weighted_var(pnl, 0.05, 0.98), golden::a_brw_var_5pct_lam098, "A BRW VaR 5%");
    expect_rel(filtered_historical_var(pnl, 0.05, 0.94), golden::a_fhs_var_5pct_lam094, "A FHS VaR 5%");
}

TEST(CrossLanguage, CaseA_BrwWeights) {
    const std::vector<double> w = brw_weights(5, 0.98);
    const std::vector<double> expected(std::begin(golden::a_brw_weights_5_lam098),
                                       std::end(golden::a_brw_weights_5_lam098));
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
    expect_rel(sigma, golden::b_sigma, "B sigma");
    expect_rel(parametric_var(w, cov, 0.01), golden::b_var_normal_1pct, "B VaR normal 1%");
    expect_rel(parametric_var(w, cov, 0.01, Dist::StudentT, 6.0), golden::b_var_t6_1pct,
               "B VaR t6 1%");
    expect_rel(parametric_var(w, cov, 0.01, Dist::Normal, 6.0, 0.0, 10),
               golden::b_var_normal_1pct_10d, "B VaR normal 1% 10d");
    expect_rel(normal_es(sigma, 0.025), golden::b_es_normal_2p5pct, "B ES normal 2.5%");
    expect_rel(student_t_es(sigma, 0.025, 6.0), golden::b_es_t6_2p5pct, "B ES t6 2.5%");
    expect_rel(cornish_fisher_var(sigma, 0.01, -0.5, 1.0), golden::b_cf_var_1pct_skew_m05_kurt_1,
               "B CF VaR 1%");
}

TEST(CrossLanguage, CaseC_CovarianceEstimators) {
    const Matrix r = case_c_returns();
    const Matrix s = sample_covariance(r);
    const Matrix e = ewma_covariance(r, 0.94);
    expect_rel(s(0, 0), golden::c_sample_cov_00, "C sample cov 00");
    expect_rel(s(0, 1), golden::c_sample_cov_01, "C sample cov 01");
    expect_rel(s(1, 2), golden::c_sample_cov_12, "C sample cov 12");
    expect_rel(e(0, 0), golden::c_ewma_cov_00, "C ewma cov 00");
    expect_rel(e(0, 1), golden::c_ewma_cov_01, "C ewma cov 01");
    expect_rel(e(2, 2), golden::c_ewma_cov_22, "C ewma cov 22");

    const std::vector<double> w = {2.0e6, -1.0e6, 5.0e5};
    expect_rel(parametric_var(w, s, 0.01), golden::c_var_normal_1pct, "C VaR normal 1%");
    expect_rel(parametric_var(w, s, 0.05, Dist::StudentT, 8.0), golden::c_var_t8_5pct,
               "C VaR t8 5%");
}

TEST(CrossLanguage, CaseC_BacktestStatistics) {
    const KupiecResult k7 = kupiec_pof(250, 7, 0.01);
    expect_rel(k7.lr, golden::c_kupiec_lr_250_7, "C kupiec LR (250, 7)");
    expect_rel(k7.pvalue, golden::c_kupiec_p_250_7, "C kupiec p (250, 7)");
    const KupiecResult k0 = kupiec_pof(250, 0, 0.01);
    expect_rel(k0.lr, golden::c_kupiec_lr_250_0, "C kupiec LR (250, 0)");
    expect_rel(k0.pvalue, golden::c_kupiec_p_250_0, "C kupiec p (250, 0)");

    // Exception pattern: t % 40 == 0, plus days 100 and 101 (9 exceptions).
    std::vector<std::uint8_t> ex(250, 0);
    for (int t = 0; t < 250; t += 40) ex[t] = 1;
    ex[100] = 1;
    ex[101] = 1;
    int count = 0;
    for (std::uint8_t v : ex) count += v;
    ASSERT_EQ(count, 9);
    const ChristoffersenResult ind = christoffersen_independence(ex);
    expect_rel(ind.lr, golden::c_christoffersen_ind_lr, "C christoffersen ind LR");
    expect_rel(ind.pvalue, golden::c_christoffersen_ind_p, "C christoffersen ind p");
    const ConditionalCoverageResult cc = christoffersen_cc(ex, 0.01);
    expect_rel(cc.lr, golden::c_christoffersen_cc_lr, "C christoffersen cc LR");
    expect_rel(cc.pvalue, golden::c_christoffersen_cc_p, "C christoffersen cc p");
}
