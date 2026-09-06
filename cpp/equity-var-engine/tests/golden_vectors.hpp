// GENERATED FILE -- do not edit by hand.
//
// Source : python/equity/03-var-es-engine (eq_var)
// Emitted: tools/gen_golden_header.py from
//          python/equity/03-var-es-engine/tests/golden/golden_vectors.json
// Regenerate the JSON with the Python project's
// tests/golden/generate_golden.py, then rerun this script and commit
// both outputs together.  `tools/gen_golden_header.py --check` fails
// if this header does not match the JSON.
//
// Golden VaR/ES/backtest vectors for cross-validating the C++/Rust equity-var-engine against the eq_var Python reference. Inputs are closed-form sin/cos series (no RNG); tolerance for consumers: 1e-9 relative + 1e-12 absolute (libm-vs-numpy ulp noise in sin/cos).

#ifndef EQVAR_TESTS_GOLDEN_VECTORS_HPP
#define EQVAR_TESTS_GOLDEN_VECTORS_HPP

namespace eqvar::golden {

// Consumer tolerances declared by the generator.
inline constexpr double tol_rel = 1e-09;
inline constexpr double tol_abs = 1e-12;

// Reference values.
inline constexpr double a_hist_var_1pct = 122.41292223752643;
inline constexpr double a_hist_var_5pct = 104.55228353749273;
inline constexpr double a_es_5pct = 122.6373207703405;
inline constexpr double a_es_1pct = 141.75681075495305;
inline constexpr double a_brw_var_5pct_lam098 = 108.2348601407293;
inline constexpr double a_fhs_var_5pct_lam094 = 109.3777910164513;
inline constexpr double a_brw_weights_5_lam098[5] = {0.1920016255921656, 0.19592002611445472, 0.19991839399434155, 0.20399836121871587, 0.2081615930803223};
inline constexpr double b_sigma = 9962.429422585637;
inline constexpr double b_var_normal_1pct = 23176.07650751402;
inline constexpr double b_var_t6_1pct = 25563.37478743866;
inline constexpr double b_var_normal_1pct_10d = 73289.18899006478;
inline constexpr double b_es_normal_2p5pct = 23290.19532123021;
inline constexpr double b_es_t6_2p5pct = 26486.475881703944;
inline constexpr double b_cf_var_1pct_skew_m05_kurt_1 = 28230.62626871169;
inline constexpr double c_sample_cov_00 = 6.197010057346647e-05;
inline constexpr double c_sample_cov_01 = 3.367477490492125e-05;
inline constexpr double c_sample_cov_12 = 3.643904950084309e-05;
inline constexpr double c_ewma_cov_00 = 6.0856537940667185e-05;
inline constexpr double c_ewma_cov_01 = 3.159039791408869e-05;
inline constexpr double c_ewma_cov_22 = 6.69554835931717e-05;
inline constexpr double c_var_normal_1pct = 24030.56180968714;
inline constexpr double c_var_t8_5pct = 16635.172157903933;
inline constexpr double c_kupiec_lr_250_7 = 5.496990447792683;
inline constexpr double c_kupiec_p_250_7 = 0.019049230890526535;
inline constexpr double c_kupiec_lr_250_0 = 5.025167926750726;
inline constexpr double c_kupiec_p_250_0 = 0.02498150305344973;
inline constexpr double c_christoffersen_ind_lr = 1.1893563491117192;
inline constexpr double c_christoffersen_ind_p = 0.2754594267438438;
inline constexpr double c_christoffersen_cc_lr = 11.418386981709475;
inline constexpr double c_christoffersen_cc_p = 0.0033153453232678356;

}  // namespace eqvar::golden

#endif  // EQVAR_TESTS_GOLDEN_VECTORS_HPP
