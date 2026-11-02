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
inline constexpr double a_hist_var_1pct = 141.75681075495305;
inline constexpr double a_hist_var_5pct = 108.2348601407293;
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
inline constexpr double c_ewma_cov_00 = 6.0765624368713895e-05;
inline constexpr double c_ewma_cov_01 = 3.154393297548715e-05;
inline constexpr double c_ewma_cov_22 = 6.693013810519e-05;
inline constexpr double c_var_normal_1pct = 24030.561809687137;
inline constexpr double c_var_t8_5pct = 16635.17215790393;
inline constexpr double c_kupiec_lr_250_7 = 5.496990447792697;
inline constexpr double c_kupiec_p_250_7 = 0.01904923089052634;
inline constexpr double c_kupiec_lr_250_0 = 5.025167926750721;
inline constexpr double c_kupiec_p_250_0 = 0.024981503053449754;
inline constexpr double c_christoffersen_ind_lr = 1.1893563491117192;
inline constexpr double c_christoffersen_ind_p = 0.2754594267438438;
inline constexpr double c_christoffersen_cc_lr = 11.418386981709489;
inline constexpr double c_christoffersen_cc_p = 0.0033153453232678126;
inline constexpr double conv_var_a010 = 1434.4;
inline constexpr double conv_es_a010 = 1463.968;
inline constexpr double conv_var_a025 = 1353.23;
inline constexpr double conv_es_a025 = 1414.5228000000002;
inline constexpr double conv_var_a050 = 1193.76;
inline constexpr double conv_es_a050 = 1349.7856;
inline constexpr double conv_basel_zone_n250_x4_a010 = 0.0;
inline constexpr double conv_basel_k_n250_x4_a010 = 3.0;
inline constexpr double conv_basel_cdf_n250_x4_a010 = 0.8921876269036253;
inline constexpr double conv_basel_zone_n250_x5_a010 = 1.0;
inline constexpr double conv_basel_k_n250_x5_a010 = 3.4;
inline constexpr double conv_basel_cdf_n250_x5_a010 = 0.9588168159301517;
inline constexpr double conv_basel_zone_n250_x9_a010 = 1.0;
inline constexpr double conv_basel_k_n250_x9_a010 = 3.85;
inline constexpr double conv_basel_cdf_n250_x9_a010 = 0.9997498099312595;
inline constexpr double conv_basel_zone_n250_x10_a010 = 2.0;
inline constexpr double conv_basel_k_n250_x10_a010 = 4.0;
inline constexpr double conv_basel_cdf_n250_x10_a010 = 0.999946101370953;
inline constexpr double conv_basel_zone_n500_x8_a010 = 0.0;
inline constexpr double conv_basel_k_n500_x8_a010 = 3.0;
inline constexpr double conv_basel_cdf_n500_x8_a010 = 0.9328898400862948;
inline constexpr double conv_basel_zone_n500_x9_a010 = 1.0;
inline constexpr double conv_basel_k_n500_x9_a010 = 3.5;
inline constexpr double conv_basel_cdf_n500_x9_a010 = 0.9688978933515117;
inline constexpr double conv_basel_zone_n500_x12_a010 = 1.0;
inline constexpr double conv_basel_k_n500_x12_a010 = 3.75;
inline constexpr double conv_basel_cdf_n500_x12_a010 = 0.9980995068161164;
inline constexpr double conv_basel_zone_n1000_x15_a010 = 1.0;
inline constexpr double conv_basel_k_n1000_x15_a010 = 3.4;
inline constexpr double conv_basel_cdf_n1000_x15_a010 = 0.9521294142420523;
inline constexpr double conv_basel_zone_n250_x12_a025 = 1.0;
inline constexpr double conv_basel_k_n250_x12_a025 = 3.65;
inline constexpr double conv_basel_cdf_n250_x12_a025 = 0.989001974898185;
inline constexpr double conv_kupiec_lr_n250_x5 = 1.956809788230622;
inline constexpr double conv_kupiec_lr_n500_x9 = 2.6125706200034813;
inline constexpr double conv_kupiec_lr_n250_x0 = 5.025167926750721;
inline constexpr double conv_as_z2 = -3.1854547692307698;
inline constexpr double conv_cf_valid_s0122_km0427 = 1.0;
inline constexpr double conv_cf_valid_s00_k00 = 1.0;
inline constexpr double conv_cf_valid_s10_k10 = 1.0;
inline constexpr double conv_cf_valid_sm0105_k80001 = 0.0;
inline constexpr double conv_cf_valid_a001_s0122_km0427 = 1.0;
inline constexpr double conv_higham_a01 = 0.5000000000001574;
inline constexpr double conv_clip_a01 = 0.49999999996052624;
inline constexpr double conv_higham_a02 = 0.5000000000001575;
inline constexpr double conv_clip_a02 = 0.4999999999605266;
inline constexpr double conv_higham_a12 = -0.5000000000001574;
inline constexpr double conv_clip_a12 = -0.4999999999605262;
inline constexpr double conv_ewma_sigma_last = 0.007821944025280376;

}  // namespace eqvar::golden

#endif  // EQVAR_TESTS_GOLDEN_VECTORS_HPP
