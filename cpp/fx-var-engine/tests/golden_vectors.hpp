// GENERATED FILE -- do not edit by hand.
//
// Source : python/fx/03-var-es-engine (fx_var)
// Emitted: tools/gen_golden_header.py from
//          python/fx/03-var-es-engine/tests/golden/golden_vectors.json
// Regenerate the JSON with the Python project's
// tests/golden/generate_golden.py, then rerun this script and commit
// both outputs together.  `tools/gen_golden_header.py --check` fails
// if this header does not match the JSON.
//
// Golden VaR/ES/backtest vectors for cross-validating the C++/Rust fx-var-engine against the fx_var Python reference. Inputs are closed-form sin/cos series (no RNG). Tolerances for consumers: P&L-scale figures 1e-6 absolute (case A) / 1e-8 relative (case B); Kupiec LR 1e-10, p-values 1e-11 (Kupiec) and 1e-10 (Christoffersen); Basel cumulative probabilities 1e-12 absolute.

#ifndef FXVAR_TESTS_GOLDEN_VECTORS_HPP
#define FXVAR_TESTS_GOLDEN_VECTORS_HPP

namespace fxvar::golden {

// Consumer tolerances declared by the generator.
inline constexpr double tol_case_a_abs = 1e-06;
inline constexpr double tol_case_b_rel = 1e-08;
inline constexpr double tol_kupiec_lr_abs = 1e-10;
inline constexpr double tol_kupiec_p_abs = 1e-11;
inline constexpr double tol_christoffersen_abs = 1e-10;
inline constexpr double tol_basel_cum_prob_abs = 1e-12;
inline constexpr double tol_python_self_check_ulps = 16.0;
inline constexpr double tol_conv_higham_abs = 1e-08;
inline constexpr double tol_conv_clip_rel = 1e-12;

// Reference values.
inline constexpr double a_book_pnl_scenario_17 = 58177.374898099806;
inline constexpr double a_plain_var_99 = 61919.80890587624;
inline constexpr double a_plain_es_99 = 62006.12006224846;
inline constexpr double a_plain_var_975 = 61237.42600889597;
inline constexpr double a_plain_es_975 = 61777.936082718574;
inline constexpr double a_age_var_99_decay_0995 = 61874.26268531149;
inline constexpr double a_age_es_99_decay_0995 = 61977.524965941084;
inline constexpr double b_normal_var_99_1d = 153339.50441962917;
inline constexpr double b_normal_es_99_1d = 175675.62972002864;
inline constexpr double b_t5_var_99_1d = 171803.12389091408;
inline constexpr double b_t5_es_99_1d = 227327.53149741463;
inline constexpr double b_normal_var_99_10d = 484902.0892474838;
inline constexpr double b_normal_es_99_10d = 555535.1192996588;
inline constexpr double c_kupiec_lr_8_250 = 7.7335507244945205;
inline constexpr double c_kupiec_p_8_250 = 0.0054204051941277994;
inline constexpr double c_christoffersen_ind_lr = 1.0063610339314124;
inline constexpr double c_christoffersen_ind_p = 0.3157762037622499;
inline constexpr double c_basel_cum_prob_4 = 0.8921876269036253;
inline constexpr const char* c_basel_zone_4 = "green";
inline constexpr double c_basel_multiplier_4 = 3.0;
inline constexpr double c_basel_cum_prob_5 = 0.9588168159301517;
inline constexpr const char* c_basel_zone_5 = "yellow";
inline constexpr double c_basel_multiplier_5 = 3.4;
inline constexpr double c_basel_cum_prob_10 = 0.999946101370953;
inline constexpr const char* c_basel_zone_10 = "red";
inline constexpr double c_basel_multiplier_10 = 4.0;
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
inline constexpr double conv_kupiec_lr_n500_x9 = 2.612570620003453;
inline constexpr double conv_kupiec_lr_n250_x0 = 5.025167926750726;
inline constexpr double conv_as_z2 = -3.1854547692307698;
inline constexpr double conv_cf_valid_s0122_km0427 = 1.0;
inline constexpr double conv_cf_valid_s00_k00 = 1.0;
inline constexpr double conv_cf_valid_s10_k10 = 1.0;
inline constexpr double conv_cf_valid_sm0105_k80001 = 0.0;
inline constexpr double conv_cf_valid_a001_s0122_km0427 = 1.0;
inline constexpr double conv_higham_a01 = 0.5000000000001574;
inline constexpr double conv_higham_a02 = 0.5000000000001575;
inline constexpr double conv_higham_a12 = -0.5000000000001574;
inline constexpr double conv_clip_a01 = 0.49999999996052624;
inline constexpr double conv_clip_a02 = 0.4999999999605266;
inline constexpr double conv_clip_a12 = -0.4999999999605262;
inline constexpr double conv_ewma_sigma_last = 0.007821944025280376;

}  // namespace fxvar::golden

#endif  // FXVAR_TESTS_GOLDEN_VECTORS_HPP
