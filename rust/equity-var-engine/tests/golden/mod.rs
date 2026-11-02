//! GENERATED FILE -- do not edit by hand.
//!
//! Source : python/equity/03-var-es-engine (eq_var)
//! Emitted: tools/gen_golden_rs.py from
//!          python/equity/03-var-es-engine/tests/golden/golden_vectors.json
//!
//! Regenerate the JSON with the Python project's
//! tests/golden/generate_golden.py, then rerun this script and commit
//! both outputs together.  `tools/gen_golden_rs.py --check` fails if
//! this module does not match the JSON.
//!
//! Golden VaR/ES/backtest vectors for cross-validating the C++/Rust equity-var-engine against the eq_var Python reference. Inputs are closed-form sin/cos series (no RNG); tolerance for consumers: 1e-9 relative + 1e-12 absolute (libm-vs-numpy ulp noise in sin/cos).

#![allow(dead_code, clippy::excessive_precision, clippy::unreadable_literal)]

// Consumer tolerances declared by the generator.
pub const TOL_REL: f64 = 1.0e-9;
pub const TOL_ABS: f64 = 1.0e-12;

// Reference values.
pub const A_HIST_VAR_1PCT: f64 = 141.75681075495305;
pub const A_HIST_VAR_5PCT: f64 = 108.2348601407293;
pub const A_ES_5PCT: f64 = 122.6373207703405;
pub const A_ES_1PCT: f64 = 141.75681075495305;
pub const A_BRW_VAR_5PCT_LAM098: f64 = 108.2348601407293;
pub const A_FHS_VAR_5PCT_LAM094: f64 = 112.05212695060598;
pub const A_BRW_WEIGHTS_5_LAM098: [f64; 5] = [0.1920016255921656, 0.19592002611445472, 0.19991839399434155, 0.20399836121871587, 0.2081615930803223];
pub const B_SIGMA: f64 = 9962.429422585637;
pub const B_VAR_NORMAL_1PCT: f64 = 23176.07650751402;
pub const B_VAR_T6_1PCT: f64 = 25563.37478743866;
pub const B_VAR_NORMAL_1PCT_10D: f64 = 73289.18899006478;
pub const B_ES_NORMAL_2P5PCT: f64 = 23290.19532123021;
pub const B_ES_T6_2P5PCT: f64 = 26486.475881703944;
pub const B_CF_VAR_1PCT_SKEW_M05_KURT_1: f64 = 28230.62626871169;
pub const C_SAMPLE_COV_00: f64 = 6.197010057346647e-5;
pub const C_SAMPLE_COV_01: f64 = 3.367477490492125e-5;
pub const C_SAMPLE_COV_12: f64 = 3.643904950084309e-5;
pub const C_EWMA_COV_00: f64 = 6.0765624368713895e-5;
pub const C_EWMA_COV_01: f64 = 3.154393297548715e-5;
pub const C_EWMA_COV_22: f64 = 6.693013810519e-5;
pub const C_VAR_NORMAL_1PCT: f64 = 24030.561809687137;
pub const C_VAR_T8_5PCT: f64 = 16635.17215790393;
pub const C_KUPIEC_LR_250_7: f64 = 5.496990447792697;
pub const C_KUPIEC_P_250_7: f64 = 0.01904923089052634;
pub const C_KUPIEC_LR_250_0: f64 = 5.025167926750721;
pub const C_KUPIEC_P_250_0: f64 = 0.024981503053449754;
pub const C_CHRISTOFFERSEN_IND_LR: f64 = 1.1893563491117192;
pub const C_CHRISTOFFERSEN_IND_P: f64 = 0.2754594267438438;
pub const C_CHRISTOFFERSEN_CC_LR: f64 = 11.418386981709489;
pub const C_CHRISTOFFERSEN_CC_P: f64 = 0.0033153453232678126;
pub const CONV_VAR_A010: f64 = 1434.4;
pub const CONV_ES_A010: f64 = 1463.968;
pub const CONV_VAR_A025: f64 = 1353.23;
pub const CONV_ES_A025: f64 = 1414.5228000000002;
pub const CONV_VAR_A050: f64 = 1193.76;
pub const CONV_ES_A050: f64 = 1349.7856;
pub const CONV_BASEL_ZONE_N250_X4_A010: f64 = 0.0;
pub const CONV_BASEL_K_N250_X4_A010: f64 = 3.0;
pub const CONV_BASEL_CDF_N250_X4_A010: f64 = 0.8921876269036253;
pub const CONV_BASEL_ZONE_N250_X5_A010: f64 = 1.0;
pub const CONV_BASEL_K_N250_X5_A010: f64 = 3.4;
pub const CONV_BASEL_CDF_N250_X5_A010: f64 = 0.9588168159301517;
pub const CONV_BASEL_ZONE_N250_X9_A010: f64 = 1.0;
pub const CONV_BASEL_K_N250_X9_A010: f64 = 3.85;
pub const CONV_BASEL_CDF_N250_X9_A010: f64 = 0.9997498099312595;
pub const CONV_BASEL_ZONE_N250_X10_A010: f64 = 2.0;
pub const CONV_BASEL_K_N250_X10_A010: f64 = 4.0;
pub const CONV_BASEL_CDF_N250_X10_A010: f64 = 0.999946101370953;
pub const CONV_BASEL_ZONE_N500_X8_A010: f64 = 0.0;
pub const CONV_BASEL_K_N500_X8_A010: f64 = 3.0;
pub const CONV_BASEL_CDF_N500_X8_A010: f64 = 0.9328898400862948;
pub const CONV_BASEL_ZONE_N500_X9_A010: f64 = 1.0;
pub const CONV_BASEL_K_N500_X9_A010: f64 = 3.5;
pub const CONV_BASEL_CDF_N500_X9_A010: f64 = 0.9688978933515117;
pub const CONV_BASEL_ZONE_N500_X12_A010: f64 = 1.0;
pub const CONV_BASEL_K_N500_X12_A010: f64 = 3.75;
pub const CONV_BASEL_CDF_N500_X12_A010: f64 = 0.9980995068161164;
pub const CONV_BASEL_ZONE_N1000_X15_A010: f64 = 1.0;
pub const CONV_BASEL_K_N1000_X15_A010: f64 = 3.4;
pub const CONV_BASEL_CDF_N1000_X15_A010: f64 = 0.9521294142420523;
pub const CONV_BASEL_ZONE_N250_X12_A025: f64 = 1.0;
pub const CONV_BASEL_K_N250_X12_A025: f64 = 3.65;
pub const CONV_BASEL_CDF_N250_X12_A025: f64 = 0.989001974898185;
pub const CONV_KUPIEC_LR_N250_X5: f64 = 1.956809788230622;
pub const CONV_KUPIEC_LR_N500_X9: f64 = 2.6125706200034813;
pub const CONV_KUPIEC_LR_N250_X0: f64 = 5.025167926750721;
pub const CONV_AS_Z2: f64 = -3.1854547692307698;
pub const CONV_CF_VALID_S0122_KM0427: f64 = 1.0;
pub const CONV_CF_VALID_S00_K00: f64 = 1.0;
pub const CONV_CF_VALID_S10_K10: f64 = 1.0;
pub const CONV_CF_VALID_SM0105_K80001: f64 = 0.0;
pub const CONV_CF_VALID_A001_S0122_KM0427: f64 = 1.0;
pub const CONV_HIGHAM_A01: f64 = 0.5000000000001574;
pub const CONV_CLIP_A01: f64 = 0.49999999996052624;
pub const CONV_HIGHAM_A02: f64 = 0.5000000000001575;
pub const CONV_CLIP_A02: f64 = 0.4999999999605266;
pub const CONV_HIGHAM_A12: f64 = -0.5000000000001574;
pub const CONV_CLIP_A12: f64 = -0.4999999999605262;
pub const CONV_EWMA_SIGMA_LAST: f64 = 0.007821944025280376;
