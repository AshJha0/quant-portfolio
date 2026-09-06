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
pub const A_HIST_VAR_1PCT: f64 = 122.41292223752643;
pub const A_HIST_VAR_5PCT: f64 = 104.55228353749273;
pub const A_ES_5PCT: f64 = 122.6373207703405;
pub const A_ES_1PCT: f64 = 141.75681075495305;
pub const A_BRW_VAR_5PCT_LAM098: f64 = 108.2348601407293;
pub const A_FHS_VAR_5PCT_LAM094: f64 = 109.3777910164513;
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
pub const C_EWMA_COV_00: f64 = 6.0856537940667185e-5;
pub const C_EWMA_COV_01: f64 = 3.159039791408869e-5;
pub const C_EWMA_COV_22: f64 = 6.69554835931717e-5;
pub const C_VAR_NORMAL_1PCT: f64 = 24030.56180968714;
pub const C_VAR_T8_5PCT: f64 = 16635.172157903933;
pub const C_KUPIEC_LR_250_7: f64 = 5.496990447792683;
pub const C_KUPIEC_P_250_7: f64 = 0.019049230890526535;
pub const C_KUPIEC_LR_250_0: f64 = 5.025167926750726;
pub const C_KUPIEC_P_250_0: f64 = 0.02498150305344973;
pub const C_CHRISTOFFERSEN_IND_LR: f64 = 1.1893563491117192;
pub const C_CHRISTOFFERSEN_IND_P: f64 = 0.2754594267438438;
pub const C_CHRISTOFFERSEN_CC_LR: f64 = 11.418386981709475;
pub const C_CHRISTOFFERSEN_CC_P: f64 = 0.0033153453232678356;
