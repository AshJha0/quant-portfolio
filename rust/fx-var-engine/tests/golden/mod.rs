//! GENERATED FILE -- do not edit by hand.
//!
//! Source : python/fx/03-var-es-engine (fx_var)
//! Emitted: tools/gen_golden_rs.py from
//!          python/fx/03-var-es-engine/tests/golden/golden_vectors.json
//!
//! Regenerate the JSON with the Python project's
//! tests/golden/generate_golden.py, then rerun this script and commit
//! both outputs together.  `tools/gen_golden_rs.py --check` fails if
//! this module does not match the JSON.
//!
//! Golden VaR/ES/backtest vectors for cross-validating the C++/Rust fx-var-engine against the fx_var Python reference. Inputs are closed-form sin/cos series (no RNG). Tolerances for consumers: P&L-scale figures 1e-6 absolute (case A) / 1e-8 relative (case B); Kupiec LR 1e-10, p-values 1e-11 (Kupiec) and 1e-10 (Christoffersen); Basel cumulative probabilities 1e-12 absolute.

#![allow(dead_code, clippy::excessive_precision, clippy::unreadable_literal)]

// Consumer tolerances declared by the generator.
pub const TOL_CASE_A_ABS: f64 = 1.0e-6;
pub const TOL_CASE_B_REL: f64 = 1.0e-8;
pub const TOL_KUPIEC_LR_ABS: f64 = 1.0e-10;
pub const TOL_KUPIEC_P_ABS: f64 = 1.0e-11;
pub const TOL_CHRISTOFFERSEN_ABS: f64 = 1.0e-10;
pub const TOL_BASEL_CUM_PROB_ABS: f64 = 1.0e-12;
pub const TOL_PYTHON_SELF_CHECK_ULPS: f64 = 16.0;
pub const TOL_CONV_HIGHAM_ABS: f64 = 1.0e-8;
pub const TOL_CONV_CLIP_REL: f64 = 1.0e-12;

// Reference values.
pub const A_BOOK_PNL_SCENARIO_17: f64 = 58177.374898099806;
pub const A_PLAIN_VAR_99: f64 = 61919.80890587624;
pub const A_PLAIN_ES_99: f64 = 62006.12006224846;
pub const A_PLAIN_VAR_975: f64 = 61237.42600889597;
pub const A_PLAIN_ES_975: f64 = 61777.936082718574;
pub const A_AGE_VAR_99_DECAY_0995: f64 = 61874.26268531149;
pub const A_AGE_ES_99_DECAY_0995: f64 = 61977.524965941084;
pub const B_NORMAL_VAR_99_1D: f64 = 153339.50441962917;
pub const B_NORMAL_ES_99_1D: f64 = 175675.62972002864;
pub const B_T5_VAR_99_1D: f64 = 171803.12389091408;
pub const B_T5_ES_99_1D: f64 = 227327.53149741463;
pub const B_NORMAL_VAR_99_10D: f64 = 484902.0892474838;
pub const B_NORMAL_ES_99_10D: f64 = 555535.1192996588;
pub const C_KUPIEC_LR_8_250: f64 = 7.7335507244945205;
pub const C_KUPIEC_P_8_250: f64 = 0.0054204051941277994;
pub const C_CHRISTOFFERSEN_IND_LR: f64 = 1.0063610339314124;
pub const C_CHRISTOFFERSEN_IND_P: f64 = 0.3157762037622499;
pub const C_BASEL_CUM_PROB_4: f64 = 0.8921876269036253;
pub const C_BASEL_ZONE_4: &str = "green";
pub const C_BASEL_MULTIPLIER_4: f64 = 3.0;
pub const C_BASEL_CUM_PROB_5: f64 = 0.9588168159301517;
pub const C_BASEL_ZONE_5: &str = "yellow";
pub const C_BASEL_MULTIPLIER_5: f64 = 3.4;
pub const C_BASEL_CUM_PROB_10: f64 = 0.999946101370953;
pub const C_BASEL_ZONE_10: &str = "red";
pub const C_BASEL_MULTIPLIER_10: f64 = 4.0;
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
pub const CONV_KUPIEC_LR_N500_X9: f64 = 2.612570620003453;
pub const CONV_KUPIEC_LR_N250_X0: f64 = 5.025167926750726;
pub const CONV_AS_Z2: f64 = 1.0;
pub const CONV_CF_VALID_S0122_KM0427: f64 = 1.0;
pub const CONV_CF_VALID_S00_K00: f64 = 1.0;
pub const CONV_CF_VALID_S10_K10: f64 = 1.0;
pub const CONV_CF_VALID_SM0105_K80001: f64 = 0.0;
pub const CONV_CF_VALID_A001_S0122_KM0427: f64 = 1.0;
pub const CONV_HIGHAM_A01: f64 = 0.5000000000001574;
pub const CONV_HIGHAM_A02: f64 = 0.5000000000001575;
pub const CONV_HIGHAM_A12: f64 = -0.5000000000001574;
pub const CONV_CLIP_A01: f64 = 0.49999999996052624;
pub const CONV_CLIP_A02: f64 = 0.4999999999605266;
pub const CONV_CLIP_A12: f64 = -0.4999999999605262;
pub const CONV_EWMA_SIGMA_LAST: f64 = 0.007821944025280376;
