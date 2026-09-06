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

// Reference values.
pub const A_BOOK_PNL_SCENARIO_17: f64 = 58177.37489810074;
pub const A_PLAIN_VAR_99: f64 = 61919.80890587624;
pub const A_PLAIN_ES_99: f64 = 62006.12006224847;
pub const A_PLAIN_VAR_975: f64 = 61237.42600889597;
pub const A_PLAIN_ES_975: f64 = 61777.93608271857;
pub const A_AGE_VAR_99_DECAY_0995: f64 = 61874.26268531149;
pub const A_AGE_ES_99_DECAY_0995: f64 = 61977.52496594109;
pub const B_NORMAL_VAR_99_1D: f64 = 153339.50441962917;
pub const B_NORMAL_ES_99_1D: f64 = 175675.6297200285;
pub const B_T5_VAR_99_1D: f64 = 171803.12389091405;
pub const B_T5_ES_99_1D: f64 = 227327.5314974144;
pub const B_NORMAL_VAR_99_10D: f64 = 484902.0892474838;
pub const B_NORMAL_ES_99_10D: f64 = 555535.1192996583;
pub const C_KUPIEC_LR_8_250: f64 = 7.7335507244945205;
pub const C_KUPIEC_P_8_250: f64 = 0.0054204051941277994;
pub const C_CHRISTOFFERSEN_IND_LR: f64 = 1.0063610339314124;
pub const C_CHRISTOFFERSEN_IND_P: f64 = 0.3157762037622499;
pub const C_BASEL_CUM_PROB_4: f64 = 0.8921876269036249;
pub const C_BASEL_ZONE_4: &str = "green";
pub const C_BASEL_MULTIPLIER_4: f64 = 3.0;
pub const C_BASEL_CUM_PROB_5: f64 = 0.9588168159301514;
pub const C_BASEL_ZONE_5: &str = "yellow";
pub const C_BASEL_MULTIPLIER_5: f64 = 3.4;
pub const C_BASEL_CUM_PROB_10: f64 = 0.999946101370953;
pub const C_BASEL_ZONE_10: &str = "red";
pub const C_BASEL_MULTIPLIER_10: f64 = 4.0;
