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

// Reference values.
inline constexpr double a_book_pnl_scenario_17 = 58177.37489810074;
inline constexpr double a_plain_var_99 = 61919.80890587624;
inline constexpr double a_plain_es_99 = 62006.12006224847;
inline constexpr double a_plain_var_975 = 61237.42600889597;
inline constexpr double a_plain_es_975 = 61777.93608271857;
inline constexpr double a_age_var_99_decay_0995 = 61874.26268531149;
inline constexpr double a_age_es_99_decay_0995 = 61977.52496594109;
inline constexpr double b_normal_var_99_1d = 153339.50441962917;
inline constexpr double b_normal_es_99_1d = 175675.6297200285;
inline constexpr double b_t5_var_99_1d = 171803.12389091405;
inline constexpr double b_t5_es_99_1d = 227327.5314974144;
inline constexpr double b_normal_var_99_10d = 484902.0892474838;
inline constexpr double b_normal_es_99_10d = 555535.1192996583;
inline constexpr double c_kupiec_lr_8_250 = 7.7335507244945205;
inline constexpr double c_kupiec_p_8_250 = 0.0054204051941277994;
inline constexpr double c_christoffersen_ind_lr = 1.0063610339314124;
inline constexpr double c_christoffersen_ind_p = 0.3157762037622499;
inline constexpr double c_basel_cum_prob_4 = 0.8921876269036249;
inline constexpr const char* c_basel_zone_4 = "green";
inline constexpr double c_basel_multiplier_4 = 3.0;
inline constexpr double c_basel_cum_prob_5 = 0.9588168159301514;
inline constexpr const char* c_basel_zone_5 = "yellow";
inline constexpr double c_basel_multiplier_5 = 3.4;
inline constexpr double c_basel_cum_prob_10 = 0.999946101370953;
inline constexpr const char* c_basel_zone_10 = "red";
inline constexpr double c_basel_multiplier_10 = 4.0;

}  // namespace fxvar::golden

#endif  // FXVAR_TESTS_GOLDEN_VECTORS_HPP
