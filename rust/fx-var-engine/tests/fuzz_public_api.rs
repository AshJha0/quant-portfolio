//! Fuzz walk over every public entry point (S12, H-1).
//!
//! Each property takes valid baseline arguments, replaces one argument (or
//! one element of a slice / matrix) with NaN, +inf or -inf, or uses an
//! empty / length-1 / negative input where that is invalid, and asserts
//! `Err`. A plausible number from bad input is a failure.
//!
//! Total functions (`norm_pdf`, `norm_cdf`, `cornish_fisher_z`,
//! `Matrix::matvec`, `Rng::gamma`, `Rng::chi_square`) return a float: for
//! them NaN in gives NaN out. `registry_covers_every_public_fn` fails when a
//! new `pub fn` appears in `src/` without being named here.
#![allow(deprecated)]
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use fx_var_engine::backtest::*;
use fx_var_engine::book::*;
use fx_var_engine::expected_shortfall::*;
use fx_var_engine::historical::*;
use fx_var_engine::matrix::Matrix;
use fx_var_engine::monte_carlo::*;
use fx_var_engine::parametric::*;
use fx_var_engine::psd::*;
use fx_var_engine::returns::*;
use fx_var_engine::rng::Rng;
use fx_var_engine::stats::*;
use fx_var_engine::stress::*;
use fx_var_engine::tail::{empirical_var_es_into, empirical_var_into, tail_rank, weighted_var_es};
use fx_var_engine::{FxVarError, KernelError, McDist, TailDist};
use proptest::prelude::*;

macro_rules! check {
    ($e:expr) => {
        prop_assert!($e, "{}", stringify!($e))
    };
}

fn bad() -> impl Strategy<Value = f64> {
    prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)]
}

fn bad_or_neg() -> impl Strategy<Value = f64> {
    prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY), -1e6..-1e-12]
}

fn bad_alpha() -> impl Strategy<Value = f64> {
    prop_oneof![bad(), -1.0..=0.0, 0.5..2.0]
}

fn bad_prob() -> impl Strategy<Value = f64> {
    prop_oneof![bad(), -1.0..-1e-9f64, 1.000_000_001..2.0f64]
}

fn pnl(n: usize) -> Vec<f64> {
    (0..n).map(|i| 1000.0 * (0.37 * i as f64 + 0.1).sin()).collect()
}

fn poison(mut v: Vec<f64>, i: usize, x: f64) -> Vec<f64> {
    let k = i % v.len();
    v[k] = x;
    v
}

fn market() -> Market {
    Market::new([("EUR", 1.10), ("JPY", 0.0090)], [("USD", 0.05), ("EUR", 0.03), ("JPY", 0.001)]).unwrap()
}

fn book() -> Book {
    Book::new(
        vec![
            Position::Spot(SpotPosition::new("EURUSD", 10e6, None)),
            Position::Forward(ForwardPosition::new("USDJPY", 5e6, 0.5, None)),
        ],
        "USD",
    )
    .unwrap()
}

fn factors() -> Vec<String> {
    CompiledBook::new(&book(), &market()).unwrap().factors().to_vec()
}

fn rets(n: usize) -> ReturnsMatrix {
    let f = factors();
    let mut d = Matrix::zeros(n, f.len());
    for t in 0..n {
        for j in 0..f.len() {
            let s = if f[j].starts_with("FX:") { 0.006 } else { 0.0004 };
            d.set(t, j, s * ((0.1 * t as f64 + j as f64).sin() + 0.3 * (0.7 * t as f64 * (j + 1) as f64).cos()));
        }
    }
    ReturnsMatrix::new(f, d).unwrap()
}

fn rets_poison(n: usize, i: usize, x: f64) -> ReturnsMatrix {
    let mut r = rets(n);
    let k = r.n_factors();
    r.data.set(i % n, i % k, x);
    r
}

fn cov() -> FactorCov {
    sample_cov(&rets(200)).unwrap()
}

fn mat2() -> Matrix {
    Matrix::from_rows(&[vec![1e-4, 2e-5], vec![2e-5, 4e-4]]).unwrap()
}

fn mat2_poison(i: usize, x: f64) -> Matrix {
    let mut m = mat2();
    match i % 3 {
        0 => m.set(0, 0, x),
        1 => {
            m.set(0, 1, x);
            m.set(1, 0, x);
        }
        _ => m.set(1, 1, x),
    }
    m
}

const W: [f64; 2] = [1.0e6, -5.0e5];

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn tail_and_closed_forms(x in bad(), a in bad_alpha(), neg in -1e6..-1e-9f64, df in prop_oneof![bad(), -5.0..=2.0f64], i in 0usize..1000) {
        let p = pnl(50);
        check!(empirical_var(&poison(p.clone(), i, x), 0.01, None).is_err());
        check!(empirical_var(&p, a, None).is_err());
        check!(empirical_var(&[], 0.01, None).is_err());
        check!(empirical_es(&poison(p.clone(), i, x), 0.01, None).is_err());
        check!(empirical_es(&p, 0.01, Some(&poison(vec![1.0; 50], i, x))).is_err());
        check!(empirical_es(&p, 0.01, Some(&[1.0])).is_err());
        check!(empirical_var_es(&p, a, None).is_err());
        check!(empirical_var_es(&p, 0.01, Some(&poison(vec![1.0; 50], i, neg))).is_err());
        check!(weighted_var_es(&p, &vec![0.0; 50], 0.05).is_err());
        check!(fx_var_engine::tail::empirical_var(&poison(p.clone(), i, x), 0.01).is_err());
        check!(fx_var_engine::tail::empirical_var_es(&p, a).is_err());
        check!(fx_var_engine::tail::validate_alpha(a).is_err());
        check!(tail_rank(0, 0.01).is_err());
        check!(tail_rank(10, a).is_err());
        check!(validate_alpha(a).is_err());
        check!(validate_horizon(x).is_err());
        check!(validate_horizon(neg).is_err());
        check!(normal_var(x, 0.01, 0.0).is_err());
        check!(normal_var(neg, 0.01, 0.0).is_err());
        check!(normal_var(1.0, a, 0.0).is_err());
        check!(normal_var(1.0, 0.01, x).is_err());
        check!(normal_es(x, 0.01, 0.0).is_err());
        check!(normal_es(1.0, a, 0.0).is_err());
        check!(student_t_var(1.0, 0.01, df, 0.0).is_err());
        check!(student_t_var(x, 0.01, 5.0, 0.0).is_err());
        check!(student_t_es(1.0, 0.01, df, 0.0).is_err());
        check!(student_t_es(1.0, a, 5.0, 0.0).is_err());
        check!(student_t_es(1.0, 0.01, 5.0, x).is_err());
    }

    #[test]
    fn parametric_family(x in bad(), a in bad_alpha(), neg in -1e6..-1e-9f64, i in 0usize..1000) {
        check!(portfolio_sigma(&[x, 1.0], &mat2()).is_err());
        check!(portfolio_sigma(&W, &mat2_poison(i, x)).is_err());
        check!(var_covar(&W, &mat2(), a, 1.0, TailDist::Normal, 6.0, 0.0).is_err());
        check!(var_covar(&W, &mat2(), 0.01, x, TailDist::Normal, 6.0, 0.0).is_err());
        check!(var_covar(&W, &mat2(), 0.01, 1.0, TailDist::Normal, 6.0, x).is_err());
        check!(var_covar(&W, &mat2(), 0.01, 1.0, TailDist::StudentT, x, 0.0).is_err());
        check!(var_covar(&[x, 1.0], &mat2(), 0.01, 1.0, TailDist::Normal, 6.0, 0.0).is_err());
        let o = ParametricOptions { alpha: a, ..Default::default() };
        check!(parametric_var(&book(), &market(), &rets(120), &o).is_err());
        check!(parametric_var(&book(), &market(), &rets_poison(120, i, x), &ParametricOptions::default()).is_err());
        check!(parametric_var(&Book::empty(), &market(), &rets(120), &ParametricOptions::default()).is_err());
        let o = ParametricOptions { ewma_lambda: x, cov_method: CovMethod::Ewma, ..Default::default() };
        check!(parametric_var(&book(), &market(), &rets(120), &o).is_err());
        check!(cornish_fisher_valid(x, 0.0, 0.01).is_err());
        check!(cornish_fisher_valid(0.0, 0.0, a).is_err());
        check!(cornish_fisher_var(x, 0.0, 0.0, 0.01, 0.0, 1.0, true).is_err());
        check!(cornish_fisher_var(neg, 0.0, 0.0, 0.01, 0.0, 1.0, true).is_err());
        check!(cornish_fisher_var(1.0, x, 0.0, 0.01, 0.0, 1.0, false).is_err());
        check!(cornish_fisher_var(1.0, 0.0, x, 0.01, 0.0, 1.0, false).is_err());
        check!(cornish_fisher_var(1.0, 0.0, 0.0, a, 0.0, 1.0, true).is_err());
        check!(cornish_fisher_var(1.0, 0.0, 0.0, 0.01, x, 1.0, true).is_err());
        check!(cornish_fisher_var(1.0, 0.0, 0.0, 0.01, 0.0, x, true).is_err());
        check!(!cornish_fisher_domain_ok(x, 0.0, 4.0, 801));
        check!(cornish_fisher_z(f64::NAN, 0.1, 0.1).is_nan());
    }

    #[test]
    fn historical_family(x in bad(), a in bad_alpha(), l in bad_or_neg(), i in 0usize..1000) {
        let o = HistoricalOptions { alpha: a, ..Default::default() };
        check!(historical_var(&book(), &market(), &rets(120), &o).is_err());
        check!(historical_var(&book(), &market(), &rets_poison(120, i, x), &HistoricalOptions::default()).is_err());
        check!(historical_var(&Book::empty(), &market(), &rets(120), &HistoricalOptions::default()).is_err());
        check!(historical_var(&book(), &market(), &rets(10), &HistoricalOptions::default()).is_err());
        let o = HistoricalOptions { horizon_days: x, ..Default::default() };
        check!(historical_var(&book(), &market(), &rets(120), &o).is_err());
        let o = HistoricalOptions { method: HsMethod::Age, decay: l, ..Default::default() };
        check!(historical_var(&book(), &market(), &rets(120), &o).is_err());
        let o = HistoricalOptions { method: HsMethod::Filtered, ewma_lambda: l, ..Default::default() };
        check!(historical_var(&book(), &market(), &rets(120), &o).is_err());
    }

    #[test]
    fn backtests(x in bad(), a in bad_alpha(), neg in -100i64..0, i in 0usize..1000) {
        let p = pnl(100);
        check!(kupiec_pof(&KupiecInput { n_obs: 250, n_exceptions: 5, alpha: a }).is_err());
        check!(kupiec_pof(&KupiecInput { n_obs: neg, n_exceptions: 0, alpha: 0.01 }).is_err());
        check!(kupiec_pof(&KupiecInput { n_obs: 250, n_exceptions: neg, alpha: 0.01 }).is_err());
        check!(kupiec_pof(&KupiecInput { n_obs: 250, n_exceptions: 251, alpha: 0.01 }).is_err());
        check!(kupiec_pof_confidence(5, 250, x).is_err());
        check!(basel_traffic_light(&BaselInput { n_obs: 250, n_exceptions: 5, alpha: a }).is_err());
        check!(basel_traffic_light(&BaselInput { n_obs: 250, n_exceptions: 300, alpha: 0.01 }).is_err());
        check!(basel_traffic_light_confidence(5, 250, x).is_err());
        check!(christoffersen_independence(&[1]).is_err());
        check!(christoffersen_independence(&[0, 2]).is_err());
        check!(conditional_coverage(&[0, 1, 0, 0], a).is_err());
        check!(evaluate_var_backtest(&poison(p.clone(), i, x), &[100.0; 100], 0.01).is_err());
        check!(evaluate_var_backtest(&p, &poison(vec![100.0; 100], i, x), 0.01).is_err());
        check!(evaluate_var_backtest(&p, &[100.0; 100], a).is_err());
        check!(evaluate_var_backtest(&p, &[-1.0; 100], 0.01).is_err());
        check!(evaluate_var_backtest(&[1.0], &[1.0], 0.01).is_err());
        check!(acerbi_szekely_z2(&poison(p.clone(), i, x), &[100.0], &[120.0], 0.025).is_err());
        check!(acerbi_szekely_z2(&p, &[x], &[120.0], 0.025).is_err());
        check!(acerbi_szekely_z2(&p, &[100.0], &[x], 0.025).is_err());
        check!(acerbi_szekely_z2(&p, &[100.0], &[120.0], a).is_err());
        check!(acerbi_szekely_z2(&[], &[1.0], &[1.0], 0.025).is_err());
    }

    #[test]
    fn monte_carlo(x in bad(), a in bad_alpha(), df in prop_oneof![bad(), -5.0..=2.0f64], pr in bad_prob(), i in 0usize..1000) {
        let c = cov();
        let mut cp = c.clone();
        let k = cp.cov.rows();
        cp.cov.set(i % k, i % k, x);
        let j = JumpSpec::default();
        check!(simulate_factor_returns(&cp, 100, McDist::Normal, 6.0, &j, 1, 1.0).is_err());
        check!(simulate_factor_returns(&c, 0, McDist::Normal, 6.0, &j, 1, 1.0).is_err());
        check!(simulate_factor_returns(&c, 100, McDist::StudentT, df, &j, 1, 1.0).is_err());
        check!(simulate_factor_returns(&c, 100, McDist::Normal, 6.0, &j, 1, x).is_err());
        let jp = JumpSpec { prob: pr, ..Default::default() };
        check!(simulate_factor_returns(&c, 100, McDist::Jump, 6.0, &jp, 1, 1.0).is_err());
        let mut jm = JumpSpec { prob: 0.01, ..Default::default() };
        jm.mean.insert("FX:EUR".into(), x);
        check!(simulate_factor_returns_jitter(&c, 100, McDist::Jump, 6.0, &jm, 1, 1.0).is_err());
        let mut js = JumpSpec { prob: 0.01, ..Default::default() };
        js.stdev.insert("FX:EUR".into(), if x.is_nan() { x } else { -1.0 });
        check!(simulate_factor_returns(&c, 100, McDist::Jump, 6.0, &js, 1, 1.0).is_err());
        let o = MonteCarloOptions { n_scenarios: 500, alpha: a, ..Default::default() };
        check!(monte_carlo_var(&book(), &market(), &c, &o).is_err());
        let o = MonteCarloOptions { n_scenarios: 500, n_boot: 1, ..Default::default() };
        check!(monte_carlo_var(&book(), &market(), &c, &o).is_err());
        let o = MonteCarloOptions { n_scenarios: 500, ..Default::default() };
        check!(monte_carlo_var(&book(), &market(), &cp, &o).is_err());
        check!(monte_carlo_var(&Book::empty(), &market(), &c, &o).is_err());
        let p = pnl(100);
        check!(var_standard_error(&poison(p.clone(), i, x), 0.01).is_err());
        check!(var_standard_error(&p, a).is_err());
        check!(var_standard_error(&p[..5], 0.01).is_err());
        check!(var_standard_error_bootstrap(&poison(p.clone(), i, x), 0.01, 10, 1).is_err());
        check!(var_standard_error_bootstrap(&p, a, 10, 1).is_err());
        check!(var_standard_error_bootstrap(&p, 0.01, 1, 1).is_err());
    }

    #[test]
    fn linear_algebra_and_returns(x in bad(), l in bad_or_neg(), i in 0usize..1000) {
        let mp = mat2_poison(i, x);
        check!(Matrix::from_rows(&[vec![1.0], vec![1.0, 2.0]]).is_err());
        check!(mp.cholesky().is_err());
        check!(mp.cholesky_with_jitter(8).is_err());
        check!(mp.cholesky_repair(PsdRepair::None).is_err());
        check!(cholesky_repair(&mp, PsdRepair::Clip).is_err());
        check!(check_symmetric(&mp).is_err());
        check!(symmetric_eigen(&mp).is_err());
        check!(clip_psd(&mp).is_err());
        check!(higham_nearest(&mp).is_err());
        check!(mp.solve_spd(&[1.0, 1.0]).is_err());
        check!(mat2().solve_spd(&[x, 1.0]).is_err());
        check!(mp.quad_form(&W).is_err());
        check!(mat2().quad_form(&[x, 1.0]).is_err());
        check!(!mp.is_symmetric(1e-12));
        check!(!mp.is_symmetric_rel(1e-12));
        check!(!mp.all_finite());
        check!(!mp.max_abs().is_finite() || x.is_nan());
        let y = mat2().matvec(&[x, 1.0]).unwrap();
        check!(y.iter().all(|v| !v.is_finite()));
        check!(ReturnsMatrix::new(vec!["FX:EUR".into()], Matrix::zeros(3, 2)).is_err());
        check!(sample_cov(&rets_poison(50, i, x)).is_err());
        check!(sample_cov(&rets(1)).is_err());
        check!(ewma_cov(&rets(50), l).is_err());
        check!(ewma_cov(&rets_poison(50, i, x), 0.94).is_err());
        check!(ewma_volatility(&rets(50), l).is_err());
        check!(ewma_volatility(&rets_poison(50, i, x), 0.94).is_err());
        check!(ewma_volatility(&rets(1), 0.94).is_err());
        check!(validate_returns(&rets_poison(100, i, x), &factors(), 60).is_err());
        check!(validate_returns(&rets(10), &factors(), 60).is_err());
        check!(flag_peg_factors(&rets(50), x).is_err());
        check!(flag_peg_factors(&rets(50), l).is_err());
        check!(rets(50).select(&["FX:XXX".to_string()]).is_err());
        check!(cov().select(&["FX:XXX".to_string()]).is_err());
        check!(rets(5).column_index("FX:XXX").is_none());
    }

    #[test]
    fn book_and_stress(x in bad(), neg in -1e6..-1e-9f64, i in 0usize..1000) {
        check!(Market::new([("EUR", x)], [("USD", 0.05)]).is_err());
        check!(Market::new([("EUR", neg)], [("USD", 0.05)]).is_err());
        check!(Market::new([("EUR", 1.1)], [("USD", x)]).is_err());
        check!(Book::new(vec![Position::Spot(SpotPosition::new("EURUSD", x, None))], "USD").is_err());
        check!(Book::new(vec![Position::Spot(SpotPosition::new("EURUSD", 1.0, Some(x)))], "USD").is_err());
        check!(Book::new(vec![Position::Forward(ForwardPosition::new("EURUSD", 1.0, x, None))], "USD").is_err());
        check!(Book::new(vec![Position::Forward(ForwardPosition::new("EURUSD", 1.0, neg, None))], "USD").is_err());
        check!(Book::new(vec![Position::Forward(ForwardPosition::new("EURUSD", 1.0, 0.5, Some(x)))], "USD").is_err());
        check!(Book::new(vec![Position::Cash(CashPosition::new("EUR", x))], "USD").is_err());
        let mut b = book();
        check!(b.add(Position::Spot(SpotPosition::new("EURUSD", x, None))).is_err());
        check!(split_pair("EUR").is_err());
        check!(fx_factor("USD").is_err());
        check!(market().spot("XXX").is_err());
        check!(market().rate("XXX").is_err());
        check!(market().cross("EURXXX").is_err());
        check!(market().forward("EURUSD", x).is_err());
        check!(market().forward("EURUSD", neg).is_err());
        let cb = CompiledBook::new(&book(), &market()).unwrap();
        let k = cb.factors().len();
        check!(cb.value_usd(&poison(vec![0.0; k], i, x)).is_err());
        check!(cb.pnl(&poison(vec![0.0; k], i, x)).is_err());
        check!(cb.pnl(&[]).is_err());
        let mut m = HashMap::new();
        m.insert("FX:EUR".to_string(), x);
        check!(cb.pnl_map(&m).is_err());
        check!(cb.pnl_scenarios(&rets_poison(20, i, x)).is_err());
        check!(cb.linear_exposures(x).is_err());
        check!(cb.linear_exposures(neg).is_err());
        check!(CompiledBook::new(&Book::empty(), &market()).is_err());
        check!(simple_to_log(x).is_err());
        check!(simple_to_log(-1.0).is_err());
        check!(usd_broad_move(&["EUR".to_string()], x).is_err());
        check!(usd_broad_move(&[], 0.05).is_err());
        check!(peg_break_scenario("HKD", x, &HashMap::new()).is_err());
        let mut cg = HashMap::new();
        cg.insert("CNY".to_string(), x);
        check!(peg_break_scenario("HKD", -0.2, &cg).is_err());
        let mut sc = historical_scenarios();
        sc.insert("bad".into(), Scenario { name: "bad".into(), shocks: m.clone(), description: String::new() });
        check!(run_stress(&book(), &market(), &sc).is_err());
        check!(run_stress(&Book::empty(), &market(), &historical_scenarios()).is_err());
        check!(reverse_stress_linear(&W, &mat2(), x).is_err());
        check!(reverse_stress_linear(&W, &mat2(), neg).is_err());
        check!(reverse_stress_linear(&[x, 1.0], &mat2(), 3.0).is_err());
        check!(reverse_stress_linear(&W, &mat2_poison(i, x), 3.0).is_err());
        check!(reverse_stress_for_loss(&W, &mat2(), x).is_err());
        check!(reverse_stress_for_loss(&W, &mat2(), neg).is_err());
        check!(reverse_stress_for_loss(&[x, 1.0], &mat2(), 1e3).is_err());
        check!(reverse_stress_numerical(&W, &mat2(), x, 1).is_err());
        check!(reverse_stress_numerical(&[x, 1.0], &mat2(), 3.0, 1).is_err());
        check!(reverse_stress_numerical(&W, &mat2_poison(i, x), 3.0, 1).is_err());
    }

    #[test]
    fn special_functions(x in bad(), neg in -1e3..-1e-9f64, p in bad_prob()) {
        check!(inv_norm_cdf(p).is_err());
        check!(inv_norm_cdf(0.0).is_err());
        check!(ln_gamma(x).is_err());
        check!(ln_gamma(neg).is_err());
        check!(reg_lower_gamma(x, 1.0).is_err());
        check!(reg_lower_gamma(1.0, neg).is_err());
        check!(reg_upper_gamma(neg, 1.0).is_err());
        check!(reg_upper_gamma(1.0, x).is_err());
        check!(reg_inc_beta(x, 1.0, 0.5).is_err());
        check!(reg_inc_beta(1.0, neg, 0.5).is_err());
        check!(reg_inc_beta(1.0, 1.0, p).is_err());
        check!(chi2_sf(1.0, x).is_err());
        check!(chi2_sf(1.0, neg).is_err());
        check!(chi2_sf(f64::NAN, 1.0).is_err());
        check!(binom_cdf(3, 10, p).is_err());
        check!(binom_cdf(3, -1, 0.5).is_err());
        check!(student_t_pdf(0.0, x).is_err());
        check!(student_t_pdf(f64::NAN, 5.0).is_err());
        check!(student_t_cdf(0.0, neg).is_err());
        check!(student_t_cdf(f64::NAN, 5.0).is_err());
        check!(student_t_quantile(p, 5.0).is_err());
        check!(student_t_quantile(0.3, x).is_err());
        check!(moments(&poison(pnl(10), 3, x)).is_err());
        check!(moments(&[1.0]).is_err());
        check!(sample_std(&poison(pnl(10), 3, x)).is_err());
        check!(sample_std(&[1.0]).is_err());
        check!(norm_pdf(f64::NAN).is_nan());
        check!(norm_cdf(f64::NAN).is_nan());
        let mut rng = Rng::new(7);
        check!(rng.gamma(x).is_nan());
        check!(rng.chi_square(neg).is_nan());
        let e = FxVarError::invalid("x");
        check!(matches!(e, FxVarError::Invalid(_)));
        check!(matches!(FxVarError::numerical("y"), FxVarError::Numerical(_)));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// The allocation-free kernels reject poisoned input with a typed
    /// error and never panic.
    #[test]
    fn latency_kernels_reject_poison(x in bad(), a in bad_alpha(), i in 0usize..1000) {
        let p = pnl(50);
        let mut ws = vec![0.0; 50];
        check!(empirical_var_es_into(&poison(p.clone(), i, x), 0.05, &mut ws) == Err(KernelError::NonFinite));
        check!(empirical_var_into(&poison(p.clone(), i, x), 0.05, &mut ws) == Err(KernelError::NonFinite));
        check!(empirical_var_es_into(&p, a, &mut ws) == Err(KernelError::BadAlpha));
        check!(empirical_var_into(&p, a, &mut ws) == Err(KernelError::BadAlpha));
        check!(empirical_var_es_into(&p, 0.05, &mut ws[..10]) == Err(KernelError::ScratchTooSmall));
        check!(empirical_var_es_into(&[], 0.05, &mut ws) == Err(KernelError::EmptySample));
        let cb = CompiledBook::new(&book(), &market()).unwrap();
        let k = cb.factors().len();
        check!(cb.pnl_row(&poison(vec![0.001; k], i, x)) == Err(KernelError::NonFinite));
        check!(cb.pnl_row(&vec![0.001; k + 1]) == Err(KernelError::LengthMismatch));
        let r = rets_poison(20, i, x);
        let mut out = vec![0.0; 20];
        check!(cb.pnl_aligned_into(&r.data, &mut out) == Err(KernelError::NonFinite));
        check!(cb.pnl_aligned_into(&rets(20).data, &mut out[..19]) == Err(KernelError::LengthMismatch));
        let mut m = Matrix::zeros(2, 2);
        m.as_mut_slice()[i % 4] = x;
        check!(!m.all_finite());
        let opts = MonteCarloOptions { alpha: a, n_scenarios: 100, ..Default::default() };
        check!(monte_carlo_var_with_threads(&book(), &market(), &cov(), &opts, 3).is_err());
        check!(var_standard_error_bootstrap_with_threads(&poison(p.clone(), i, x), 0.05, 50, 1, 4).is_err());
        check!(var_standard_error_bootstrap_with_threads(&p, a, 50, 1, 4).is_err());
    }
}

/// Every `pub fn` in `src/` must be named in this file (or listed as an
/// accessor / constructor without a float input to poison).
#[test]
fn registry_covers_every_public_fn() {
    let srcs = [
        include_str!("../src/backtest.rs"),
        include_str!("../src/book.rs"),
        include_str!("../src/expected_shortfall.rs"),
        include_str!("../src/historical.rs"),
        include_str!("../src/lib.rs"),
        include_str!("../src/matrix.rs"),
        include_str!("../src/monte_carlo.rs"),
        include_str!("../src/parametric.rs"),
        include_str!("../src/psd.rs"),
        include_str!("../src/returns.rs"),
        include_str!("../src/rng.rs"),
        include_str!("../src/stats.rs"),
        include_str!("../src/stress.rs"),
        include_str!("../src/tail.rs"),
    ];
    let me = include_str!("fuzz_public_api.rs");
    let exempt = [
        "zeros", "identity", "rows", "cols", "get", "set", "row", "row_mut", "as_slice", "new",
        "empty", "base", "positions", "is_empty", "currencies", "factors", "value0_usd",
        "n_obs", "n_factors", "ir_factor", "next_u64", "uniform", "uniform_open", "normal",
    ];
    let mut missing = Vec::new();
    for src in srcs {
        for line in src.lines() {
            let t = line.trim_start();
            if let Some(rest) = t.strip_prefix("pub fn ") {
                let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                let used = me.contains(&format!("{name}(")) || me.contains(&format!(".{name}("));
                if !used && !exempt.contains(&name.as_str()) {
                    missing.push(name);
                }
            }
        }
    }
    assert!(missing.is_empty(), "public fns without a fuzz entry: {missing:?}");
}
