#![allow(clippy::needless_range_loop)]
//! Fuzz walk over every public entry point (S12, H-1).
//!
//! Each property takes valid baseline arguments, replaces one argument (or
//! one element of a slice) with NaN, +inf or -inf, or uses an empty /
//! length-1 / negative input where that is invalid, and asserts `Err`.
//! A plausible number from bad input is a failure.
//!
//! Total functions (`erf`, `erfc`, `normal_cdf`, `normal_pdf`,
//! `cornish_fisher_z`, `Matrix::matvec`, `Rng::gamma`, `Rng::chi_square`)
//! return a float rather than a `Result`: for them the property is that NaN
//! in gives NaN out. `registry_covers_every_public_fn` fails when a new
//! `pub fn` appears in `src/` without being named in this file.
#![allow(deprecated)]

use eq_var_engine::backtest::*;
use eq_var_engine::expected_shortfall::*;
use eq_var_engine::historical::*;
use eq_var_engine::matrix::*;
use eq_var_engine::monte_carlo::*;
use eq_var_engine::parametric::*;
use eq_var_engine::psd::*;
use eq_var_engine::rng::Rng;
use eq_var_engine::stats::*;
use eq_var_engine::tail::*;
use eq_var_engine::TailModel;
use proptest::prelude::*;

/// `prop_assert!` with the expression as a literal message (struct
/// literals contain braces, which the default message would parse).
macro_rules! check {
    ($e:expr) => {
        prop_assert!($e, "{}", stringify!($e))
    };
}

fn bad() -> impl Strategy<Value = f64> {
    prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)]
}

fn bad_or_neg() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(f64::NAN),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        -1e6..-1e-12
    ]
}

fn bad_alpha() -> impl Strategy<Value = f64> {
    prop_oneof![bad(), -1.0..=0.0, 0.5..2.0]
}

fn pnl(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 1000.0 * (0.37 * i as f64 + 0.1).sin())
        .collect()
}

fn poison(mut v: Vec<f64>, i: usize, x: f64) -> Vec<f64> {
    let k = i % v.len();
    v[k] = x;
    v
}

fn cov2() -> Matrix {
    Matrix::from_vec(2, 2, vec![1e-4, 2e-5, 2e-5, 4e-4]).unwrap()
}

fn cov2_poison(i: usize, x: f64) -> Matrix {
    let mut d = vec![1e-4, 2e-5, 2e-5, 4e-4];
    d[i % 4] = x;
    if i % 4 == 1 || i % 4 == 2 {
        d[1] = x;
        d[2] = x;
    }
    Matrix::from_vec(2, 2, d).unwrap()
}

fn panel(t: usize) -> Matrix {
    let mut m = Matrix::zeros(t, 2);
    for i in 0..t {
        m.set(i, 0, 0.01 * (0.3 * i as f64).sin());
        m.set(i, 1, 0.02 * (0.7 * i as f64 + 1.0).cos());
    }
    m
}

fn panel_poison(t: usize, i: usize, x: f64) -> Matrix {
    let mut m = panel(t);
    m.set(i % t, i % 2, x);
    m
}

const W: [f64; 2] = [1.0e6, -5.0e5];

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn historical_family(x in bad(), a in bad_alpha(), l in bad_or_neg(), i in 0usize..1000) {
        let p = pnl(120);
        check!(historical_var(&poison(p.clone(), i, x), 0.01).is_err());
        check!(historical_var(&p, a).is_err());
        check!(historical_var(&[], 0.01).is_err());
        check!(historical_var(&[1.0], 0.01).is_err());
        check!(age_weighted_var(&poison(p.clone(), i, x), 0.01, 0.97).is_err());
        check!(age_weighted_var(&p, a, 0.97).is_err());
        check!(age_weighted_var(&p, 0.01, l).is_err());
        check!(brw_weights(10, l).is_err());
        check!(brw_weights(0, 0.9).is_err());
        check!(ewma_volatility(&poison(p.clone(), i, x), 0.94).is_err());
        check!(ewma_volatility(&p, l).is_err());
        check!(ewma_volatility(&[1.0], 0.94).is_err());
        check!(ewma_volatility_forecast(&poison(p.clone(), i, x), 0.94).is_err());
        check!(ewma_volatility_forecast(&[], 0.94).is_err());
        check!(filtered_historical_var(&poison(p.clone(), i, x), 0.01, 0.94).is_err());
        check!(filtered_historical_var(&p, a, 0.94).is_err());
        check!(fhs_factor_scenarios(&panel_poison(80, i, x), 0.94).is_err());
        check!(fhs_factor_scenarios(&panel(80), l).is_err());
        check!(fhs_factor_scenarios(&panel(1), 0.94).is_err());
        check!(filtered_historical_var_factors(&panel_poison(80, i, x), &W, 0.01, 0.94).is_err());
        check!(filtered_historical_var_factors(&panel(80), &[x, 1.0], 0.01, 0.94).is_err());
        check!(filtered_historical_var_factors(&panel(80), &W, a, 0.94).is_err());
        check!(filtered_historical_var_factors(&panel(80), &[], 0.01, 0.94).is_err());
        check!(scale_var_sqrt_time(x, 10).is_err());
        check!(scale_var_sqrt_time(1.0, 0).is_err());
        check!(overlapping_horizon_pnl(&poison(p.clone(), i, x), 5).is_err());
        check!(overlapping_horizon_pnl(&p, 0).is_err());
        check!(overlapping_horizon_pnl(&[], 1).is_err());
        check!(linear_quantile(&poison(p.clone(), i, x), 0.5).is_err());
        check!(linear_quantile(&p, x).is_err());
        check!(linear_quantile(&[], 0.5).is_err());
    }

    #[test]
    fn tail_estimators(x in bad(), a in bad_alpha(), w in bad_or_neg(), i in 0usize..1000) {
        let p = pnl(50);
        check!(empirical_var(&poison(p.clone(), i, x), 0.01).is_err());
        check!(empirical_var(&p, a).is_err());
        check!(empirical_var(&[], 0.01).is_err());
        check!(empirical_var_es(&poison(p.clone(), i, x), 0.01).is_err());
        check!(empirical_var_es(&p, a).is_err());
        check!(weighted_var_es(&poison(p.clone(), i, x), &vec![1.0; 50], 0.05).is_err());
        check!(weighted_var_es(&p, &poison(vec![1.0; 50], i, w), 0.05).is_err());
        check!(weighted_var_es(&p, &vec![0.0; 50], 0.05).is_err());
        check!(weighted_var_es(&p, &[1.0], 0.05).is_err());
        check!(weighted_var_es(&p, &vec![1.0; 50], a).is_err());
        check!(tail_rank(0, 0.01).is_err());
        check!(tail_rank(10, a).is_err());
        check!(validate_alpha(a).is_err());
        check!(expected_shortfall(&poison(p.clone(), i, x), 0.025).is_err());
        check!(expected_shortfall(&p, a).is_err());
        check!(expected_shortfall(&[1.0], 0.025).is_err());
    }

    #[test]
    fn closed_forms(x in bad(), a in bad_alpha(), neg in -1e6..-1e-9f64, df in prop_oneof![bad(), -5.0..=2.0f64], i in 0usize..1000) {
        check!(normal_es(x, 0.01, 0.0).is_err());
        check!(normal_es(neg, 0.01, 0.0).is_err());
        check!(normal_es(1.0, a, 0.0).is_err());
        check!(normal_es(1.0, 0.01, x).is_err());
        check!(student_t_es(x, 0.01, 5.0, 0.0).is_err());
        check!(student_t_es(1.0, 0.01, df, 0.0).is_err());
        check!(student_t_es(1.0, a, 5.0, 0.0).is_err());
        check!(student_t_es(1.0, 0.01, 5.0, x).is_err());
        let c = cov2();
        let cp = cov2_poison(i, x);
        check!(portfolio_sigma(&[x, 1.0], &c).is_err());
        check!(portfolio_sigma(&W, &cp).is_err());
        check!(portfolio_sigma(&[], &c).is_err());
        check!(parametric_var(&W, &cp, 0.01, TailModel::Normal).is_err());
        check!(parametric_var(&W, &c, a, TailModel::Normal).is_err());
        check!(parametric_var(&W, &c, 0.01, TailModel::StudentT { df }).is_err());
        check!(parametric_var_full(&W, &c, 0.01, TailModel::Normal, x, 1).is_err());
        check!(parametric_var_full(&W, &c, 0.01, TailModel::Normal, 0.0, 0).is_err());
        check!(parametric_es(&W, &cp, 0.025, TailModel::Normal).is_err());
        check!(parametric_es(&[x, 1.0], &c, 0.025, TailModel::Normal).is_err());
        check!(parametric_es_full(&W, &c, a, TailModel::Normal, 0.0, 1).is_err());
        check!(parametric_es_full(&W, &c, 0.025, TailModel::Normal, x, 1).is_err());
        check!(tail_quantile(a, TailModel::Normal).is_err());
        check!(tail_quantile(0.01, TailModel::StudentT { df }).is_err());
        check!(cornish_fisher_valid(x, 0.0, 0.01).is_err());
        check!(cornish_fisher_valid(0.0, x, 0.01).is_err());
        check!(cornish_fisher_valid(0.0, 0.0, a).is_err());
        check!(cornish_fisher_var(x, 0.01, 0.0, 0.0, 0.0, true).is_err());
        check!(cornish_fisher_var(neg, 0.01, 0.0, 0.0, 0.0, true).is_err());
        check!(cornish_fisher_var(1.0, a, 0.0, 0.0, 0.0, true).is_err());
        check!(cornish_fisher_var(1.0, 0.01, x, 0.0, 0.0, false).is_err());
        check!(cornish_fisher_var(1.0, 0.01, 0.0, x, 0.0, false).is_err());
        check!(cornish_fisher_var(1.0, 0.01, 0.0, 0.0, x, false).is_err());
        check!(!cornish_fisher_domain_ok(x, 0.0, 3.5, 10));
    }

    #[test]
    fn backtests(x in bad(), a in bad_alpha(), neg in -100i64..0, i in 0usize..1000) {
        let p = pnl(250);
        check!(exceptions_from_pnl(&poison(p.clone(), i, x), &[100.0]).is_err());
        check!(exceptions_from_pnl(&p, &[x]).is_err());
        check!(exceptions_from_pnl(&p, &[-1.0]).is_err());
        check!(exceptions_from_pnl(&[], &[1.0]).is_err());
        check!(kupiec_pof(&KupiecInput { n_obs: 250, n_exceptions: 5, alpha: a }).is_err());
        check!(kupiec_pof(&KupiecInput { n_obs: neg, n_exceptions: 0, alpha: 0.01 }).is_err());
        check!(kupiec_pof(&KupiecInput { n_obs: 250, n_exceptions: neg, alpha: 0.01 }).is_err());
        check!(kupiec_pof(&KupiecInput { n_obs: 250, n_exceptions: 251, alpha: 0.01 }).is_err());
        check!(kupiec_pof_positional(250, 5, a).is_err());
        check!(basel_traffic_light(&BaselInput { n_obs: 250, n_exceptions: 5, alpha: a }).is_err());
        check!(basel_traffic_light(&BaselInput { n_obs: neg, n_exceptions: 0, alpha: 0.01 }).is_err());
        check!(basel_traffic_light_250(300, 250).is_err());
        check!(christoffersen_cc(&[0, 1, 0, 0], a).is_err());
        check!(christoffersen_independence(&[1]).is_err());
        check!(christoffersen_independence(&[]).is_err());
        check!(acerbi_szekely_z2(&poison(p.clone(), i, x), &[100.0], &[120.0], 0.025).is_err());
        check!(acerbi_szekely_z2(&p, &[x], &[120.0], 0.025).is_err());
        check!(acerbi_szekely_z2(&p, &[100.0], &[x], 0.025).is_err());
        check!(acerbi_szekely_z2(&p, &[100.0], &[120.0], a).is_err());
        check!(acerbi_szekely_z2(&p, &[-1.0], &[120.0], 0.025).is_err());
        check!(acerbi_szekely_z2(&[], &[1.0], &[1.0], 0.025).is_err());
    }

    #[test]
    fn linear_algebra(x in bad(), i in 0usize..1000, l in bad_or_neg()) {
        let cp = cov2_poison(i, x);
        check!(Matrix::from_vec(2, 2, vec![1.0]).is_err());
        check!(cp.cholesky().is_err());
        check!(cp.cholesky_jitter(1e-10, 5).is_err());
        check!(cp.cholesky_repair(PsdRepair::None).is_err());
        check!(cholesky_repair(&cp, PsdRepair::Higham).is_err());
        check!(cholesky_repair(&Matrix::zeros(0, 0), PsdRepair::None).is_err());
        check!(check_symmetric(&cp).is_err());
        check!(symmetric_eigen(&cp).is_err());
        check!(clip_psd(&cp).is_err());
        check!(higham_nearest(&cp).is_err());
        check!(!cp.is_symmetric());
        check!(!cp.all_finite());
        check!(sample_covariance(&panel_poison(30, i, x)).is_err());
        check!(sample_covariance(&panel(1)).is_err());
        check!(ewma_covariance(&panel_poison(30, i, x), 0.94).is_err());
        check!(ewma_covariance(&panel(30), l).is_err());
        check!(covariance_from_vols(&[x, 0.1], &Matrix::identity(2)).is_err());
        check!(covariance_from_vols(&[-0.1, 0.1], &Matrix::identity(2)).is_err());
        check!(covariance_from_vols(&[0.1, 0.1], &cp).is_err());
        let y = cov2().matvec(&[x, 1.0]).unwrap();
        check!(y.iter().all(|v| !v.is_finite()));
    }

    #[test]
    fn monte_carlo(x in bad(), a in bad_alpha(), df in prop_oneof![bad(), -5.0..=2.0f64], i in 0usize..1000) {
        let c = cov2();
        let cp = cov2_poison(i, x);
        let p = pnl(200);
        check!(simulate_factor_returns(&cp, 200, TailModel::Normal, 1).is_err());
        check!(simulate_factor_returns(&c, 0, TailModel::Normal, 1).is_err());
        check!(simulate_factor_returns(&c, 200, TailModel::StudentT { df }, 1).is_err());
        check!(simulate_factor_returns_jitter(&cp, 200, TailModel::Normal, 1).is_err());
        check!(portfolio_pnl(&panel_poison(10, i, x), &W).is_err());
        check!(portfolio_pnl(&panel(10), &[x, 1.0]).is_err());
        check!(portfolio_pnl(&panel(10), &[]).is_err());
        check!(monte_carlo_pnl(&[x, 1.0], &c, 200, TailModel::Normal, 1).is_err());
        check!(monte_carlo_pnl(&W, &cp, 200, TailModel::Normal, 1).is_err());
        check!(monte_carlo_var(&W, &c, a, 200, TailModel::Normal, 1).is_err());
        check!(monte_carlo_var(&W, &c, 0.01, 10, TailModel::Normal, 1).is_err());
        check!(monte_carlo_es(&W, &c, a, 200, TailModel::Normal, 1).is_err());
        check!(monte_carlo_es(&[x, 1.0], &c, 0.025, 200, TailModel::Normal, 1).is_err());
        check!(monte_carlo_run(&W, &c, a, 200, TailModel::Normal, 1, 10).is_err());
        check!(monte_carlo_run(&[x, 1.0], &c, 0.01, 200, TailModel::Normal, 1, 10).is_err());
        check!(monte_carlo_run(&W, &cp, 0.01, 200, TailModel::Normal, 1, 10).is_err());
        check!(monte_carlo_run(&W, &c, 0.01, 200, TailModel::Normal, 1, 1).is_err());
        check!(monte_carlo_run(&[], &c, 0.01, 200, TailModel::Normal, 1, 10).is_err());
        check!(var_bootstrap_se(&poison(p.clone(), i, x), 0.01, 10, 1).is_err());
        check!(var_bootstrap_se(&p, a, 10, 1).is_err());
        check!(var_bootstrap_se(&p[..5], 0.01, 10, 1).is_err());
        check!(var_kde_se(&poison(p.clone(), i, x), 0.01).is_err());
        check!(var_kde_se(&p, a).is_err());
        check!(var_kde_se(&[1.0], 0.01).is_err());
        check!(var_order_statistic_se(&poison(p.clone(), i, x), 0.01).is_err());
        check!(var_order_statistic_se(&p, a).is_err());
    }

    #[test]
    fn special_functions(x in bad(), neg in -1e3..-1e-9f64, p in prop_oneof![bad(), -1.0..-1e-9f64, 1.000_000_001..2.0f64]) {
        check!(binomial_cdf(3, 10, p).is_err());
        check!(chi2_sf(1.0, x).is_err());
        check!(chi2_sf(1.0, neg).is_err());
        check!(chi2_sf(f64::NAN, 1.0).is_err());
        check!(ln_gamma(x).is_err());
        check!(ln_gamma(neg).is_err());
        check!(mean(&poison(pnl(10), 3, x)).is_err());
        check!(mean(&[]).is_err());
        check!(stdev(&poison(pnl(10), 3, x)).is_err());
        check!(stdev(&[1.0]).is_err());
        check!(population_variance(&poison(pnl(10), 3, x)).is_err());
        check!(skewness(&poison(pnl(10), 3, x)).is_err());
        check!(skewness(&[1.0, 2.0]).is_err());
        check!(excess_kurtosis(&poison(pnl(10), 3, x)).is_err());
        check!(excess_kurtosis(&[1.0, 2.0, 3.0]).is_err());
        check!(normal_ppf(p).is_err());
        check!(regularized_gamma_p(x, 1.0).is_err());
        check!(regularized_gamma_p(1.0, x).is_err());
        check!(regularized_gamma_q(neg, 1.0).is_err());
        check!(regularized_gamma_q(1.0, neg).is_err());
        check!(regularized_beta(x, 1.0, 0.5).is_err());
        check!(regularized_beta(1.0, neg, 0.5).is_err());
        check!(regularized_beta(1.0, 1.0, p).is_err());
        check!(student_t_pdf(0.0, x).is_err());
        check!(student_t_pdf(f64::NAN, 5.0).is_err());
        check!(student_t_cdf(0.0, neg).is_err());
        check!(student_t_cdf(f64::NAN, 5.0).is_err());
        check!(student_t_ppf(p, 5.0).is_err());
        check!(student_t_ppf(0.3, x).is_err());
        // Total functions: NaN in, NaN out.
        check!(erf(f64::NAN).is_nan());
        check!(erfc(f64::NAN).is_nan());
        check!(normal_cdf(f64::NAN).is_nan());
        check!(normal_pdf(f64::NAN).is_nan());
        check!(cornish_fisher_z(f64::NAN, 0.1, 0.1).is_nan());
        check!(cornish_fisher_z(1.0, f64::NAN, 0.1).is_nan());
        let mut rng = Rng::new(7);
        check!(rng.gamma(x).is_nan());
        check!(rng.chi_square(neg).is_nan());
    }
}

/// Every `pub fn` in `src/` must be named in this file (or listed as an
/// accessor / constructor without float inputs).
#[test]
fn registry_covers_every_public_fn() {
    let srcs = [
        include_str!("../src/backtest.rs"),
        include_str!("../src/expected_shortfall.rs"),
        include_str!("../src/historical.rs"),
        include_str!("../src/lib.rs"),
        include_str!("../src/matrix.rs"),
        include_str!("../src/monte_carlo.rs"),
        include_str!("../src/parametric.rs"),
        include_str!("../src/psd.rs"),
        include_str!("../src/rng.rs"),
        include_str!("../src/stats.rs"),
        include_str!("../src/tail.rs"),
    ];
    let me = include_str!("fuzz_public_api.rs");
    // Accessors and constructors with no float input to poison.
    let exempt = [
        "zeros",
        "identity",
        "rows",
        "cols",
        "get",
        "set",
        "row",
        "as_slice",
        "as_str",
        "new",
        "next_u64",
        "uniform",
        "standard_normal",
    ];
    let mut missing = Vec::new();
    for src in srcs {
        for line in src.lines() {
            let t = line.trim_start();
            if let Some(rest) = t.strip_prefix("pub fn ") {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                let used = me.contains(&format!("{name}(")) || me.contains(&format!(".{name}("));
                if !used && !exempt.contains(&name.as_str()) {
                    missing.push(name);
                }
            }
        }
    }
    assert!(
        missing.is_empty(),
        "public fns without a fuzz entry: {missing:?}"
    );
}
