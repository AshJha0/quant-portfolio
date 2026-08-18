//! Monte Carlo VaR: convergence to the parametric closed form within 3 SE,
//! Student-t tail fattening, bitwise seed determinism, moment matching.

use eq_var_engine::expected_shortfall::normal_es;
use eq_var_engine::matrix::{covariance_from_vols, sample_covariance, Matrix};
use eq_var_engine::monte_carlo::*;
use eq_var_engine::parametric::{parametric_var, portfolio_sigma};
use eq_var_engine::rng::Rng;
use eq_var_engine::stats::normal_ppf;
use eq_var_engine::TailModel;

fn demo_cov() -> Matrix {
    let vols = [0.010, 0.015, 0.020];
    let corr = Matrix::from_vec(3, 3, vec![1.0, 0.5, 0.25, 0.5, 1.0, 0.3, 0.25, 0.3, 1.0]).unwrap();
    covariance_from_vols(&vols, &corr).unwrap()
}

const EXPOSURES: [f64; 3] = [1.0e6, -5.0e5, 2.0e5];

#[test]
fn bitwise_seed_determinism() {
    let cov = demo_cov();
    let a = monte_carlo_var(&EXPOSURES, &cov, 0.01, 20_000, TailModel::Normal, 42).unwrap();
    let b = monte_carlo_var(&EXPOSURES, &cov, 0.01, 20_000, TailModel::Normal, 42).unwrap();
    assert_eq!(a, b); // bitwise, not approximate
    let ea = monte_carlo_es(&EXPOSURES, &cov, 0.01, 20_000, TailModel::Normal, 42).unwrap();
    let eb = monte_carlo_es(&EXPOSURES, &cov, 0.01, 20_000, TailModel::Normal, 42).unwrap();
    assert_eq!(ea, eb);
    let c = monte_carlo_var(&EXPOSURES, &cov, 0.01, 20_000, TailModel::Normal, 43).unwrap();
    assert_ne!(a, c); // a different seed must move the estimate
}

#[test]
fn normal_converges_to_parametric_within_3se() {
    let cov = demo_cov();
    let exact_var = parametric_var(&EXPOSURES, &cov, 0.01, TailModel::Normal).unwrap();
    let exact_es = normal_es(portfolio_sigma(&EXPOSURES, &cov).unwrap(), 0.01, 0.0).unwrap();
    let pnl = monte_carlo_pnl(&EXPOSURES, &cov, 200_000, TailModel::Normal, 7).unwrap();
    let mc_var = monte_carlo_var(&EXPOSURES, &cov, 0.01, 200_000, TailModel::Normal, 7).unwrap();
    let mc_es = monte_carlo_es(&EXPOSURES, &cov, 0.01, 200_000, TailModel::Normal, 7).unwrap();
    let se = var_order_statistic_se(&pnl, 0.01).unwrap();
    assert!(se > 0.0);
    assert!(se < 0.02 * exact_var);
    assert!((mc_var - exact_var).abs() < 3.0 * se, "mc={mc_var} exact={exact_var} se={se}");
    assert!((mc_es - exact_es).abs() < 4.0 * se);
}

#[test]
fn student_t_fatter_than_normal_at_99_and_converges_to_closed_form() {
    let cov = demo_cov();
    let n_var = monte_carlo_var(&EXPOSURES, &cov, 0.01, 200_000, TailModel::Normal, 11).unwrap();
    let n_es = monte_carlo_es(&EXPOSURES, &cov, 0.01, 200_000, TailModel::Normal, 11).unwrap();
    let t_var = monte_carlo_var(
        &EXPOSURES,
        &cov,
        0.01,
        200_000,
        TailModel::StudentT { df: 5.0 },
        11,
    )
    .unwrap();
    let t_es = monte_carlo_es(
        &EXPOSURES,
        &cov,
        0.01,
        200_000,
        TailModel::StudentT { df: 5.0 },
        11,
    )
    .unwrap();
    assert!(t_var > n_var);
    assert!(t_es > n_es);
    let exact_t = parametric_var(&EXPOSURES, &cov, 0.01, TailModel::StudentT { df: 5.0 }).unwrap();
    let pnl_t = monte_carlo_pnl(&EXPOSURES, &cov, 200_000, TailModel::StudentT { df: 5.0 }, 11)
        .unwrap();
    let se_t = var_order_statistic_se(&pnl_t, 0.01).unwrap();
    assert!((t_var - exact_t).abs() < 3.0 * se_t);
}

#[test]
fn es_dominates_var() {
    let cov = demo_cov();
    for seed in [0u64, 1, 2] {
        let var = monte_carlo_var(&EXPOSURES, &cov, 0.025, 50_000, TailModel::Normal, seed).unwrap();
        let es = monte_carlo_es(&EXPOSURES, &cov, 0.025, 50_000, TailModel::Normal, seed).unwrap();
        assert!(es >= var, "seed={seed}");
    }
}

#[test]
fn simulate_factor_returns_matches_target_covariance_normal_and_t() {
    let cov = demo_cov();
    for tail in [TailModel::Normal, TailModel::StudentT { df: 6.0 }] {
        let scen = simulate_factor_returns(&cov, 200_000, tail, 3).unwrap();
        let hat = sample_covariance(&scen).unwrap();
        for i in 0..3 {
            assert!((hat.get(i, i) / cov.get(i, i) - 1.0).abs() < 0.05, "i={i}");
            for j in 0..3 {
                let tol = 0.10 * (cov.get(i, i) * cov.get(j, j)).sqrt();
                assert!((hat.get(i, j) - cov.get(i, j)).abs() < tol);
            }
        }
    }
}

#[test]
fn rng_uniform_in_open_interval_and_gaussian_moments() {
    let mut rng = Rng::new(123);
    let (mut s, mut s2) = (0.0, 0.0);
    for _ in 0..100_000 {
        let u = rng.uniform();
        assert!(u > 0.0 && u < 1.0);
        let g = normal_ppf(u).unwrap();
        s += g;
        s2 += g * g;
    }
    assert!((s / 1e5).abs() < 0.015);
    assert!((s2 / 1e5 - 1.0).abs() < 0.02);
}

#[test]
fn rng_chi_squared_mean_and_variance() {
    let mut rng = Rng::new(99);
    let df = 6.0;
    let (mut s, mut s2) = (0.0, 0.0);
    let n = 50_000;
    for _ in 0..n {
        let x = rng.chi_square(df);
        assert!(x > 0.0);
        s += x;
        s2 += x * x;
    }
    let m = s / n as f64;
    assert!((m - df).abs() < 0.07);
    assert!((s2 / n as f64 - m * m - 2.0 * df).abs() < 0.5);
}

#[test]
fn monte_carlo_validation() {
    let cov = demo_cov();
    assert!(monte_carlo_var(&[], &cov, 0.01, 1000, TailModel::Normal, 0).is_err());
    assert!(monte_carlo_var(&[1.0], &cov, 0.01, 1000, TailModel::Normal, 0).is_err());
    assert!(monte_carlo_var(&EXPOSURES, &cov, 0.6, 1000, TailModel::Normal, 0).is_err());
    assert!(monte_carlo_var(&EXPOSURES, &cov, 0.01, 1000, TailModel::StudentT { df: 2.0 }, 0)
        .is_err());
    assert!(simulate_factor_returns(&cov, 0, TailModel::Normal, 0).is_err());
    let tiny_pnl = vec![0.0; 50];
    assert!(var_order_statistic_se(&tiny_pnl, 0.01).is_err());
}
