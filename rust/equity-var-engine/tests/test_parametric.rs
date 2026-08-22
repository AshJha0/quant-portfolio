//! Parametric (variance-covariance) VaR: closed-form identities, Student-t
//! tail behaviour, Cornish-Fisher with validity-domain check.

use eq_var_engine::matrix::Matrix;
use eq_var_engine::parametric::*;
use eq_var_engine::stats::{normal_ppf, student_t_ppf};
use eq_var_engine::TailModel;

#[test]
fn portfolio_sigma_hand_computed_two_asset() {
    let cov = Matrix::from_vec(2, 2, vec![4.0e-4, 1.0e-4, 1.0e-4, 2.5e-4]).unwrap();
    let w = [1.0e6, -2.0e6];
    let expected = (1e12 * 4e-4 - 2.0 * 1e6 * 2e6 * 1e-4 + 4e12 * 2.5e-4_f64).sqrt();
    let got = portfolio_sigma(&w, &cov).unwrap();
    assert!((got - expected).abs() < 1e-12 * expected);
}

#[test]
fn portfolio_sigma_diversification_never_hurts() {
    let cov = Matrix::from_vec(2, 2, vec![4.0e-4, 1.0e-4, 1.0e-4, 2.5e-4]).unwrap();
    let w = [1.0e6, 1.0e6];
    let standalone = 1e6 * 4e-4_f64.sqrt() + 1e6 * 2.5e-4_f64.sqrt();
    assert!(portfolio_sigma(&w, &cov).unwrap() < standalone);
}

#[test]
fn parametric_var_normal_closed_form() {
    let cov = Matrix::from_vec(1, 1, vec![4.0e-4]).unwrap();
    let w = [1.0e6];
    let sigma = 2.0e4;
    let expected_99 = -normal_ppf(0.01).unwrap() * sigma;
    let v99 = parametric_var(&w, &cov, 0.01, TailModel::Normal).unwrap();
    assert!((v99 - expected_99).abs() < 1e-12 * expected_99);
    let expected_95 = 1.6448536269514722 * sigma;
    let v95 = parametric_var(&w, &cov, 0.05, TailModel::Normal).unwrap();
    assert!((v95 - expected_95).abs() < 1e-8);
    let shifted =
        parametric_var_full(&w, &cov, 0.01, TailModel::Normal, 500.0, 1).unwrap();
    assert!((shifted - (expected_99 - 500.0)).abs() < 1e-9);
}

#[test]
fn parametric_var_student_t_fatter_and_variance_matched() {
    let cov = Matrix::from_vec(1, 1, vec![4.0e-4]).unwrap();
    let w = [1.0e6];
    let v_norm = parametric_var(&w, &cov, 0.01, TailModel::Normal).unwrap();
    let v_t4 = parametric_var(&w, &cov, 0.01, TailModel::StudentT { df: 4.0 }).unwrap();
    let v_t100 = parametric_var(&w, &cov, 0.01, TailModel::StudentT { df: 100.0 }).unwrap();
    assert!(v_t4 > v_norm);
    assert!(v_t4 > v_t100);
    assert!((v_t100 - v_norm).abs() < 0.02 * v_norm);
    let expected = -student_t_ppf(0.01, 6.0).unwrap() * (4.0f64 / 6.0).sqrt() * 2.0e4;
    let v_t6 = parametric_var(&w, &cov, 0.01, TailModel::StudentT { df: 6.0 }).unwrap();
    assert!((v_t6 - expected).abs() < 1e-12 * expected.abs());
}

#[test]
fn parametric_var_sqrt_time_horizon_scaling() {
    let cov = Matrix::from_vec(1, 1, vec![4.0e-4]).unwrap();
    let w = [1.0e6];
    let v1 = parametric_var(&w, &cov, 0.01, TailModel::Normal).unwrap();
    let v10 = parametric_var_full(&w, &cov, 0.01, TailModel::Normal, 0.0, 10).unwrap();
    assert!((v10 - v1 * 10f64.sqrt()).abs() < 1e-9);
    assert!(parametric_var_full(&w, &cov, 0.01, TailModel::Normal, 0.0, 0).is_err());
}

#[test]
fn parametric_var_validation() {
    let cov = Matrix::from_vec(1, 1, vec![4.0e-4]).unwrap();
    let w = [1.0e6];
    assert!(parametric_var(&w, &cov, 0.0, TailModel::Normal).is_err());
    assert!(parametric_var(&w, &cov, 0.5, TailModel::Normal).is_err());
    assert!(parametric_var(&w, &cov, 0.01, TailModel::StudentT { df: 2.0 }).is_err());
    assert!(parametric_var(&[], &cov, 0.01, TailModel::Normal).is_err());
    assert!(parametric_var(&[1.0, 2.0], &cov, 0.01, TailModel::Normal).is_err());
}

#[test]
fn cornish_fisher_reduces_to_normal_at_zero_moments() {
    let z = normal_ppf(0.01).unwrap();
    assert!((cornish_fisher_z(z, 0.0, 0.0) - z).abs() < 1e-15);
    let v = cornish_fisher_var(1.0e4, 0.01, 0.0, 0.0, 0.0, true).unwrap();
    assert!((v - (-z * 1.0e4)).abs() < 1e-9);
}

#[test]
fn cornish_fisher_left_skew_and_fat_tails_raise_var() {
    let base = cornish_fisher_var(1.0e4, 0.01, 0.0, 0.0, 0.0, true).unwrap();
    let skewed = cornish_fisher_var(1.0e4, 0.01, -0.2, 0.0, 0.0, true).unwrap();
    let kurt = cornish_fisher_var(1.0e4, 0.01, 0.0, 1.0, 0.0, true).unwrap();
    assert!(skewed > base);
    assert!(kurt > base);
    // Values pinned to the Python reference (eq_var.cornish_fisher_var).
    assert!((skewed - 24583.575119223973).abs() < 1e-6);
    assert!((kurt - 25601.356024614324).abs() < 1e-6);
}

#[test]
fn cornish_fisher_domain_check_rejects_non_monotone_region() {
    assert!(cornish_fisher_domain_ok(0.0, 0.0, 3.5, 2001));
    assert!(cornish_fisher_domain_ok(-0.5, 1.0, 3.5, 2001));
    assert!(!cornish_fisher_domain_ok(3.0, 0.0, 3.5, 2001));
    assert!(cornish_fisher_var(1.0e4, 0.01, 3.0, 0.0, 0.0, true).is_err());
    assert!(cornish_fisher_var(1.0e4, 0.01, 3.0, 0.0, 0.0, false).is_ok());
    assert!(cornish_fisher_var(-1.0, 0.01, 0.0, 0.0, 0.0, true).is_err());
}

#[test]
fn cornish_fisher_domain_check_is_exact_not_grid_resolution_dependent() {
    // Regression for the closed-form rewrite of cornish_fisher_domain_ok.
    // (skew, excess_kurt) placed so the derivative's parabola vertex falls
    // almost exactly between two nodes of the *old* 2001-point grid on
    // [-3.5, 3.5]: the old grid-sampled derivative check reported this as
    // monotone (every sampled node was positive) even though the true
    // minimum of the derivative between those nodes is ~ -1.0e-6
    // (non-monotone).
    let skew = -0.010499946187942602;
    let excess_kurt = 8.000105998830488;
    assert!(!cornish_fisher_domain_ok(skew, excess_kurt, 3.5, 2001));
    assert!(cornish_fisher_var(1.0, 0.01, skew, excess_kurt, 0.0, true).is_err());
}
