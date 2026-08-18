//! Edge cases from the documentation contract: empty inputs, single-asset
//! portfolios, alpha at the boundaries of (0, 0.5), zero-variance assets,
//! degenerate P&L series, and returns/pnl-mapping validation. Each case here
//! is also documented in docs/VALIDATION.md.

use eq_var_engine::expected_shortfall::expected_shortfall;
use eq_var_engine::historical::{historical_var, linear_quantile, MIN_OBS};
use eq_var_engine::matrix::Matrix;
use eq_var_engine::monte_carlo::{monte_carlo_var, portfolio_pnl};
use eq_var_engine::parametric::{parametric_var, portfolio_sigma};
use eq_var_engine::stats::{excess_kurtosis, mean, normal_ppf, skewness};
use eq_var_engine::TailModel;

#[test]
fn empty_inputs_error() {
    let empty: Vec<f64> = vec![];
    let cov = Matrix::from_vec(1, 1, vec![1e-4]).unwrap();
    assert!(historical_var(&empty, 0.01).is_err());
    assert!(expected_shortfall(&empty, 0.01).is_err());
    assert!(linear_quantile(&empty, 0.5).is_err());
    assert!(portfolio_sigma(&empty, &cov).is_err());
    assert!(parametric_var(&empty, &cov, 0.01, TailModel::Normal).is_err());
    assert!(monte_carlo_var(&empty, &cov, 0.01, 1000, TailModel::Normal, 0).is_err());
    assert!(portfolio_pnl(&Matrix::zeros(5, 1), &empty).is_err());
    assert!(mean(&empty).is_err());
}

#[test]
fn single_asset_portfolio_is_exact() {
    let (vol, w0) = (0.02, 1.0e6);
    let cov = Matrix::from_vec(1, 1, vec![vol * vol]).unwrap();
    let w = [w0];
    let sigma = w0 * vol;
    assert!((portfolio_sigma(&w, &cov).unwrap() - sigma).abs() < 1e-9);
    let var = parametric_var(&w, &cov, 0.01, TailModel::Normal).unwrap();
    assert!((var - (-normal_ppf(0.01).unwrap() * sigma)).abs() < 1e-9);
    let mc = monte_carlo_var(&w, &cov, 0.01, 100_000, TailModel::Normal, 5).unwrap();
    let se = eq_var_engine::monte_carlo::var_order_statistic_se(
        &eq_var_engine::monte_carlo::monte_carlo_pnl(&w, &cov, 100_000, TailModel::Normal, 5)
            .unwrap(),
        0.01,
    )
    .unwrap();
    assert!((mc - var).abs() < 3.0 * se);
    // Short single-asset exposure has identical risk (symmetry of the normal).
    let ws = [-w0];
    let var_short = parametric_var(&ws, &cov, 0.01, TailModel::Normal).unwrap();
    assert!((var_short - var).abs() < 1e-9);
}

#[test]
fn alpha_bounds_rejected_everywhere() {
    let pnl: Vec<f64> = (0..100).map(|t| (1.3 * t as f64).sin() * 100.0).collect();
    let cov = Matrix::from_vec(1, 1, vec![1e-4]).unwrap();
    let w = [1.0e6];
    for bad in [0.0, -0.01, 0.5, 0.75, 1.0] {
        assert!(historical_var(&pnl, bad).is_err(), "{bad}");
        assert!(expected_shortfall(&pnl, bad).is_err(), "{bad}");
        assert!(parametric_var(&w, &cov, bad, TailModel::Normal).is_err(), "{bad}");
    }
    assert!(historical_var(&pnl, 0.499).is_ok());
    assert!(historical_var(&pnl, 0.011).is_ok());
}

#[test]
fn zero_variance_asset_handled_throughout() {
    // Asset 2 is riskless: covariance is singular but PSD.
    let cov = Matrix::from_vec(2, 2, vec![4.0e-4, 0.0, 0.0, 0.0]).unwrap();
    let w = [1.0e6, 5.0e5];
    let sigma = portfolio_sigma(&w, &cov).unwrap();
    assert!((sigma - 1.0e6 * 0.02).abs() < 1e-9);
    assert!(cov.cholesky().is_err()); // plain Cholesky fails on the singular matrix
    let l = cov.cholesky_jitter(1e-10, 12).unwrap();
    assert_eq!(l.rows(), 2);
    let mc = monte_carlo_var(&w, &cov, 0.01, 100_000, TailModel::Normal, 17).unwrap();
    assert!((mc - (-normal_ppf(0.01).unwrap() * sigma)).abs() < 0.05 * sigma);
}

#[test]
fn fully_riskless_portfolio() {
    let cov = Matrix::zeros(2, 2);
    let w = [1.0e6, -1.0e6];
    assert_eq!(portfolio_sigma(&w, &cov).unwrap(), 0.0);
    assert_eq!(parametric_var(&w, &cov, 0.01, TailModel::Normal).unwrap(), 0.0);
    assert_eq!(
        eq_var_engine::expected_shortfall::normal_es(0.0, 0.01, 0.0).unwrap(),
        0.0
    );
}

#[test]
fn constant_pnl_series() {
    let flat = vec![25.0; 100];
    assert_eq!(historical_var(&flat, 0.05).unwrap(), -25.0);
    assert_eq!(expected_shortfall(&flat, 0.05).unwrap(), -25.0);
    let zeros = vec![0.0; 100];
    assert_eq!(
        eq_var_engine::historical::filtered_historical_var(&zeros, 0.05, 0.94).unwrap(),
        0.0
    );
    assert_eq!(skewness(&flat).unwrap(), 0.0);
    assert_eq!(excess_kurtosis(&flat).unwrap(), 0.0);
}

#[test]
fn non_finite_pnl_rejected() {
    let mut pnl = vec![1.0; 100];
    pnl[3] = f64::INFINITY;
    assert!(historical_var(&pnl, 0.05).is_err());
    pnl[3] = f64::NAN;
    assert!(expected_shortfall(&pnl, 0.05).is_err());
    assert!(eq_var_engine::historical::age_weighted_var(&pnl, 0.05, 0.98).is_err());
}

#[test]
fn portfolio_pnl_shape_mismatch() {
    let panel = Matrix::from_vec(10, 3, vec![0.01; 30]).unwrap();
    assert!(portfolio_pnl(&panel, &[1.0, 2.0]).is_err());
    let ok = [1.0, 2.0, 3.0];
    let pnl = portfolio_pnl(&panel, &ok).unwrap();
    assert_eq!(pnl.len(), 10);
    assert!((pnl[0] - 0.06).abs() < 1e-15);
}

#[test]
fn min_obs_guard_exact_boundary() {
    let ok: Vec<f64> = (0..MIN_OBS).map(|t| (0.9 * t as f64).sin()).collect();
    assert!(historical_var(&ok, 0.05).is_ok());
    let short_series = &ok[..ok.len() - 1];
    assert!(historical_var(short_series, 0.05).is_err());
}

#[test]
fn deep_tail_alpha_on_small_sample() {
    // alpha far below 1/n: the type-7 quantile interpolates just above the
    // sample minimum (h = (n-1) alpha < 1) — the tail is unresolvable and the
    // estimate is pinned to the worst observed loss, never extrapolated
    // beyond it. Documented failure mode (docs/VALIDATION.md).
    let pnl: Vec<f64> = (0..100)
        .map(|t| 10.0 * (2.3 * t as f64).sin() - if t == 50 { 500.0 } else { 0.0 })
        .collect();
    let v = historical_var(&pnl, 1e-6).unwrap();
    let mut sorted = pnl.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let h = 99.0 * 1e-6;
    let expected = -(sorted[0] + h * (sorted[1] - sorted[0]));
    assert!((v - expected).abs() < 1e-12 * expected.abs());
    assert!(v <= -sorted[0]); // never exceeds the worst observed loss
    assert!(v > -sorted[1]); // ... but stays pinned to it
}
