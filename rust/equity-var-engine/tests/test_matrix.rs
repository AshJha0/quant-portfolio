//! Matrix, Cholesky (clean + jitter fallback), covariance estimators.

use eq_var_engine::matrix::{covariance_from_vols, ewma_covariance, sample_covariance, Matrix};
use eq_var_engine::EqVarError;

fn spd3() -> Matrix {
    // SPD matrix built as A = B B^T + I from a fixed B.
    Matrix::from_vec(3, 3, vec![4.0, 2.0, 0.6, 2.0, 5.0, 1.5, 0.6, 1.5, 3.0]).unwrap()
}

fn reconstruct(l: &Matrix) -> Matrix {
    let n = l.rows();
    let mut a = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += l.get(i, k) * l.get(j, k);
            }
            a.set(i, j, s);
        }
    }
    a
}

fn max_abs_diff(a: &Matrix, b: &Matrix) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice().iter())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn cholesky_reconstructs_spd() {
    let a = spd3();
    let l = a.cholesky().unwrap();
    assert!(max_abs_diff(&reconstruct(&l), &a) < 1e-12);
    assert_eq!(l.get(0, 1), 0.0);
    assert_eq!(l.get(0, 2), 0.0);
    assert_eq!(l.get(1, 2), 0.0);
}

#[test]
fn cholesky_jitter_path_on_singular_matrix() {
    // Perfectly correlated 2x2 — exactly singular, plain Cholesky must fail
    // and the diagonal-jitter fallback must engage.
    let a = Matrix::from_vec(2, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    assert!(a.cholesky().is_err());
    let l = a.cholesky_jitter(1e-10, 12).unwrap();
    assert!(max_abs_diff(&reconstruct(&l), &a) < 1e-6);
}

#[test]
fn cholesky_jitter_path_on_zero_variance_asset() {
    let a = Matrix::from_vec(2, 2, vec![0.01, 0.0, 0.0, 0.0]).unwrap();
    let l = a.cholesky_jitter(1e-10, 12).unwrap();
    assert!(max_abs_diff(&reconstruct(&l), &a) < 1e-8);
}

#[test]
fn cholesky_rejects_bad_input() {
    let non_square = Matrix::from_vec(2, 3, vec![1.0; 6]).unwrap();
    assert!(matches!(non_square.cholesky(), Err(EqVarError::InvalidInput(_))));
    let asym = Matrix::from_vec(2, 2, vec![1.0, 0.5, -0.5, 1.0]).unwrap();
    assert!(matches!(asym.cholesky(), Err(EqVarError::InvalidInput(_))));
    // Badly indefinite: jitter cannot rescue a huge negative eigenvalue.
    let bad = Matrix::from_vec(2, 2, vec![1.0, 1e8, 1e8, 1.0]).unwrap();
    assert!(matches!(bad.cholesky_jitter(1e-10, 12), Err(EqVarError::Numerical(_))));
}

#[test]
fn matvec_known_product_and_shape_check() {
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let x = [1.0, 0.5, -1.0];
    let y = a.matvec(&x).unwrap();
    assert_eq!(y.len(), 2);
    assert_eq!(y[0], -1.0);
    assert_eq!(y[1], 0.5);
    assert!(a.matvec(&[1.0, 2.0]).is_err());
}

#[test]
fn sample_covariance_hand_computed_2x2() {
    let r = Matrix::from_vec(3, 2, vec![0.01, 0.02, -0.01, 0.00, 0.03, 0.01]).unwrap();
    let s = sample_covariance(&r).unwrap();
    assert!((s.get(0, 0) - 4.0e-4).abs() < 1e-18);
    assert!((s.get(1, 1) - 1.0e-4).abs() < 1e-18);
    assert!((s.get(0, 1) - 1.0e-4).abs() < 1e-18);
    assert_eq!(s.get(0, 1), s.get(1, 0));
    assert!(sample_covariance(&Matrix::zeros(1, 2)).is_err());
}

#[test]
fn ewma_covariance_single_point_limit_and_validation() {
    let r = Matrix::from_vec(3, 2, vec![0.01, 0.02, -0.01, 0.00, 0.03, 0.01]).unwrap();
    let s = sample_covariance(&r).unwrap();
    let e = ewma_covariance(&r, 0.999999).unwrap();
    assert!(max_abs_diff(&e, &s) < 1e-8);
    assert!(ewma_covariance(&r, 1.0).is_err());
    assert!(ewma_covariance(&r, 0.0).is_err());
}

#[test]
fn covariance_from_vols_known() {
    let vols = [0.01, 0.02];
    let corr = Matrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 1.0]).unwrap();
    let cov = covariance_from_vols(&vols, &corr).unwrap();
    assert_eq!(cov.get(0, 0), 1e-4);
    assert_eq!(cov.get(0, 1), 1e-4);
    assert_eq!(cov.get(1, 1), 4e-4);
}
