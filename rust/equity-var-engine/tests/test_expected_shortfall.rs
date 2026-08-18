//! Expected Shortfall: hand-exact empirical tail integral, the closed-form
//! normal ES identity against numerical quadrature to 1e-10, ES >= VaR.

use eq_var_engine::expected_shortfall::*;
use eq_var_engine::historical::historical_var;
use eq_var_engine::stats::{normal_pdf, normal_ppf, student_t_ppf};

#[test]
fn empirical_es_hand_exact_tiny_arrays() {
    // Sorted: -10 -8 -6 -4 -2 1 2 3 4 5 (n = 10).
    let pnl = [1.0, -10.0, 2.0, -8.0, 3.0, -6.0, 4.0, -4.0, 5.0, -2.0];
    assert!((expected_shortfall(&pnl, 0.2).unwrap() - 9.0).abs() < 1e-12);
    // alpha = 0.25: an = 2.5 -> fractional weight 0.5 on the 3rd order stat:
    // -( -10 - 8 + 0.5*(-6) ) / 2.5 = 8.4.
    assert!((expected_shortfall(&pnl, 0.25).unwrap() - 8.4).abs() < 1e-12);
    // alpha = 0.1: an = 1 -> the single worst loss.
    assert!((expected_shortfall(&pnl, 0.1).unwrap() - 10.0).abs() < 1e-12);
}

#[test]
fn empirical_es_dominates_var_on_same_sample() {
    let pnl: Vec<f64> = (0..250)
        .map(|t| {
            let td = t as f64;
            1.0e4 * (2.1 * td).sin() + 3.0e3 * (0.37 * td * td).cos()
        })
        .collect();
    for alpha in [0.01, 0.025, 0.05, 0.10] {
        assert!(
            expected_shortfall(&pnl, alpha).unwrap() >= historical_var(&pnl, alpha).unwrap(),
            "alpha={alpha}"
        );
    }
}

#[test]
fn normal_es_identity_vs_numerical_quadrature() {
    let sigma = 1.7e4;
    for alpha in [0.01, 0.025, 0.05] {
        let z = normal_ppf(alpha).unwrap();
        let lo = -14.0;
        let n = 40_000usize;
        let h = (z - lo) / n as f64;
        let mut integral = 0.0;
        for i in 0..=n {
            let x = lo + h * i as f64;
            let f = x * normal_pdf(x);
            let w = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            integral += w * f;
        }
        integral *= h / 3.0;
        let es_quad = -sigma * integral / alpha;
        let es = normal_es(sigma, alpha, 0.0).unwrap();
        assert!((es / es_quad - 1.0).abs() < 1e-10, "alpha={alpha}");
    }
}

#[test]
fn normal_es_exceeds_var_and_scales_with_sigma() {
    let sigma = 1.0e4;
    for alpha in [0.01, 0.025, 0.05] {
        let es = normal_es(sigma, alpha, 0.0).unwrap();
        let var = -normal_ppf(alpha).unwrap() * sigma;
        assert!(es > var, "alpha={alpha}");
    }
    let a = normal_es(2.0 * sigma, 0.01, 0.0).unwrap();
    let b = normal_es(sigma, 0.01, 0.0).unwrap();
    assert!((a - 2.0 * b).abs() < 1e-9);
    let shifted = normal_es(sigma, 0.01, 100.0).unwrap();
    assert!((shifted - (b - 100.0)).abs() < 1e-12);
    assert!((normal_es(0.0, 0.01, 0.0).unwrap()).abs() < 1e-15);
}

#[test]
fn student_t_es_fatter_than_normal_and_normal_limit() {
    let sigma = 1.0e4;
    assert!(student_t_es(sigma, 0.01, 4.0, 0.0).unwrap() > normal_es(sigma, 0.01, 0.0).unwrap());
    assert!(
        student_t_es(sigma, 0.025, 6.0, 0.0).unwrap() > normal_es(sigma, 0.025, 0.0).unwrap()
    );
    let big_df = student_t_es(sigma, 0.025, 1.0e5, 0.0).unwrap();
    let normal = normal_es(sigma, 0.025, 0.0).unwrap();
    assert!((big_df - normal).abs() < 1e-3 * normal);
    let t_var = -student_t_ppf(0.01, 6.0).unwrap() * (4.0f64 / 6.0).sqrt() * sigma;
    assert!(student_t_es(sigma, 0.01, 6.0, 0.0).unwrap() > t_var);
}

#[test]
fn es_validation() {
    let tiny = [1.0, -1.0, 2.0];
    assert!(expected_shortfall(&tiny, 0.05).is_err());
    let mut pnl = vec![1.0; 50];
    pnl[0] = -1.0;
    assert!(expected_shortfall(&pnl, 0.0).is_err());
    assert!(expected_shortfall(&pnl, 0.5).is_err());
    assert!(normal_es(-1.0, 0.01, 0.0).is_err());
    assert!(student_t_es(1.0, 0.01, 2.0, 0.0).is_err());
}
