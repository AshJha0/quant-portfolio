//! Special functions: inverse normal CDF vs known quantiles, chi-squared
//! p-values via the regularized incomplete gamma, Student-t quantiles via
//! the incomplete beta, sample moments. Reference values from scipy 1.17.1.

use eq_var_engine::stats::*;

#[test]
fn normal_ppf_known_quantiles() {
    assert!((normal_ppf(0.975).unwrap() - 1.959963984540054).abs() < 1e-9);
    assert!((normal_ppf(0.01).unwrap() - (-2.3263478740408408)).abs() < 1e-9);
    assert!((normal_ppf(0.995).unwrap() - 2.5758293035489004).abs() < 1e-9);
    assert!((normal_ppf(0.05).unwrap() - (-1.6448536269514722)).abs() < 1e-9);
    assert_eq!(normal_ppf(0.5).unwrap(), 0.0);
}

#[test]
fn normal_ppf_symmetry_and_round_trip() {
    for p in [1e-8, 1e-4, 0.01, 0.025, 0.3, 0.5, 0.7, 0.975, 0.9999] {
        let a = normal_ppf(p).unwrap();
        let b = normal_ppf(1.0 - p).unwrap();
        let tol = 1e-9 * a.abs().max(1.0);
        assert!((a - (-b)).abs() < tol, "p={p} a={a} b={b}");
        assert!((normal_cdf(a) - p).abs() < 1e-9_f64.max(1e-9 * p), "p={p}");
    }
}

#[test]
fn normal_ppf_rejects_out_of_range() {
    assert!(normal_ppf(0.0).is_err());
    assert!(normal_ppf(1.0).is_err());
    assert!(normal_ppf(-0.1).is_err());
}

#[test]
fn normal_cdf_pdf_known_values() {
    assert!((normal_cdf(0.0) - 0.5).abs() < 1e-15);
    assert!((normal_cdf(1.959963984540054) - 0.975).abs() < 1e-13);
    assert!((normal_pdf(0.0) - 0.3989422804014327).abs() < 1e-15);
    let h = 1e-6;
    let d = (normal_cdf(1.0 + h) - normal_cdf(1.0 - h)) / (2.0 * h);
    assert!((d - normal_pdf(1.0)).abs() < 1e-9);
}

#[test]
fn incomplete_gamma_chi2_pvalues() {
    assert!((chi2_sf(3.841458820694124, 1.0).unwrap() - 0.05).abs() < 1e-11);
    assert!((chi2_sf(5.991464547107979, 2.0).unwrap() - 0.05).abs() < 1e-11);
    assert!((chi2_sf(2.0, 1.0).unwrap() - 0.15729920705028105).abs() < 1e-11);
    assert_eq!(chi2_sf(0.0, 1.0).unwrap(), 1.0);
}

#[test]
fn incomplete_gamma_p_plus_q_is_one() {
    for a in [0.5, 1.0, 2.5, 10.0] {
        for x in [0.1, 1.0, 5.0, 30.0] {
            let p = regularized_gamma_p(a, x).unwrap();
            let q = regularized_gamma_q(a, x).unwrap();
            assert!((p + q - 1.0).abs() < 1e-12, "a={a} x={x}");
        }
    }
}

#[test]
fn incomplete_beta_binomial_cdf() {
    assert!((binomial_cdf(4, 250, 0.01).unwrap() - 0.892_187_626_903_625_1).abs() < 1e-11);
    assert!((binomial_cdf(9, 250, 0.01).unwrap() - 0.999_749_809_931_259_5).abs() < 1e-11);
    assert_eq!(binomial_cdf(250, 250, 0.01).unwrap(), 1.0);
    assert_eq!(binomial_cdf(300, 250, 0.01).unwrap(), 1.0); // k >= n saturates
}

#[test]
fn student_t_quantiles_vs_scipy() {
    assert!((student_t_ppf(0.01, 6.0).unwrap() - (-3.142_668_403_291_007)).abs() < 1e-8);
    assert!((student_t_ppf(0.05, 8.0).unwrap() - (-1.859_548_037_530_898)).abs() < 1e-8);
    assert!((student_t_ppf(0.025, 4.5).unwrap() - (-2.658_912_347_204_404)).abs() < 1e-8);
    assert_eq!(student_t_ppf(0.5, 6.0).unwrap(), 0.0);
}

#[test]
fn student_t_cdf_ppf_round_trip_and_limits() {
    assert!((student_t_cdf(-2.0, 5.0).unwrap() - 0.050969739414929174).abs() < 1e-11);
    for p in [0.001, 0.01, 0.1, 0.6, 0.99] {
        let x = student_t_ppf(p, 7.0).unwrap();
        assert!((student_t_cdf(x, 7.0).unwrap() - p).abs() < 1e-10, "p={p}");
    }
    assert!((student_t_ppf(0.01, 1e6).unwrap() - normal_ppf(0.01).unwrap()).abs() < 1e-4);
    assert!(student_t_ppf(0.01, 4.0).unwrap() < normal_ppf(0.01).unwrap());
}

#[test]
fn moments_known_sample() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(mean(&x).unwrap(), 3.0);
    assert!((stdev(&x).unwrap() - 2.5f64.sqrt()).abs() < 1e-15);
    assert!(skewness(&x).unwrap().abs() < 1e-15);
    assert!((excess_kurtosis(&x).unwrap() - (-1.3)).abs() < 1e-11);
    let y = [1.0, 1.0, 1.0, 10.0];
    assert!((skewness(&y).unwrap() - 1.1547005383792515).abs() < 1e-11);
}

#[test]
fn erfc_matches_libm_reference() {
    // A handful of reference values (from libm erfc) over [-2.5, 7].
    let cases: [(f64, f64); 6] = [
        (0.0, 1.0),
        (1.0, 0.15729920705028513),
        (-1.0, 1.842_700_792_949_715),
        (2.0, 0.004677734981047265),
        (-2.5, 1.999593047982555),
        (5.0, 1.5374597944280351e-12),
    ];
    for (x, expected) in cases {
        let got = erfc(x);
        let rel = (got - expected).abs() / expected.abs().max(1e-300);
        assert!(rel < 1e-9, "x={x} got={got} expected={expected}");
    }
}
