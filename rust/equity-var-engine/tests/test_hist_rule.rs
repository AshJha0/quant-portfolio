//! Historical VaR rule: k-th worst and type-7 interpolated against the
//! mpmath oracle (`python/equity/03-var-es-engine/tests/golden/var_oracle.py`).

mod oracle;

use eq_var_engine::historical::{HistRule, historical_var_with_rule, interpolated_var};

fn conv_sample() -> Vec<f64> {
    (0..250)
        .map(|i| {
            let i = f64::from(i);
            let x = 1000.0 * (0.37 * i + 0.1).sin() * (1.0 + 0.5 * (0.11 * i).cos());
            (x * 100.0 + 0.5).floor() / 100.0
        })
        .collect()
}

#[test]
fn both_rules_match_oracle() {
    let p = conv_sample();
    let rows = [
        (
            0.004,
            oracle::HIST_KTH_VAR_A0004,
            oracle::HIST_INTERP_VAR_A0004,
        ),
        (
            0.01,
            oracle::HIST_KTH_VAR_A001,
            oracle::HIST_INTERP_VAR_A001,
        ),
        (
            0.025,
            oracle::HIST_KTH_VAR_A0025,
            oracle::HIST_INTERP_VAR_A0025,
        ),
        (
            0.05,
            oracle::HIST_KTH_VAR_A005,
            oracle::HIST_INTERP_VAR_A005,
        ),
        (0.1, oracle::HIST_KTH_VAR_A01, oracle::HIST_INTERP_VAR_A01),
    ];
    for (a, kth, interp) in rows {
        assert_eq!(
            historical_var_with_rule(&p, a, HistRule::KthWorst).unwrap(),
            kth,
            "{a}"
        );
        let v = historical_var_with_rule(&p, a, HistRule::Interpolated).unwrap();
        assert!(
            (v - interp).abs() <= 1e-12 * interp.abs(),
            "{a}: {v} vs {interp}"
        );
    }
}

#[test]
fn mar99_interpolates_third_and_fourth_worst() {
    let mut l: Vec<f64> = conv_sample().iter().map(|x| -x).collect();
    l.sort_by(|a, b| b.total_cmp(a));
    let v = interpolated_var(&conv_sample(), 0.01).unwrap();
    assert!((v - (l[2] + 0.49 * (l[3] - l[2]))).abs() <= 1e-12 * l[2]);
}

#[test]
fn rejects_empty_and_non_finite() {
    assert!(interpolated_var(&[], 0.01).is_err());
    assert!(interpolated_var(&[1.0, f64::NAN], 0.01).is_err());
}
