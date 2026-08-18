//! Historical-simulation VaR: hand-exact quantiles on tiny arrays, BRW
//! weights, EWMA filtering, sqrt-time scaling, input validation.

use eq_var_engine::historical::*;

fn demo_pnl() -> Vec<f64> {
    (0..100)
        .map(|t| {
            let td = t as f64;
            100.0 * (0.7 * td + 1.0).sin() - if t % 17 == 0 { 150.0 } else { 0.0 }
        })
        .collect()
}

#[test]
fn quantile_linear_hand_exact_tiny_arrays() {
    let x = [1.0, 2.0, 3.0, 4.0];
    assert!((linear_quantile(&x, 0.0).unwrap() - 1.0).abs() < 1e-15);
    assert!((linear_quantile(&x, 1.0).unwrap() - 4.0).abs() < 1e-15);
    assert!((linear_quantile(&x, 0.5).unwrap() - 2.5).abs() < 1e-15);
    assert!((linear_quantile(&x, 0.25).unwrap() - 1.75).abs() < 1e-15); // h = 0.75
    let unsorted = [3.0, 1.0, 2.0];
    assert!((linear_quantile(&unsorted, 0.5).unwrap() - 2.0).abs() < 1e-15);
    assert!((linear_quantile(&unsorted, 0.75).unwrap() - 2.5).abs() < 1e-15); // h = 1.5
    let single = [7.0];
    assert!((linear_quantile(&single, 0.3).unwrap() - 7.0).abs() < 1e-15);
}

#[test]
fn quantile_linear_validation() {
    assert!(linear_quantile(&[], 0.5).is_err());
    let x = [1.0, 2.0];
    assert!(linear_quantile(&x, -0.1).is_err());
    assert!(linear_quantile(&x, 1.1).is_err());
}

#[test]
fn historical_var_matches_quantile_and_monotone_in_alpha() {
    let pnl = demo_pnl();
    assert!(
        (historical_var(&pnl, 0.05).unwrap() - (-linear_quantile(&pnl, 0.05).unwrap())).abs()
            < 1e-12
    );
    assert!(historical_var(&pnl, 0.01).unwrap() >= historical_var(&pnl, 0.05).unwrap());
    assert!(historical_var(&pnl, 0.05).unwrap() >= historical_var(&pnl, 0.10).unwrap());
}

#[test]
fn historical_var_validation() {
    let tiny = vec![0.0; MIN_OBS - 1];
    assert!(historical_var(&tiny, 0.01).is_err());
    let pnl = demo_pnl();
    assert!(historical_var(&pnl, 0.0).is_err());
    assert!(historical_var(&pnl, 0.5).is_err());
    let mut bad = pnl.clone();
    bad[10] = f64::NAN;
    assert!(historical_var(&bad, 0.01).is_err());
}

#[test]
fn brw_weights_sum_to_one_and_monotone_in_recency() {
    for n in [1usize, 5, 250] {
        let w = brw_weights(n, 0.98).unwrap();
        assert_eq!(w.len(), n);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "n={n}");
        for i in 1..n {
            assert!(w[i] > w[i - 1], "weights must increase with recency");
        }
    }
    let w = brw_weights(10, 0.95).unwrap();
    assert!((w[9] / w[8] - 1.0 / 0.95).abs() < 1e-12);
    assert!(brw_weights(10, 1.0).is_err());
    assert!(brw_weights(0, 0.98).is_err());
}

#[test]
fn age_weighted_var_lambda_near_one_recovers_step_historical() {
    let pnl = demo_pnl();
    let brw = age_weighted_var(&pnl, 0.053, 0.999999).unwrap();
    let mut sorted = pnl.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((brw - (-sorted[5])).abs() < 1e-9);
}

#[test]
fn age_weighted_var_recent_crash_dominates_under_low_lambda() {
    let mut recent = vec![10.0; 100];
    let mut old = vec![10.0; 100];
    *recent.last_mut().unwrap() = -500.0;
    old[0] = -500.0;
    let var_recent = age_weighted_var(&recent, 0.05, 0.94).unwrap();
    let var_old = age_weighted_var(&old, 0.05, 0.94).unwrap();
    assert!(var_recent > var_old);
    assert!((var_recent - 500.0).abs() < 1e-9);
}

#[test]
fn ewma_volatility_no_lookahead_and_riskmetrics_recursion() {
    let x = [1.0, -2.0, 3.0, -1.0, 2.0];
    let lam = 0.94;
    let sig = ewma_volatility(&x, lam).unwrap();
    assert_eq!(sig.len(), x.len());
    let mu: f64 = x.iter().sum::<f64>() / 5.0;
    let mut s2: f64 = x.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / 5.0;
    assert!((sig[0] - s2.sqrt()).abs() < 1e-14);
    for t in 1..x.len() {
        s2 = lam * s2 + (1.0 - lam) * x[t - 1] * x[t - 1];
        assert!((sig[t] - s2.sqrt()).abs() < 1e-14, "t={t}");
    }
    assert!(ewma_volatility(&[1.0], lam).is_err());
    assert!(ewma_volatility(&x, 0.0).is_err());
}

#[test]
fn filtered_historical_var_scale_equivariant_and_regime_responsive() {
    let pnl = demo_pnl();
    let scaled: Vec<f64> = pnl.iter().map(|v| v * 3.0).collect();
    assert!(
        (filtered_historical_var(&scaled, 0.05, 0.94).unwrap()
            - 3.0 * filtered_historical_var(&pnl, 0.05, 0.94).unwrap())
        .abs()
            < 1e-9
    );
    let mut calm = pnl.clone();
    for v in calm.iter_mut().skip(60) {
        *v *= 0.05;
    }
    assert!(
        filtered_historical_var(&calm, 0.05, 0.94).unwrap()
            < historical_var(&calm, 0.05).unwrap()
    );
}

#[test]
fn sqrt_time_scaling_and_validation() {
    assert!((scale_var_sqrt_time(100.0, 1).unwrap() - 100.0).abs() < 1e-15);
    assert!((scale_var_sqrt_time(100.0, 4).unwrap() - 200.0).abs() < 1e-12);
    assert!((scale_var_sqrt_time(50.0, 10).unwrap() - 50.0 * 10f64.sqrt()).abs() < 1e-12);
    assert!(scale_var_sqrt_time(100.0, 0).is_err());
}

#[test]
fn overlapping_horizon_pnl_basic() {
    let pnl = demo_pnl();
    let windows = overlapping_horizon_pnl(&pnl, 5).unwrap();
    assert_eq!(windows.len(), pnl.len() - 5 + 1);
    let hand: f64 = pnl[0..5].iter().sum();
    assert!((windows[0] - hand).abs() < 1e-9);
    assert!(overlapping_horizon_pnl(&pnl, 0).is_err());
}
