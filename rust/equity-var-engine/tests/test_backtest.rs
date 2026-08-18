//! Backtesting: Kupiec LR hand-computed, chi2 p-values, Christoffersen
//! clustering detection, Basel traffic-light exact zone boundaries.

use eq_var_engine::backtest::*;
use eq_var_engine::stats::chi2_sf;

#[test]
fn exceptions_indicator_and_broadcast() {
    let pnl = [-120.0, 50.0, -99.9, -100.1, 0.0];
    let var1 = [100.0]; // scalar broadcast
    let ex = exceptions_from_pnl(&pnl, &var1).unwrap();
    assert_eq!(ex, vec![1, 0, 0, 1, 0]);
    let vard = [130.0, 100.0, 90.0, 100.0, 100.0];
    assert_eq!(exceptions_from_pnl(&pnl, &vard).unwrap(), vec![0, 0, 1, 1, 0]);
    assert!(exceptions_from_pnl(&pnl, &[-1.0]).is_err());
    assert!(exceptions_from_pnl(&pnl, &[1.0, 2.0]).is_err());
}

#[test]
fn kupiec_hand_computed_lr() {
    let (t, x, p): (f64, f64, f64) = (250.0, 5.0, 0.01);
    let pihat = x / t;
    let ll0 = (t - x) * (1.0 - p).ln() + x * p.ln();
    let ll1 = (t - x) * (1.0 - pihat).ln() + x * pihat.ln();
    let lr_hand = -2.0 * (ll0 - ll1);
    let r = kupiec_pof(250, 5, 0.01).unwrap();
    assert!((r.lr - lr_hand).abs() < 1e-12);
    assert!((r.pvalue - chi2_sf(lr_hand, 1.0).unwrap()).abs() < 1e-14);
    assert!((r.expected - 2.5).abs() < 1e-12);
    assert!((r.rate - 0.02).abs() < 1e-12);
}

#[test]
fn kupiec_chi2_critical_value_gives_point_oh_five() {
    assert!((chi2_sf(3.841458820694124, 1.0).unwrap() - 0.05).abs() < 1e-4);
    let exact = kupiec_pof(200, 2, 0.01).unwrap();
    assert!(exact.lr.abs() < 1e-12);
    assert!((exact.pvalue - 1.0).abs() < 1e-12);
}

#[test]
fn kupiec_degenerate_counts_and_monotonicity() {
    let zero = kupiec_pof(250, 0, 0.01).unwrap();
    assert!((zero.lr - (-2.0 * 250.0 * 0.99f64.ln())).abs() < 1e-12);
    assert!(kupiec_pof(250, 10, 0.01).unwrap().lr > kupiec_pof(250, 6, 0.01).unwrap().lr);
    assert!(kupiec_pof(250, 10, 0.01).unwrap().pvalue < 0.05);
    assert!(kupiec_pof(0, 0, 0.01).is_err());
    assert!(kupiec_pof(250, 251, 0.01).is_err());
    assert!(kupiec_pof(250, -1, 0.01).is_err());
}

#[test]
fn christoffersen_transition_counts_on_tiny_pattern() {
    let ex = [0u8, 1, 1, 0, 1];
    let r = christoffersen_independence(&ex).unwrap();
    assert!((r.n00 - 0.0).abs() < 1e-12);
    assert!((r.n01 - 2.0).abs() < 1e-12);
    assert!((r.n10 - 1.0).abs() < 1e-12);
    assert!((r.n11 - 1.0).abs() < 1e-12);
    assert!((r.pi01 - 1.0).abs() < 1e-12);
    assert!((r.pi11 - 0.5).abs() < 1e-12);
}

#[test]
fn christoffersen_detects_planted_clustering() {
    let mut clustered = vec![0u8; 250];
    let mut spread = vec![0u8; 250];
    for v in clustered.iter_mut().take(110).skip(100) {
        *v = 1;
    }
    let mut t = 12;
    while t < 250 {
        spread[t] = 1;
        t += 25;
    }
    let c = christoffersen_independence(&clustered).unwrap();
    let s = christoffersen_independence(&spread).unwrap();
    assert!(c.pvalue < 0.001, "a run of 10 exceptions must reject independence");
    assert!(c.pi11 > c.pi01);
    assert!(s.pvalue > 0.10, "evenly spread exceptions must not reject");
    assert!(c.lr > s.lr);
    assert!(christoffersen_independence(&[1u8]).is_err());
}

#[test]
fn christoffersen_conditional_coverage_is_sum_of_components() {
    let mut ex = vec![0u8; 250];
    for v in ex.iter_mut().take(108).skip(100) {
        *v = 1;
    }
    let uc = kupiec_pof(250, 8, 0.01).unwrap();
    let ind = christoffersen_independence(&ex).unwrap();
    let cc = christoffersen_cc(&ex, 0.01).unwrap();
    assert!((cc.lr - (uc.lr + ind.lr)).abs() < 1e-12);
    assert!((cc.lr_uc - uc.lr).abs() < 1e-12);
    assert!((cc.lr_ind - ind.lr).abs() < 1e-12);
    assert!((cc.pvalue - chi2_sf(cc.lr, 2.0).unwrap()).abs() < 1e-14);
    assert!(cc.pvalue < 0.05);
}

#[test]
fn basel_exact_zone_boundaries_at_250_obs() {
    assert_eq!(basel_traffic_light(0, 250).unwrap().zone, BaselZone::Green);
    let r4 = basel_traffic_light(4, 250).unwrap();
    assert_eq!(r4.zone, BaselZone::Green);
    assert!((r4.multiplier - 3.0).abs() < 1e-12);

    let r5 = basel_traffic_light(5, 250).unwrap();
    assert_eq!(r5.zone, BaselZone::Yellow);
    assert!((r5.multiplier - 3.40).abs() < 1e-12);
    assert!((basel_traffic_light(6, 250).unwrap().multiplier - 3.50).abs() < 1e-12);
    assert!((basel_traffic_light(7, 250).unwrap().multiplier - 3.65).abs() < 1e-12);
    assert!((basel_traffic_light(8, 250).unwrap().multiplier - 3.75).abs() < 1e-12);
    let r9 = basel_traffic_light(9, 250).unwrap();
    assert_eq!(r9.zone, BaselZone::Yellow);
    assert!((r9.multiplier - 3.85).abs() < 1e-12);

    let r10 = basel_traffic_light(10, 250).unwrap();
    assert_eq!(r10.zone, BaselZone::Red);
    assert!((r10.multiplier - 4.0).abs() < 1e-12);
    assert_eq!(basel_traffic_light(15, 250).unwrap().zone, BaselZone::Red);
    assert!(basel_traffic_light(-1, 250).is_err());
}

#[test]
fn basel_binomial_zone_probabilities() {
    assert!(
        (basel_traffic_light(4, 250).unwrap().cumulative_prob - 0.892_187_626_903_625_1).abs()
            < 1e-10
    );
    assert!(
        (basel_traffic_light(9, 250).unwrap().cumulative_prob - 0.999_749_809_931_259_5).abs()
            < 1e-10
    );
    assert_eq!(BaselZone::Green.as_str(), "green");
    assert_eq!(BaselZone::Yellow.as_str(), "yellow");
    assert_eq!(BaselZone::Red.as_str(), "red");
}
