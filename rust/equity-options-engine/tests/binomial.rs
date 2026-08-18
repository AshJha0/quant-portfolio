//! CRR tree: convergence to Black-Scholes, American premia, edge cases.

mod common;

use eq_options_engine::{
    bs_price, crr_price, early_exercise_premium, Exercise, OptionType,
};

#[test]
fn european_tree_converges_to_black_scholes() {
    let (s, k, t, r, q, sigma) = (100.0, 100.0, 1.0, 0.05, 0.02, 0.2);
    for ot in [OptionType::Call, OptionType::Put] {
        let bs = bs_price(s, k, t, r, sigma, q, ot).unwrap();
        let tree = crr_price(s, k, t, r, sigma, q, ot, Exercise::European, 2000).unwrap();
        assert_close!(tree, bs, 2e-3);
    }
}

#[test]
fn european_tree_error_decays_with_steps() {
    let (s, k, t, r, q, sigma) = (100.0, 110.0, 0.75, 0.03, 0.01, 0.25);
    let bs = bs_price(s, k, t, r, sigma, q, OptionType::Call).unwrap();
    let err = |n: usize| {
        (crr_price(s, k, t, r, sigma, q, OptionType::Call, Exercise::European, n).unwrap() - bs)
            .abs()
    };
    // Average adjacent step counts to smooth the odd/even oscillation.
    let coarse = 0.5 * (err(50) + err(51));
    let fine = 0.5 * (err(800) + err(801));
    assert!(
        fine < coarse / 4.0,
        "tree error did not decay: coarse {coarse:.3e}, fine {fine:.3e}"
    );
    assert!(fine < 5e-3);
}

#[test]
fn american_geq_european_across_grid() {
    for &(s, k, t, r, q, sigma) in &[
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.2),
        (90.0, 100.0, 0.5, 0.08, 0.0, 0.3),
        (100.0, 90.0, 2.0, 0.02, 0.05, 0.15),
        (50.0, 60.0, 1.5, 0.04, 0.01, 0.4),
    ] {
        for ot in [OptionType::Call, OptionType::Put] {
            let amer = crr_price(s, k, t, r, sigma, q, ot, Exercise::American, 400).unwrap();
            let euro = crr_price(s, k, t, r, sigma, q, ot, Exercise::European, 400).unwrap();
            assert!(
                amer >= euro - 1e-12,
                "American {amer} < European {euro} at (S={s}, K={k}, {ot:?})"
            );
        }
    }
}

#[test]
fn q_zero_american_call_equals_european() {
    // Without dividends, early exercise of a call is never optimal.
    let (s, k, t, r, sigma) = (100.0, 95.0, 1.0, 0.06, 0.25);
    let amer = crr_price(s, k, t, r, sigma, 0.0, OptionType::Call, Exercise::American, 600).unwrap();
    let euro = crr_price(s, k, t, r, sigma, 0.0, OptionType::Call, Exercise::European, 600).unwrap();
    assert_close!(amer, euro, 1e-12);
    let prem =
        early_exercise_premium(s, k, t, r, sigma, 0.0, OptionType::Call, 600).unwrap();
    assert_eq!(prem, 0.0);
}

#[test]
fn itm_american_put_carries_positive_premium() {
    let prem = early_exercise_premium(80.0, 100.0, 1.0, 0.08, 0.2, 0.0, OptionType::Put, 500)
        .unwrap();
    assert!(prem > 0.01, "expected a material premium, got {prem}");
    // Deep ITM American put is worth at least intrinsic:
    let amer = crr_price(80.0, 100.0, 1.0, 0.08, 0.2, 0.0, OptionType::Put, Exercise::American, 500)
        .unwrap();
    assert!(amer >= 20.0 - 1e-12);
}

#[test]
fn t_zero_and_sigma_zero_edge_cases() {
    // T = 0 -> intrinsic.
    let v = crr_price(105.0, 100.0, 0.0, 0.05, 0.2, 0.0, OptionType::Call, Exercise::American, 100)
        .unwrap();
    assert_eq!(v, 5.0);
    // sigma = 0, European: equals BS deterministic limit.
    let (s, k, t, r, q) = (100.0, 95.0, 1.0, 0.05, 0.02);
    let tree = crr_price(s, k, t, r, 0.0, q, OptionType::Call, Exercise::European, 100).unwrap();
    let bs = bs_price(s, k, t, r, 0.0, q, OptionType::Call).unwrap();
    assert_close!(tree, bs, 1e-12);
    // sigma = 0, American put with high r: exercising early beats waiting.
    let amer = crr_price(100.0, 120.0, 1.0, 0.10, 0.0, 0.0, OptionType::Put, Exercise::American, 200)
        .unwrap();
    assert_close!(amer, 20.0, 1e-12); // exercise now: K - S = 20
}

#[test]
fn degenerate_spot_and_strike() {
    // S = 0: American put pays K immediately; European discounts it.
    let ap = crr_price(0.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Put, Exercise::American, 100)
        .unwrap();
    assert_eq!(ap, 100.0);
    let ep = crr_price(0.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Put, Exercise::European, 100)
        .unwrap();
    assert_close!(ep, 100.0 * (-0.05_f64).exp(), 1e-12);
    // K = 0: American call on a dividend payer is worth spot.
    let ac = crr_price(100.0, 0.0, 1.0, 0.05, 0.2, 0.03, OptionType::Call, Exercise::American, 100)
        .unwrap();
    assert_eq!(ac, 100.0);
}

#[test]
fn invalid_inputs_return_err() {
    assert!(
        crr_price(-1.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, Exercise::European, 100)
            .is_err()
    );
    assert!(
        crr_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, Exercise::European, 0)
            .is_err()
    );
    // dt too large for the drift: p outside (0, 1).
    assert!(
        crr_price(100.0, 100.0, 10.0, 0.9, 0.01, 0.0, OptionType::Call, Exercise::European, 1)
            .is_err()
    );
}
