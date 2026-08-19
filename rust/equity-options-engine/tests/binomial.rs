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
    // Non-finite inputs (including r/q) are rejected, not propagated.
    assert!(
        crr_price(100.0, 100.0, 1.0, f64::NAN, 0.2, 0.0, OptionType::Call, Exercise::European, 100)
            .is_err()
    );
    assert!(
        crr_price(100.0, 100.0, 1.0, 0.05, 0.2, f64::INFINITY, OptionType::Call,
                  Exercise::American, 100)
            .is_err()
    );
    assert!(
        crr_price(f64::INFINITY, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call,
                  Exercise::European, 100)
            .is_err()
    );
    // An absurd step count is rejected up front instead of hanging or
    // aborting in the allocator.
    use eq_options_engine::binomial::MAX_STEPS;
    assert!(
        crr_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call,
                  Exercise::European, MAX_STEPS + 1)
            .is_err()
    );
}

#[test]
fn extreme_moneyness_and_long_expiry_on_the_tree() {
    // 100x moneyness in both directions, plus a 30-year expiry: the tree
    // must stay finite, non-negative, arbitrage-consistent, and close to
    // Black-Scholes for the European leg.
    let (r, q, sigma) = (0.03, 0.01, 0.2);
    for &(s, k, t) in &[
        (10_000.0, 100.0, 1.0),  // deep ITM call / deep OTM put
        (100.0, 10_000.0, 1.0),  // deep OTM call / deep ITM put
        (100.0, 100.0, 30.0),    // 30-year ATM
        (100.0, 100.0, 100.0),   // 100-year ATM (perpetual-ish)
    ] {
        for ot in [OptionType::Call, OptionType::Put] {
            let euro = crr_price(s, k, t, r, sigma, q, ot, Exercise::European, 1000).unwrap();
            let amer = crr_price(s, k, t, r, sigma, q, ot, Exercise::American, 1000).unwrap();
            let bs = bs_price(s, k, t, r, sigma, q, ot).unwrap();
            assert!(euro.is_finite() && amer.is_finite(), "S={s}, K={k}, T={t}, {ot:?}");
            assert!(euro >= 0.0 && amer >= euro - 1e-9, "S={s}, K={k}, T={t}, {ot:?}");
            // Scale-relative gate: at S = 1e4 an absolute 1e-3 tolerance
            // would be a 1e-7 relative demand the O(1/n) tree cannot meet.
            let scale = bs.abs().max(1.0);
            assert!(
                (euro - bs).abs() <= 5e-3 * scale,
                "S={s}, K={k}, T={t}, {ot:?}: tree {euro} vs BS {bs}"
            );
            // Static upper bounds: call <= S e^{-qT}, put <= K e^{-rT}.
            let bound = match ot {
                OptionType::Call => s * (-q * t).exp(),
                OptionType::Put => k * (-r * t).exp(),
            };
            assert!(euro <= bound + 1e-9 * scale, "S={s}, K={k}, T={t}, {ot:?}: {euro} > {bound}");
        }
    }
}

#[test]
fn negative_rates_and_single_step_trees() {
    // Negative r and negative q (post-2015 EUR/CHF reality) are legal and
    // must still converge to Black-Scholes.
    for &(r, q) in &[(-0.005, 0.0), (0.0, -0.01), (-0.01, -0.02), (0.0, 0.0)] {
        for ot in [OptionType::Call, OptionType::Put] {
            let tree = crr_price(100.0, 100.0, 1.0, r, 0.2, q, ot, Exercise::European, 2000)
                .unwrap();
            let bs = bs_price(100.0, 100.0, 1.0, r, 0.2, q, ot).unwrap();
            assert_close!(tree, bs, 2e-3);
            // A negative-carry American call is never exercised early.
            let amer = crr_price(100.0, 100.0, 1.0, r, 0.2, q, ot, Exercise::American, 2000)
                .unwrap();
            assert!(amer >= tree - 1e-9);
        }
    }
    // A one-step tree is a legal (if crude) request: it must price, not
    // panic on an empty backward-induction loop.
    let one = crr_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, Exercise::European, 1)
        .unwrap();
    assert!(one.is_finite() && one > 0.0);
    // Two-node payoff check: exp(-rT)[p (Su - K)^+ + (1-p)(Sd - K)^+].
    let (u, d) = (0.2_f64.exp(), (-0.2_f64).exp());
    let p = ((0.05_f64).exp() - d) / (u - d);
    let expected = (-0.05_f64).exp() * p * (100.0 * u - 100.0);
    assert_close!(one, expected, 1e-12);
    // Early-exercise premium is non-negative even at one step.
    let prem =
        early_exercise_premium(90.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Put, 1).unwrap();
    assert!(prem >= 0.0);
}
