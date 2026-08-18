//! Black-76: equivalence with BSM on the forward, Greeks, edge cases.

mod common;

use eq_options_engine::{
    b76_d1_d2, black76_greeks, black76_price, bs_price, forward_price, OptionType,
};

#[test]
fn black76_equals_bsm_on_forward_to_1e12() {
    for &(s, k, t, r, q, sigma) in &[
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.2),
        (100.0, 110.0, 0.25, 0.03, 0.01, 0.3),
        (100.0, 90.0, 2.0, 0.02, 0.03, 0.15),
        (50.0, 100.0, 1.0, 0.05, 0.0, 0.4),
        (200.0, 100.0, 0.5, 0.04, 0.02, 0.25),
        (100.0, 105.0, 1.0, -0.01, 0.0, 0.2),
        (1000.0, 1200.0, 1.5, 0.045, 0.015, 0.28),
    ] {
        let f = forward_price(s, t, r, q);
        for ot in [OptionType::Call, OptionType::Put] {
            let b76 = black76_price(f, k, t, r, sigma, ot).unwrap();
            let bsm = bs_price(s, k, t, r, sigma, q, ot).unwrap();
            assert_close!(b76, bsm, 1e-12);
        }
    }
}

#[test]
fn black76_put_call_parity() {
    // C - P = e^{-rT} (F - K).
    for &(f, k, t, r, sigma) in &[
        (100.0, 100.0, 1.0, 0.05, 0.2),
        (100.0, 120.0, 0.5, 0.03, 0.35),
        (80.0, 70.0, 2.0, 0.0, 0.15),
    ] {
        let c = black76_price(f, k, t, r, sigma, OptionType::Call).unwrap();
        let p = black76_price(f, k, t, r, sigma, OptionType::Put).unwrap();
        assert_close!(c - p, (-r * t).exp() * (f - k), 1e-12);
    }
}

#[test]
fn black76_greeks_match_finite_differences() {
    let (f, k, t, r, sigma) = (105.0, 100.0, 0.75, 0.04, 0.25);
    for ot in [OptionType::Call, OptionType::Put] {
        let g = black76_greeks(f, k, t, r, sigma, ot).unwrap();
        let h = 1e-5 * f;
        let up = black76_price(f + h, k, t, r, sigma, ot).unwrap();
        let dn = black76_price(f - h, k, t, r, sigma, ot).unwrap();
        assert_close!(g.delta, (up - dn) / (2.0 * h), 1e-7);
        let mid = black76_price(f, k, t, r, sigma, ot).unwrap();
        let h2 = 2e-4 * f;
        let up2 = black76_price(f + h2, k, t, r, sigma, ot).unwrap();
        let dn2 = black76_price(f - h2, k, t, r, sigma, ot).unwrap();
        assert_close!(g.gamma, (up2 - 2.0 * mid + dn2) / (h2 * h2), 1e-7);
        let hv = 1e-5;
        let vu = black76_price(f, k, t, r, sigma + hv, ot).unwrap();
        let vd = black76_price(f, k, t, r, sigma - hv, ot).unwrap();
        assert_close!(g.vega, (vu - vd) / (2.0 * hv), 1e-6);
        let ht = 1e-6;
        let tu = black76_price(f, k, t + ht, r, sigma, ot).unwrap();
        let td = black76_price(f, k, t - ht, r, sigma, ot).unwrap();
        assert_close!(g.theta, -(tu - td) / (2.0 * ht), 1e-5);
        // rho = -T * V by construction (discounting only):
        assert_close!(g.rho, -t * g.price, 1e-13);
    }
}

#[test]
fn black76_edge_cases_and_errors() {
    // T = 0 -> intrinsic, undiscounted.
    assert_eq!(
        black76_price(105.0, 100.0, 0.0, 0.05, 0.2, OptionType::Call).unwrap(),
        5.0
    );
    // sigma = 0 -> discounted intrinsic.
    let v = black76_price(105.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call).unwrap();
    assert_close!(v, (-0.05_f64).exp() * 5.0, 1e-12);
    // r = 0: futures-style, no premium discounting.
    let c = black76_price(100.0, 100.0, 1.0, 0.0, 0.2, OptionType::Call).unwrap();
    let p = black76_price(100.0, 100.0, 1.0, 0.0, 0.2, OptionType::Put).unwrap();
    assert_close!(c, p, 1e-12); // ATM-forward call == put when r = 0
    // Invalid inputs:
    assert!(black76_price(-1.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call).is_err());
    assert!(b76_d1_d2(100.0, 100.0, 0.0, 0.2).is_err());
    assert!(black76_greeks(100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call).is_err());
}
