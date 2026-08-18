//! Garman–Kohlhagen analytic identities, limits and failure behaviour.

mod common;
use common::assert_close;

use fx_options_engine::{d1, d2, gk_call, gk_price, gk_put, FxError, OptionType};

const MKT: (f64, f64, f64, f64, f64, f64) = (1.10, 1.05, 0.75, 0.0425, 0.0290, 0.0925);

#[test]
fn two_rate_put_call_parity_to_1e12() {
    // C - P = S e^{-r_f T} - K e^{-r_d T}, on a grid incl. negative rates.
    for &s in &[0.65, 1.10, 147.5] {
        for &k_mult in &[0.85, 1.0, 1.20] {
            for &t in &[0.1, 0.5, 2.0] {
                for &(rd, rf) in &[(0.0425, 0.0290), (0.0050, 0.0525), (-0.0075, -0.0050)] {
                    let k = s * k_mult;
                    let c = gk_call(s, k, t, rd, rf, 0.11).unwrap();
                    let p = gk_put(s, k, t, rd, rf, 0.11).unwrap();
                    let parity = s * (-rf * t).exp() - k * (-rd * t).exp();
                    assert_close(
                        c - p,
                        parity,
                        1e-12 * s.max(1.0),
                        &format!("parity S={s} K={k} T={t} rd={rd} rf={rf}"),
                    );
                }
            }
        }
    }
}

#[test]
fn d1_d2_differ_by_vol_sqrt_t() {
    let (s, k, t, rd, rf, sig) = MKT;
    let x1 = d1(s, k, t, rd, rf, sig).unwrap();
    let x2 = d2(s, k, t, rd, rf, sig).unwrap();
    assert_close(x1 - x2, sig * t.sqrt(), 1e-15, "d1 - d2");
}

#[test]
fn t_zero_returns_intrinsic() {
    assert_close(
        gk_call(1.10, 1.00, 0.0, 0.04, 0.02, 0.10).unwrap(),
        0.10,
        1e-15,
        "T=0 ITM call intrinsic",
    );
    assert_close(
        gk_put(1.10, 1.00, 0.0, 0.04, 0.02, 0.10).unwrap(),
        0.0,
        1e-15,
        "T=0 OTM put intrinsic",
    );
}

#[test]
fn sigma_zero_returns_discounted_forward_intrinsic() {
    let (s, k, t, rd, rf): (f64, f64, f64, f64, f64) = (1.10, 1.05, 0.75, 0.0425, 0.0290);
    let f = s * ((rd - rf) * t).exp();
    let want = (-rd * t).exp() * (f - k);
    assert_close(
        gk_call(s, k, t, rd, rf, 0.0).unwrap(),
        want,
        1e-15,
        "sigma=0 call",
    );
    assert_close(gk_put(s, k, t, rd, rf, 0.0).unwrap(), 0.0, 1e-15, "sigma=0 put");
}

#[test]
fn price_monotone_increasing_in_vol() {
    let (s, k, t, rd, rf, _) = MKT;
    let mut last = gk_call(s, k, t, rd, rf, 0.01).unwrap();
    for &sig in &[0.05, 0.10, 0.20, 0.40, 0.80] {
        let px = gk_call(s, k, t, rd, rf, sig).unwrap();
        assert!(px > last, "call not increasing in vol at sigma={sig}");
        last = px;
    }
}

#[test]
fn call_price_within_no_arbitrage_bounds() {
    let (s, k, t, rd, rf, sig) = MKT;
    let c = gk_call(s, k, t, rd, rf, sig).unwrap();
    let lower = (s * (-rf * t).exp() - k * (-rd * t).exp()).max(0.0);
    let upper = s * (-rf * t).exp();
    assert!(c >= lower && c <= upper, "call {c} outside [{lower}, {upper}]");
}

#[test]
fn gk_price_matches_call_put_wrappers() {
    let (s, k, t, rd, rf, sig) = MKT;
    assert_eq!(
        gk_price(s, k, t, rd, rf, sig, OptionType::Call).unwrap(),
        gk_call(s, k, t, rd, rf, sig).unwrap()
    );
    assert_eq!(
        gk_price(s, k, t, rd, rf, sig, OptionType::Put).unwrap(),
        gk_put(s, k, t, rd, rf, sig).unwrap()
    );
}

#[test]
fn invalid_inputs_return_err() {
    let cases: [(f64, f64, f64, f64, f64, f64); 7] = [
        (-1.0, 1.0, 1.0, 0.0, 0.0, 0.1),      // S <= 0
        (0.0, 1.0, 1.0, 0.0, 0.0, 0.1),       // S == 0
        (1.0, -1.0, 1.0, 0.0, 0.0, 0.1),      // K <= 0
        (1.0, 1.0, -0.5, 0.0, 0.0, 0.1),      // T < 0
        (1.0, 1.0, 1.0, 0.0, 0.0, -0.1),      // sigma < 0
        (f64::NAN, 1.0, 1.0, 0.0, 0.0, 0.1),  // NaN spot
        (1.0, 1.0, 1.0, f64::INFINITY, 0.0, 0.1), // infinite rate
    ];
    for (s, k, t, rd, rf, sig) in cases {
        let res = gk_call(s, k, t, rd, rf, sig);
        assert!(
            matches!(res, Err(FxError::InvalidInput(_))),
            "expected InvalidInput for S={s} K={k} T={t} rd={rd} rf={rf} sigma={sig}"
        );
    }
}

#[test]
fn d1_errors_when_vol_time_degenerate() {
    assert!(d1(1.1, 1.0, 0.0, 0.0, 0.0, 0.1).is_err(), "T=0");
    assert!(d1(1.1, 1.0, 1.0, 0.0, 0.0, 0.0).is_err(), "sigma=0");
}
