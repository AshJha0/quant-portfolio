//! Option revaluation: golden values from `eq_var.revaluation` (Python),
//! put-call parity, input rejection.

use eq_var_engine::option_reval::*;

const CALL: OptionLeg = OptionLeg {
    quantity: 10.0,
    kind: OptionKind::Call,
    spot: 100.0,
    strike: 105.0,
    tau: 0.5,
    rate: 0.03,
    carry: 0.01,
    vol: 0.25,
    factor: 0,
};
const PUT: OptionLeg = OptionLeg {
    quantity: -7.0,
    kind: OptionKind::Put,
    spot: 50.0,
    strike: 45.0,
    tau: 1.25,
    rate: 0.02,
    carry: 0.04,
    vol: 0.35,
    factor: 1,
};
const X: [f64; 6] = [0.01, -0.02, -0.05, 0.03, 0.0, 0.0];
const DV: [f64; 6] = [0.01, 0.0, 0.05, -0.02, 0.0, 0.0];

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} vs {b}");
}

#[test]
fn price_golden_and_parity() {
    let c = bsm_price(100.0, 105.0, 0.5, 0.03, 0.01, 0.25, OptionKind::Call).unwrap();
    let p = bsm_price(100.0, 105.0, 0.5, 0.03, 0.01, 0.25, OptionKind::Put).unwrap();
    close(c, 5.349804578881077, 1e-12);
    close(
        bsm_price(50.0, 45.0, 1.25, 0.02, 0.04, 0.35, OptionKind::Put).unwrap(),
        5.403708192910557,
        1e-12,
    );
    close(
        c - p,
        100.0 * (-0.005f64).exp() - 105.0 * (-0.015f64).exp(),
        1e-12,
    );
}

#[test]
fn greeks_golden() {
    let g = bsm_greeks(&CALL).unwrap();
    close(g.delta, 0.4456372919675482, 1e-12);
    close(g.gamma, 0.022263053570122843, 1e-12);
    close(g.vega, 27.828816962653555, 1e-11);
    close(g.vanna, 0.484579085972438, 1e-12);
    close(g.volga, 4.490141631849943, 1e-11);
    close(g.theta, -7.687984687232052, 1e-11);
}

#[test]
fn full_revaluation_golden() {
    let pnl = full_revaluation_pnl(&[CALL, PUT], &X, &DV, 2, 1.0 / 365.0).unwrap();
    close(pnl[0], 4.92183106078723, 1e-11);
    close(pnl[1], -0.6845739357538516, 1e-11);
    close(pnl[2], -0.1540808210950111, 1e-11);
}

#[test]
fn delta_gamma_golden() {
    let pnl = delta_gamma_pnl(&[CALL, PUT], &X, 2).unwrap();
    close(pnl[0], 2.260614231417638, 1e-11);
    close(pnl[1], -15.741558009225477, 1e-11);
    assert_eq!(pnl[2], 0.0);
}

#[test]
fn rejects_bad_input() {
    assert!(full_revaluation_pnl(&[PUT], &[0.0, 0.0], &[], 1, 0.0).is_err());
    assert!(delta_gamma_pnl(&[PUT], &[f64::NAN, 0.0], 2).is_err());
    assert!(bsm_price(100.0, 100.0, 1.0, 0.0, 0.0, 0.0, OptionKind::Call).is_err());
}
