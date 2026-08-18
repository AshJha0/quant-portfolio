//! CIP forwards, forward points and Garman–Kohlhagen == Black-76.

mod common;
use common::assert_close;

use fx_options_engine::forwards::{PIP_FACTOR_DEFAULT, PIP_FACTOR_JPY};
use fx_options_engine::{
    black76_from_spot, black76_price, cip_forward, forward_points, gk_call, gk_price, gk_put,
    synthetic_forward_from_options, OptionType,
};

#[test]
fn gk_equals_black76_to_1e12_across_grid() {
    for &s in &[0.65, 1.10, 147.5] {
        for &k_mult in &[0.85, 1.0, 1.20] {
            for &t in &[0.1, 0.5, 2.0] {
                for &(rd, rf) in &[(0.0425, 0.0290), (0.0050, 0.0525), (-0.0075, -0.0050)] {
                    for ty in [OptionType::Call, OptionType::Put] {
                        let k = s * k_mult;
                        let gk = gk_price(s, k, t, rd, rf, 0.11, ty).unwrap();
                        let b76 = black76_from_spot(s, k, t, rd, rf, 0.11, ty).unwrap();
                        assert_close(
                            gk,
                            b76,
                            1e-12 * s.max(1.0),
                            &format!("GK vs B76 S={s} K={k} T={t} rd={rd} rf={rf} {ty:?}"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cip_forward_carry_sign() {
    // r_d > r_f: base currency trades at a forward premium (F > S).
    assert!(cip_forward(1.10, 1.0, 0.05, 0.02).unwrap() > 1.10);
    // r_d < r_f: forward discount.
    assert!(cip_forward(1.10, 1.0, 0.02, 0.05).unwrap() < 1.10);
    // T = 0: forward equals spot.
    assert_close(
        cip_forward(1.10, 0.0, 0.05, 0.02).unwrap(),
        1.10,
        1e-15,
        "F(T=0)",
    );
}

#[test]
fn forward_points_scale_with_pip_factor() {
    let (s, t, rd, rf) = (150.0, 0.5, 0.0050, 0.0425); // USDJPY-style
    let f = cip_forward(s, t, rd, rf).unwrap();
    let pts = forward_points(s, t, rd, rf, PIP_FACTOR_JPY).unwrap();
    assert_close(pts, (f - s) * 100.0, 1e-10, "JPY forward points");
    assert!(pts < 0.0, "USD discount vs JPY when r_d < r_f");

    let pts_default = forward_points(1.10, t, 0.0425, 0.0290, PIP_FACTOR_DEFAULT).unwrap();
    assert!(pts_default > 0.0);
    assert!(forward_points(1.10, t, 0.0425, 0.0290, 0.0).is_err());
}

#[test]
fn synthetic_forward_recovers_cip_forward() {
    let (s, k, t, rd, rf, sig) = (1.10, 1.07, 0.75, 0.0425, 0.0290, 0.0925);
    let c = gk_call(s, k, t, rd, rf, sig).unwrap();
    let p = gk_put(s, k, t, rd, rf, sig).unwrap();
    let f_syn = synthetic_forward_from_options(c, p, k, t, rd).unwrap();
    let f_cip = cip_forward(s, t, rd, rf).unwrap();
    assert_close(f_syn, f_cip, 1e-12, "synthetic forward vs CIP");
}

#[test]
fn black76_zero_vol_and_t_zero_return_discounted_intrinsic() {
    let df = (-0.04_f64 * 0.5).exp();
    assert_close(
        black76_price(1.12, 1.05, 0.5, 0.04, 0.0, OptionType::Call).unwrap(),
        df * 0.07,
        1e-15,
        "B76 sigma=0",
    );
    assert_close(
        black76_price(1.12, 1.05, 0.0, 0.04, 0.2, OptionType::Call).unwrap(),
        0.07,
        1e-15,
        "B76 T=0",
    );
}

#[test]
fn black76_invalid_inputs_err() {
    assert!(black76_price(-1.0, 1.0, 1.0, 0.0, 0.1, OptionType::Call).is_err());
    assert!(black76_price(1.0, 0.0, 1.0, 0.0, 0.1, OptionType::Call).is_err());
    assert!(black76_price(1.0, 1.0, -1.0, 0.0, 0.1, OptionType::Call).is_err());
    assert!(black76_price(1.0, 1.0, 1.0, 0.0, -0.1, OptionType::Call).is_err());
    assert!(black76_price(f64::NAN, 1.0, 1.0, 0.0, 0.1, OptionType::Call).is_err());
    assert!(cip_forward(-1.0, 1.0, 0.0, 0.0).is_err());
    assert!(synthetic_forward_from_options(f64::NAN, 0.0, 1.0, 1.0, 0.0).is_err());
}
