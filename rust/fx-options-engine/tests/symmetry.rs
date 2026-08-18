//! Foreign–domestic symmetry (notional duality).
//!
//! A EURUSD call (right to buy EUR paying USD) is, viewed from the EUR
//! side, a USDEUR put (right to sell USD receiving EUR).  Formally:
//!
//! `C_d(S, K, T, r_d, r_f, sigma) = S * K * P_f(1/S, 1/K, T, r_f, r_d, sigma)`
//!
//! where the flipped option is priced with the rate roles swapped, its
//! premium expressed in *foreign* currency and rescaled by `S*K`.

mod common;
use common::assert_close;

use fx_options_engine::{delta, gk_price, DeltaConvention, OptionType};

#[test]
fn call_equals_flipped_put_across_grid() {
    // 3 spots x 3 strikes x 3 tenors x 3 rate pairs = 81 grid points.
    for &s in &[0.65, 1.10, 147.5] {
        for &k_mult in &[0.85, 1.0, 1.20] {
            for &t in &[0.1, 0.5, 2.0] {
                for &(rd, rf) in &[(0.0425, 0.0290), (0.0050, 0.0525), (-0.0075, -0.0050)] {
                    let k = s * k_mult;
                    let sigma = 0.11;
                    let lhs = gk_price(s, k, t, rd, rf, sigma, OptionType::Call).unwrap();
                    let rhs = s
                        * k
                        * gk_price(1.0 / s, 1.0 / k, t, rf, rd, sigma, OptionType::Put).unwrap();
                    assert_close(
                        lhs,
                        rhs,
                        1e-10 * (s * k).max(1.0),
                        &format!("symmetry S={s} K={k} T={t} rd={rd} rf={rf}"),
                    );
                }
            }
        }
    }
}

#[test]
fn put_equals_flipped_call() {
    let (s, k, t, rd, rf, sigma) = (1.10, 1.05, 0.75, 0.03, 0.01, 0.09);
    let lhs = gk_price(s, k, t, rd, rf, sigma, OptionType::Put).unwrap();
    let rhs = s * k * gk_price(1.0 / s, 1.0 / k, t, rf, rd, sigma, OptionType::Call).unwrap();
    assert_close(lhs, rhs, 1e-12, "flipped call");
}

#[test]
fn premium_adjusted_delta_is_flipped_forward_delta() {
    // The PA forward delta of a call is minus the (unadjusted) forward
    // delta of the flipped put, rescaled by the K/F notional conversion:
    // PA call delta = (K/F) N(d2), flipped put forward delta = -N(d2).
    // This is *why* PA deltas exist: they are the hedge seen from the
    // other currency's viewpoint.
    let (s, k, t, rd, rf, sigma): (f64, f64, f64, f64, f64, f64) =
        (1.10, 1.15, 0.5, 0.0425, 0.0290, 0.0825);
    let f = s * ((rd - rf) * t).exp();
    let pa = delta(
        s,
        k,
        t,
        rd,
        rf,
        sigma,
        OptionType::Call,
        DeltaConvention::ForwardPa,
    )
    .unwrap();
    let flipped = delta(
        1.0 / s,
        1.0 / k,
        t,
        rf,
        rd,
        sigma,
        OptionType::Put,
        DeltaConvention::Forward,
    )
    .unwrap();
    assert_close(pa, -(k / f) * flipped, 1e-12, "PA vs flipped forward delta");
}
