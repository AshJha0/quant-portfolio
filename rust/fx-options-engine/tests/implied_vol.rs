//! Implied volatility round trips and no-arbitrage failure behaviour.

mod common;
use common::assert_close;

use fx_options_engine::{gk_price, implied_vol, FxError, OptionType};

#[test]
fn round_trip_grid_to_1e10() {
    let (s, t, rd, rf) = (1.10, 0.5, 0.0425, 0.0290);
    for &k in &[0.95, 1.05, 1.10, 1.15, 1.30] {
        for &sig in &[0.05, 0.0925, 0.25] {
            for ty in [OptionType::Call, OptionType::Put] {
                let px = gk_price(s, k, t, rd, rf, sig, ty).unwrap();
                let iv = implied_vol(px, s, k, t, rd, rf, ty).unwrap();
                assert_close(iv, sig, 1e-10, &format!("IV round trip K={k} sigma={sig} {ty:?}"));
            }
        }
    }
}

#[test]
fn round_trip_with_negative_rates() {
    let (s, k, t, rd, rf, sig) = (1.08, 1.10, 1.0, -0.0075, -0.0050, 0.07);
    for ty in [OptionType::Call, OptionType::Put] {
        let px = gk_price(s, k, t, rd, rf, sig, ty).unwrap();
        let iv = implied_vol(px, s, k, t, rd, rf, ty).unwrap();
        assert_close(iv, sig, 1e-10, "negative-rate IV round trip");
    }
}

#[test]
fn round_trip_short_dated_wings() {
    let (s, t, rd, rf, sig) = (1.10, 0.02, 0.0425, 0.0290, 0.12);
    // K=1.00, T=0.02 (~7.3 days) is a genuine IEEE-754 floor case, not a
    // solver deficiency: the ITM call price S e^{-r_f T} N(d1) -
    // K e^{-r_d T} N(d2) is a difference of two O(1) terms giving an O(0.1)
    // result (~1 digit of cancellation), while the wing's vega is ~7.6e-9.
    // The objective sigma -> price(sigma) - price is therefore a staircase
    // in sigma with ULP-sized flats ~1.4e-8 to 2.4e-8 wide near the root
    // (verified by scanning the objective and by re-seeding Brent with
    // arbitrarily tight brackets around the true vol — every variant lands
    // within this same band, because many adjacent sigma values round to
    // the identical price double).  No root finder operating on this exact
    // price residual can resolve the root more tightly than that flat is
    // wide.  The equivalent unfixed C++ engine (`cpp/fx-options-engine`)
    // reproduces an error of the same order (2.5e-7, worse, since it still
    // has the premature-Newton-exit bug fixed here in
    // `src/implied_vol.rs`) confirming this isn't Rust-specific.  Other
    // strikes/types in this grid round-trip to <1e-12; only this one wing
    // needs the wider, still-tight, tolerance.
    for &k in &[1.00, 1.20] {
        for ty in [OptionType::Call, OptionType::Put] {
            let px = gk_price(s, k, t, rd, rf, sig, ty).unwrap();
            let iv = implied_vol(px, s, k, t, rd, rf, ty).unwrap();
            assert_close(iv, sig, 2e-8, &format!("short-dated wing K={k} {ty:?}"));
        }
    }
}

#[test]
fn price_outside_no_arbitrage_bounds_errs() {
    let (s, k, t, rd, rf) = (1.10, 1.05, 0.5, 0.0425, 0.0290);
    // Below discounted forward intrinsic.
    let res = implied_vol(0.0, s, k, t, rd, rf, OptionType::Call);
    assert!(matches!(res, Err(FxError::InvalidInput(_))), "below intrinsic");
    // Above the sigma -> inf bound (S e^{-r_f T} for a call).
    let res = implied_vol(1.20, s, k, t, rd, rf, OptionType::Call);
    assert!(matches!(res, Err(FxError::InvalidInput(_))), "above upper bound");
}

#[test]
fn zero_time_value_returns_zero_vol() {
    let (s, k, t, rd, rf): (f64, f64, f64, f64, f64) = (1.10, 0.80, 0.25, 0.0425, 0.0290);
    // Deep ITM call priced exactly at discounted forward intrinsic.  Build
    // the forward with the *same* arithmetic `implied_vol` uses internally
    // (F = S * df_f / df_d, i.e. two separate `exp` calls and a division)
    // rather than the mathematically-equivalent `S * exp((r_d - r_f) * T)`
    // — the two expressions are not bit-identical in floating point, and
    // `implied_vol`'s near-intrinsic detection (`price - lower <= 1e-16 *
    // lower.max(1.0)`) is deliberately a last-ULP check, so it only fires
    // when the caller's price matches its internal `lower` to the bit.
    // (Mirrors the C++ engine's `ZeroTimeValueReturnsZeroVol`, which
    // documents and relies on the identical construction.)
    let df_d = (-rd * t).exp();
    let df_f = (-rf * t).exp();
    let forward = s * df_f / df_d;
    let intrinsic = df_d * (forward - k);
    let iv = implied_vol(intrinsic, s, k, t, rd, rf, OptionType::Call).unwrap();
    assert_eq!(iv, 0.0);
}

#[test]
fn t_zero_and_bad_price_err() {
    assert!(implied_vol(0.05, 1.10, 1.05, 0.0, 0.04, 0.02, OptionType::Call).is_err());
    assert!(implied_vol(f64::NAN, 1.10, 1.05, 0.5, 0.04, 0.02, OptionType::Call).is_err());
}
