//! Real-life edge cases: deep ITM/OTM, negative rates, tiny tenors,
//! wide vols — each documented in docs/VALIDATION.md and pinned here.

mod common;
use common::assert_close;

use fx_options_engine::rng::Xoshiro256StarStar;
use fx_options_engine::{
    delta, gk_call, gk_price, gk_put, implied_vol, DeltaConvention, OptionType,
};

#[test]
fn deep_itm_call_approaches_discounted_forward_minus_strike() {
    let (s, k, t, rd, rf, sig): (f64, f64, f64, f64, f64, f64) =
        (2.0, 0.5, 0.5, 0.0425, 0.0290, 0.10);
    let f = s * ((rd - rf) * t).exp();
    let want = (-rd * t).exp() * (f - k);
    assert_close(
        gk_call(s, k, t, rd, rf, sig).unwrap(),
        want,
        1e-10,
        "deep ITM call ~ discounted forward intrinsic",
    );
    // Delta pinned to the discount factor.
    let d = delta(s, k, t, rd, rf, sig, OptionType::Call, DeltaConvention::Spot).unwrap();
    assert_close(d, (-rf * t).exp(), 1e-10, "deep ITM call delta");
}

#[test]
fn deep_otm_option_is_essentially_worthless_but_nonnegative() {
    let px = gk_call(1.0, 5.0, 0.25, 0.03, 0.01, 0.10).unwrap();
    assert!(px >= 0.0 && px < 1e-12, "deep OTM call {px}");
    let pp = gk_put(5.0, 1.0, 0.25, 0.03, 0.01, 0.10).unwrap();
    assert!(pp >= 0.0 && pp < 1e-12, "deep OTM put {pp}");
}

#[test]
fn negative_rates_all_identities_hold() {
    // EUR/CHF-era regime: both rates negative.
    let (s, k, t, rd, rf, sig) = (1.06, 1.08, 1.0, -0.0075, -0.0050, 0.065);
    let c = gk_call(s, k, t, rd, rf, sig).unwrap();
    let p = gk_put(s, k, t, rd, rf, sig).unwrap();
    assert!(c > 0.0 && p > 0.0);
    let parity = s * (-rf * t).exp() - k * (-rd * t).exp();
    assert_close(c - p, parity, 1e-14, "negative-rate parity");
    let iv = implied_vol(c, s, k, t, rd, rf, OptionType::Call).unwrap();
    assert_close(iv, sig, 1e-10, "negative-rate IV");
}

#[test]
fn tiny_tenor_and_high_vol_remain_finite() {
    let px = gk_call(1.10, 1.10, 1.0 / 365.0 / 24.0, 0.04, 0.02, 0.30).unwrap();
    assert!(px.is_finite() && px > 0.0);
    let px_wide = gk_call(1.10, 1.10, 2.0, 0.04, 0.02, 3.0).unwrap();
    assert!(px_wide.is_finite() && px_wide < 1.10);
}

#[test]
fn extreme_moneyness_probabilities_do_not_degenerate() {
    // 10-sigma OTM: N(d2) underflows gracefully, price stays >= 0.
    let px = gk_price(1.0, 3.5, 0.1, 0.02, 0.01, 0.10, OptionType::Call).unwrap();
    assert!(px >= 0.0 && px.is_finite());
}

#[test]
fn rng_stream_is_pinned_across_releases() {
    // Regression pin: xoshiro256** seeded via SplitMix64(42).  If these
    // values ever change, previously priced MC books are no longer
    // bit-reproducible — treat as a breaking change.
    let mut rng = Xoshiro256StarStar::new(42);
    assert_eq!(rng.next_u64(), 1546998764402558742);
    assert_eq!(rng.next_u64(), 6990951692964543102);
    assert_eq!(rng.next_u64(), 12544586762248559009);
    let mut rng = Xoshiro256StarStar::new(42);
    rng.next_u64();
    rng.next_u64();
    rng.next_u64();
    let u = rng.next_open01();
    assert!(u > 0.0 && u < 1.0);
}
