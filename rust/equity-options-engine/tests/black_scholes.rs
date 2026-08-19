//! Black-Scholes analytic identities, erfc accuracy, and edge cases.

mod common;

use eq_options_engine::black_scholes::{erfc, forward_price, norm_cdf};
use eq_options_engine::{bs_price, d1_d2, OptionType, PricingError};

/// Grid shared by the parity / Black-76 / IV tests.
fn grid() -> Vec<(f64, f64, f64, f64, f64, f64)> {
    let mut g = Vec::new();
    for &s in &[50.0, 100.0, 150.0] {
        for &k in &[50.0, 100.0, 150.0] {
            for &t in &[0.05, 0.5, 2.0] {
                for &r in &[-0.01, 0.0, 0.05] {
                    for &q in &[0.0, 0.03] {
                        for &sigma in &[0.05, 0.2, 0.8] {
                            g.push((s, k, t, r, q, sigma));
                        }
                    }
                }
            }
        }
    }
    g
}

#[test]
fn put_call_parity_holds_to_1e12_across_grid() {
    // C - P = S e^{-qT} - K e^{-rT}, checked on a 486-point grid.
    let grid = grid();
    assert_eq!(grid.len(), 486);
    for (s, k, t, r, q, sigma) in grid {
        let c = bs_price(s, k, t, r, sigma, q, OptionType::Call).unwrap();
        let p = bs_price(s, k, t, r, sigma, q, OptionType::Put).unwrap();
        let rhs = s * (-q * t).exp() - k * (-r * t).exp();
        assert_close!(c - p, rhs, 1e-12);
    }
}

#[test]
fn erfc_matches_reference_values_to_1e12() {
    // Reference values from mpmath (50 digits), truncated to double.
    let cases = [
        (0.0, 1.0),
        (0.1, 0.887537083981715107249533425065322000551732966843),
        (0.25, 0.7236736098317630356373568274229297198765836184701),
        (0.5, 0.4795001221869534623509796091771072324349211719554),
        (1.0, 0.1572992070502851306587793649773588850093113545927),
        (1.5, 0.03389485352468927325549597044461940464146059336748),
        (2.0, 0.004677734981047265),
        (3.0, 2.20904969985854413727761295823203798477070873992e-5),
        (4.0, 1.541725790028001885215967348757266692083974794355e-8),
        (5.0, 1.537459794428034850188343485383378890118422216861e-12),
        (-0.5, 1.520499877813046537649020390822892767565078828045),
        (-1.0, 1.842700792949714869341220635022641114990688645407),
        (-2.0, 1.9953222650189528),
        (-3.0, 1.999977909503001414558627223870417679620152292913),
    ];
    for (x, want) in cases {
        let got = erfc(x);
        let scale: f64 = want;
        assert!(
            (got - want).abs() <= 1e-12 * scale.abs().max(1.0),
            "erfc({x}) = {got}, want {want} (diff {:.3e})",
            (got - want).abs()
        );
    }
}

#[test]
fn norm_cdf_matches_known_quantiles() {
    assert_close!(norm_cdf(0.0), 0.5, 1e-15);
    assert_close!(norm_cdf(1.0), 0.841344746068542948585232545632, 1e-13);
    assert_close!(norm_cdf(-1.0), 0.158655253931457051414767454368, 1e-13);
    assert_close!(norm_cdf(1.959963984540054), 0.975, 1e-12);
    assert_close!(norm_cdf(2.326347874040841), 0.99, 1e-12);
    // Deep tail keeps relative accuracy (naive 1 - Phi(-x) would be 0 here):
    let tail = norm_cdf(-8.0);
    assert_rel_close!(tail, 6.22096057427178412351599517e-16, 1e-10);
    // Symmetry:
    for x in [-3.5, -1.2, 0.3, 2.7] {
        assert_close!(norm_cdf(x) + norm_cdf(-x), 1.0, 1e-14);
    }
}

#[test]
fn d1_d2_hull_example() {
    // Hull's classic: S=42, K=40, T=0.5, r=0.10, sigma=0.2.
    let (d1, d2) = d1_d2(42.0, 40.0, 0.5, 0.10, 0.2, 0.0).unwrap();
    assert_close!(d1, 0.7692626281060315, 1e-12);
    assert_close!(d2, 0.627841271868722, 1e-12);
}

#[test]
fn t_zero_returns_intrinsic() {
    let c = bs_price(105.0, 100.0, 0.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
    assert_eq!(c, 5.0);
    let p = bs_price(105.0, 100.0, 0.0, 0.05, 0.2, 0.0, OptionType::Put).unwrap();
    assert_eq!(p, 0.0);
    let p_itm = bs_price(90.0, 100.0, 0.0, 0.05, 0.2, 0.0, OptionType::Put).unwrap();
    assert_eq!(p_itm, 10.0);
}

#[test]
fn sigma_zero_returns_discounted_forward_intrinsic() {
    let (s, k, t, r, q) = (100.0, 95.0, 1.0, 0.05, 0.02);
    let f = forward_price(s, t, r, q);
    let c = bs_price(s, k, t, r, 0.0, q, OptionType::Call).unwrap();
    assert_close!(c, (-r * t).exp() * (f - k).max(0.0), 1e-14);
    let p = bs_price(s, k, t, r, 0.0, q, OptionType::Put).unwrap();
    assert_close!(p, (-r * t).exp() * (k - f).max(0.0), 1e-14);
}

#[test]
fn zero_strike_and_zero_spot_limits() {
    // K = 0: call is the dividend-adjusted forward on the stock; put is 0.
    let c = bs_price(100.0, 0.0, 2.0, 0.05, 0.2, 0.03, OptionType::Call).unwrap();
    assert_close!(c, 100.0 * (-0.03_f64 * 2.0).exp(), 1e-12);
    let p = bs_price(100.0, 0.0, 2.0, 0.05, 0.2, 0.03, OptionType::Put).unwrap();
    assert_eq!(p, 0.0);
    // S = 0: call is 0; put is the discounted strike.
    let c0 = bs_price(0.0, 100.0, 2.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
    assert_eq!(c0, 0.0);
    let p0 = bs_price(0.0, 100.0, 2.0, 0.05, 0.2, 0.0, OptionType::Put).unwrap();
    assert_close!(p0, 100.0 * (-0.05_f64 * 2.0).exp(), 1e-12);
}

#[test]
fn negative_rates_are_supported() {
    let c = bs_price(100.0, 100.0, 1.0, -0.01, 0.2, -0.005, OptionType::Call).unwrap();
    assert!(c > 0.0 && c.is_finite());
}

#[test]
fn invalid_inputs_return_err() {
    for (s, k, t, sigma) in [
        (-1.0, 100.0, 1.0, 0.2),
        (100.0, -1.0, 1.0, 0.2),
        (100.0, 100.0, -1.0, 0.2),
        (100.0, 100.0, 1.0, -0.2),
        (f64::NAN, 100.0, 1.0, 0.2),
        (100.0, 100.0, f64::NAN, 0.2),
    ] {
        let res = bs_price(s, k, t, 0.05, sigma, 0.0, OptionType::Call);
        assert!(
            matches!(res, Err(PricingError::InvalidInput(_))),
            "expected InvalidInput for (S={s}, K={k}, T={t}, sigma={sigma}), got {res:?}"
        );
    }
    assert!(d1_d2(100.0, 100.0, 0.0, 0.05, 0.2, 0.0).is_err());
    assert!(d1_d2(0.0, 100.0, 1.0, 0.05, 0.2, 0.0).is_err());
}

#[test]
fn error_display_is_informative() {
    let err = bs_price(-5.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("invalid input"), "got: {msg}");
    assert!(msg.contains("S must be >= 0"), "got: {msg}");
}

#[test]
fn price_is_monotone_in_vol_and_convex_in_strike() {
    // Property checks: vega > 0 (monotone in vol) and butterfly >= 0.
    let (s, t, r, q) = (100.0, 1.0, 0.03, 0.01);
    let mut last = -f64::INFINITY;
    for i in 1..=20 {
        let sigma = 0.05 * i as f64;
        let c = bs_price(s, 100.0, t, r, sigma, q, OptionType::Call).unwrap();
        assert!(c > last, "call price not increasing in vol at sigma={sigma}");
        last = c;
    }
    for k in (60..=140).step_by(5) {
        let k = k as f64;
        let lo = bs_price(s, k - 5.0, t, r, 0.2, q, OptionType::Call).unwrap();
        let mid = bs_price(s, k, t, r, 0.2, q, OptionType::Call).unwrap();
        let hi = bs_price(s, k + 5.0, t, r, 0.2, q, OptionType::Call).unwrap();
        assert!(
            lo - 2.0 * mid + hi >= -1e-12,
            "butterfly negative at K={k}"
        );
    }
}

#[test]
fn non_finite_inputs_are_rejected() {
    use eq_options_engine::black_scholes::{validate_inputs, validate_rates};
    let inf = f64::INFINITY;
    // Infinite S, K, T or sigma is rejected just like NaN.
    for (s, k, t, sigma) in [
        (inf, 100.0, 1.0, 0.2),
        (100.0, inf, 1.0, 0.2),
        (100.0, 100.0, inf, 0.2),
        (100.0, 100.0, 1.0, inf),
        (-inf, 100.0, 1.0, 0.2),
    ] {
        let res = bs_price(s, k, t, 0.05, sigma, 0.0, OptionType::Call);
        assert!(
            matches!(res, Err(PricingError::InvalidInput(_))),
            "expected InvalidInput for (S={s}, K={k}, T={t}, sigma={sigma}), got {res:?}"
        );
        assert!(validate_inputs(s, k, t, sigma).is_err());
    }
    // NaN / infinite r or q must not silently produce a NaN price.
    for (r, q) in [
        (f64::NAN, 0.0),
        (0.05, f64::NAN),
        (inf, 0.0),
        (0.05, -inf),
    ] {
        let res = bs_price(100.0, 100.0, 1.0, r, 0.2, q, OptionType::Call);
        assert!(
            matches!(res, Err(PricingError::InvalidInput(_))),
            "expected InvalidInput for (r={r}, q={q}), got {res:?}"
        );
        assert!(validate_rates(r, q).is_err());
    }
    assert!(validate_rates(-0.05, -0.02).is_ok()); // negative rates stay legal
}

#[test]
fn extreme_moneyness_prices_are_stable() {
    let (t, r, q, sigma) = (1.0, 0.03, 0.01, 0.2);
    // Deep ITM call (S/K = 1e4): time value vanishes, price -> parity value
    // S e^{-qT} - K e^{-rT}, with full relative accuracy (no cancellation).
    let (s, k) = (1.0e4, 1.0);
    let c = bs_price(s, k, t, r, sigma, q, OptionType::Call).unwrap();
    let parity = s * (-q * t).exp() - k * (-r * t).exp();
    assert_rel_close!(c, parity, 1e-12);
    // The matching put is worthless but must be a clean non-negative zero.
    let p = bs_price(s, k, t, r, sigma, q, OptionType::Put).unwrap();
    assert!(p >= 0.0 && p < 1e-100, "deep OTM put should underflow cleanly, got {p}");
    // Deep OTM call (S/K = 1e-4): worthless, finite, non-negative.
    let c_otm = bs_price(1.0, 1.0e4, t, r, sigma, q, OptionType::Call).unwrap();
    assert!(c_otm >= 0.0 && c_otm < 1e-100, "deep OTM call: {c_otm}");
    // Put-call parity still holds exactly at both extremes.
    for (s, k) in [(1.0e4, 1.0), (1.0, 1.0e4)] {
        let c = bs_price(s, k, t, r, sigma, q, OptionType::Call).unwrap();
        let p = bs_price(s, k, t, r, sigma, q, OptionType::Put).unwrap();
        let rhs = s * (-q * t).exp() - k * (-r * t).exp();
        assert_rel_close!(c - p, rhs, 1e-12);
    }
}

#[test]
fn very_long_and_very_short_expiries() {
    let (s, k, r, q, sigma) = (100.0, 100.0, 0.03, 0.01, 0.2);
    // 30-year option: finite, inside its static no-arbitrage bounds,
    // and parity-consistent.
    let t = 30.0;
    let c = bs_price(s, k, t, r, sigma, q, OptionType::Call).unwrap();
    let p = bs_price(s, k, t, r, sigma, q, OptionType::Put).unwrap();
    let (df_q, df_r) = ((-q * t).exp(), (-r * t).exp());
    assert!(c.is_finite() && p.is_finite());
    assert!(c <= s * df_q + 1e-12 && c >= (s * df_q - k * df_r).max(0.0) - 1e-12);
    assert_rel_close!(c - p, s * df_q - k * df_r, 1e-12);
    // T = 1e-6 (~30 seconds): price collapses to intrinsic, but stays at
    // or above the sigma->0 no-arbitrage floor (the discounted forward
    // intrinsic — which for a European put with r > 0 sits *below* the
    // undiscounted intrinsic, so `>= intrinsic` is NOT the right bound).
    let t = 1e-6;
    for (s2, ot, intrinsic) in [
        (105.0, OptionType::Call, 5.0),
        (95.0, OptionType::Put, 5.0),
    ] {
        let v = bs_price(s2, k, t, r, sigma, q, ot).unwrap();
        let floor = bs_price(s2, k, t, r, 0.0, q, ot).unwrap();
        assert!(
            (v - intrinsic).abs() < 1e-3,
            "near-expiry {ot:?}: {v} vs intrinsic {intrinsic}"
        );
        assert!(
            v >= floor - 1e-12,
            "near-expiry {ot:?}: {v} below the sigma->0 floor {floor}"
        );
    }
}
