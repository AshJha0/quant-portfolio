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

/// Every public entry point must reject NaN and +/-inf in every float
/// argument.
///
/// This is the single highest-value guard in the crate. A validation
/// written as `if x <= 0.0 { return Err(..) }` silently *accepts* NaN,
/// because every comparison against NaN is false — so a corrupt rate or
/// spot flows straight through the pricer and comes back as a NaN
/// "price". A NaN breaches no limit, colours no traffic light and
/// aggregates into a NaN book P&L, which is strictly worse than a loud
/// failure.
#[test]
fn every_entry_point_rejects_non_finite_inputs() {
    use fx_options_engine::{
        analytic_greeks, atm_dns_strike, atm_forward_strike, binomial_price, black76_from_spot,
        black76_price, cip_forward, d1, d2, finite_difference_greeks, forward_points,
        forward_to_spot_delta, mc_price, premium_adjust_spot_delta, spot_to_forward_delta,
        strike_from_delta, synthetic_forward_from_options, Exercise,
    };
    let bad = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    let (s, k, t, rd, rf, sig) = (1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925);
    let ot = OptionType::Call;

    for &b in &bad {
        // Six-argument pricing surface: poison each argument in turn.
        for slot in 0..6 {
            let (s_, k_, t_, rd_, rf_, sig_) = (
                if slot == 0 { b } else { s },
                if slot == 1 { b } else { k },
                if slot == 2 { b } else { t },
                if slot == 3 { b } else { rd },
                if slot == 4 { b } else { rf },
                if slot == 5 { b } else { sig },
            );
            let label = format!("slot {slot} = {b}");
            assert!(gk_price(s_, k_, t_, rd_, rf_, sig_, ot).is_err(), "gk_price {label}");
            assert!(d1(s_, k_, t_, rd_, rf_, sig_).is_err(), "d1 {label}");
            assert!(d2(s_, k_, t_, rd_, rf_, sig_).is_err(), "d2 {label}");
            assert!(
                analytic_greeks(s_, k_, t_, rd_, rf_, sig_, ot).is_err(),
                "analytic_greeks {label}"
            );
            assert!(
                finite_difference_greeks(s_, k_, t_, rd_, rf_, sig_, ot, 1e-5).is_err(),
                "finite_difference_greeks {label}"
            );
            assert!(
                delta(s_, k_, t_, rd_, rf_, sig_, ot, DeltaConvention::Spot).is_err(),
                "delta {label}"
            );
            assert!(
                binomial_price(s_, k_, t_, rd_, rf_, sig_, ot, 64, Exercise::American).is_err(),
                "binomial_price {label}"
            );
            assert!(
                mc_price(s_, k_, t_, rd_, rf_, sig_, ot, 256, 1, true, true).is_err(),
                "mc_price {label}"
            );
            assert!(
                black76_from_spot(s_, k_, t_, rd_, rf_, sig_, ot).is_err(),
                "black76_from_spot {label}"
            );
            if slot < 5 {
                // implied_vol solves *for* sigma, so it has no sigma slot.
                assert!(
                    implied_vol(0.02, s_, k_, t_, rd_, rf_, ot).is_err(),
                    "implied_vol {label}"
                );
            }
        }
        // Non-six-argument surfaces.
        assert!(black76_price(b, k, t, rd, sig, ot).is_err(), "black76 F={b}");
        assert!(black76_price(1.11, b, t, rd, sig, ot).is_err(), "black76 K={b}");
        assert!(black76_price(1.11, k, b, rd, sig, ot).is_err(), "black76 T={b}");
        assert!(black76_price(1.11, k, t, b, sig, ot).is_err(), "black76 r_d={b}");
        assert!(black76_price(1.11, k, t, rd, b, ot).is_err(), "black76 sigma={b}");
        assert!(cip_forward(b, t, rd, rf).is_err(), "cip_forward S={b}");
        assert!(cip_forward(s, b, rd, rf).is_err(), "cip_forward T={b}");
        assert!(cip_forward(s, t, b, rf).is_err(), "cip_forward r_d={b}");
        assert!(forward_points(s, t, rd, rf, b).is_err(), "forward_points pip={b}");
        assert!(
            synthetic_forward_from_options(b, 0.01, k, t, rd).is_err(),
            "synthetic_forward call={b}"
        );
        assert!(
            synthetic_forward_from_options(0.02, b, k, t, rd).is_err(),
            "synthetic_forward put={b}"
        );
        assert!(atm_forward_strike(b, t, rd, rf).is_err(), "atm_forward_strike S={b}");
        assert!(
            atm_dns_strike(s, t, rd, rf, b, DeltaConvention::SpotPa).is_err(),
            "atm_dns_strike sigma={b}"
        );
        assert!(
            strike_from_delta(b, s, t, rd, rf, sig, ot, DeltaConvention::Spot).is_err(),
            "strike_from_delta target={b}"
        );
        // Delta conversions: a NaN delta must not be scaled into a NaN
        // hedge ratio.
        assert!(spot_to_forward_delta(b, t, rf).is_err(), "spot_to_forward_delta {b}");
        assert!(forward_to_spot_delta(b, t, rf).is_err(), "forward_to_spot_delta {b}");
        assert!(premium_adjust_spot_delta(b, 0.02, s).is_err(), "pa delta={b}");
        assert!(premium_adjust_spot_delta(0.5, b, s).is_err(), "pa price={b}");
        assert!(premium_adjust_spot_delta(0.5, 0.02, b).is_err(), "pa spot={b}");
        // Implied vol: a NaN premium is a corrupt quote, not a vol.
        assert!(implied_vol(b, s, k, t, rd, rf, ot).is_err(), "implied_vol price={b}");
        // FD bump size.
        assert!(
            finite_difference_greeks(s, k, t, rd, rf, sig, ot, b).is_err(),
            "fd rel_bump={b}"
        );
    }
    // A zero or negative FD bump divides by zero: also rejected.
    assert!(finite_difference_greeks(s, k, t, rd, rf, sig, ot, 0.0).is_err());
    assert!(finite_difference_greeks(s, k, t, rd, rf, sig, ot, -1e-5).is_err());
    // T = 0 collapses the theta bump min(1e-6, T/4) to zero -> 0/0.
    assert!(finite_difference_greeks(s, k, 0.0, rd, rf, sig, ot, 1e-5).is_err());
    // Sanity: the same call with clean inputs still works.
    assert!(finite_difference_greeks(s, k, t, rd, rf, sig, ot, 1e-5).is_ok());
}

#[test]
fn hundredfold_moneyness_and_multidecade_tenors() {
    // 100x in both directions x {1d, 1y, 30y, 100y}: prices finite,
    // non-negative, inside the static bounds, and parity-consistent.
    let (rd, rf, sig) = (0.03, 0.01, 0.12);
    for &(s, k) in &[(1.0, 100.0), (100.0, 1.0), (1.0, 1.0)] {
        for &t in &[1.0 / 365.0, 1.0, 30.0, 100.0] {
            let c = gk_price(s, k, t, rd, rf, sig, OptionType::Call).unwrap();
            let p = gk_price(s, k, t, rd, rf, sig, OptionType::Put).unwrap();
            assert!(c.is_finite() && p.is_finite(), "S={s} K={k} T={t}");
            assert!(c >= 0.0 && p >= 0.0, "S={s} K={k} T={t}: {c}, {p}");
            // Static bounds: 0 <= C <= S e^{-r_f T}, 0 <= P <= K e^{-r_d T}.
            assert!(c <= s * (-rf * t).exp() * (1.0 + 1e-12), "call bound S={s} K={k} T={t}");
            assert!(p <= k * (-rd * t).exp() * (1.0 + 1e-12), "put bound S={s} K={k} T={t}");
            // Two-rate parity, scale-relative (a 100x book is a large
            // notional; an absolute tolerance would be a different, and
            // much harsher, demand at S = 100 than at S = 1).
            let parity = s * (-rf * t).exp() - k * (-rd * t).exp();
            let scale = s.max(k).max(1.0);
            assert!(
                (c - p - parity).abs() <= 1e-13 * scale,
                "parity S={s} K={k} T={t}: {} vs {parity}",
                c - p
            );
        }
    }
    // Deep ITM/OTM Greeks stay finite and correctly signed at 100x.
    use fx_options_engine::analytic_greeks;
    for &(s, k) in &[(1.0, 100.0), (100.0, 1.0)] {
        let g = analytic_greeks(s, k, 1.0, rd, rf, sig, OptionType::Call).unwrap();
        for (name, v) in [
            ("price", g.price),
            ("delta_spot", g.delta_spot),
            ("gamma", g.gamma),
            ("vega", g.vega),
            ("theta", g.theta),
            ("rho_domestic", g.rho_domestic),
            ("rho_foreign", g.rho_foreign),
            ("vanna", g.vanna),
            ("volga", g.volga),
        ] {
            assert!(v.is_finite(), "{name} not finite at S={s}, K={k}: {v}");
        }
        assert!(g.gamma >= 0.0 && g.vega >= 0.0);
        assert!(g.delta_spot >= 0.0 && g.delta_spot <= (-rf * 1.0f64).exp() + 1e-15);
    }
}

#[test]
fn zero_and_negative_rate_regimes_price_consistently() {
    // r_d = r_f = 0 (pegged/zero-rate), one leg zero, and deeply negative
    // rates: all legal, all must satisfy parity and round-trip through
    // implied vol.
    for &(rd, rf) in &[
        (0.0, 0.0),
        (0.0, 0.02),
        (0.02, 0.0),
        (-0.0075, -0.0050),
        (-0.05, 0.05),
        (0.05, -0.05),
    ] {
        let (s, k, t, sig) = (1.10, 1.05, 0.75, 0.09);
        let c = gk_price(s, k, t, rd, rf, sig, OptionType::Call).unwrap();
        let p = gk_price(s, k, t, rd, rf, sig, OptionType::Put).unwrap();
        let parity = s * (-rf * t).exp() - k * (-rd * t).exp();
        assert_close(c - p, parity, 1e-14, "parity under (r_d, r_f)");
        for (px, ot) in [(c, OptionType::Call), (p, OptionType::Put)] {
            let iv = implied_vol(px, s, k, t, rd, rf, ot).unwrap();
            assert_close(iv, sig, 1e-9, "IV round trip under negative rates");
        }
        // The forward and the DNS strike stay positive and finite.
        let f = fx_options_engine::cip_forward(s, t, rd, rf).unwrap();
        assert!(f > 0.0 && f.is_finite());
    }
}
