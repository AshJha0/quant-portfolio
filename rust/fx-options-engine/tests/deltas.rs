//! Delta conventions: relations, inversions, ATM strikes, failure modes.

mod common;
use common::assert_close;

use fx_options_engine::{
    atm_dns_strike, atm_forward_strike, cip_forward, delta, forward_to_spot_delta, gk_price,
    premium_adjust_spot_delta, spot_to_forward_delta, strike_from_delta, DeltaConvention,
    FxError, OptionType,
};

const MKT: (f64, f64, f64, f64, f64, f64) = (1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925);

#[test]
fn forward_delta_equals_spot_delta_times_erf_t() {
    let (s, k, t, rd, rf, sig) = MKT;
    for ty in [OptionType::Call, OptionType::Put] {
        let ds = delta(s, k, t, rd, rf, sig, ty, DeltaConvention::Spot).unwrap();
        let df = delta(s, k, t, rd, rf, sig, ty, DeltaConvention::Forward).unwrap();
        assert_close(df, ds * (rf * t).exp(), 1e-15, "fwd = spot * e^{r_f T}");
        assert_close(
            spot_to_forward_delta(ds, t, rf).unwrap(),
            df,
            1e-15,
            "spot_to_forward",
        );
        assert_close(
            forward_to_spot_delta(df, t, rf).unwrap(),
            ds,
            1e-15,
            "forward_to_spot",
        );
    }
}

#[test]
fn premium_adjusted_below_unadjusted_for_calls() {
    let (s, k, t, rd, rf, sig) = MKT;
    let spot = delta(s, k, t, rd, rf, sig, OptionType::Call, DeltaConvention::Spot).unwrap();
    let spot_pa = delta(s, k, t, rd, rf, sig, OptionType::Call, DeltaConvention::SpotPa).unwrap();
    let fwd = delta(s, k, t, rd, rf, sig, OptionType::Call, DeltaConvention::Forward).unwrap();
    let fwd_pa =
        delta(s, k, t, rd, rf, sig, OptionType::Call, DeltaConvention::ForwardPa).unwrap();
    assert!(spot_pa < spot, "PA spot {spot_pa} !< spot {spot}");
    assert!(fwd_pa < fwd, "PA forward {fwd_pa} !< forward {fwd}");
}

#[test]
fn premium_adjustment_identity_delta_pa_equals_delta_minus_premium_over_s() {
    let (s, k, t, rd, rf, sig) = MKT;
    for ty in [OptionType::Call, OptionType::Put] {
        let price = gk_price(s, k, t, rd, rf, sig, ty).unwrap();
        let spot = delta(s, k, t, rd, rf, sig, ty, DeltaConvention::Spot).unwrap();
        let spot_pa = delta(s, k, t, rd, rf, sig, ty, DeltaConvention::SpotPa).unwrap();
        assert_close(
            spot_pa,
            premium_adjust_spot_delta(spot, price, s).unwrap(),
            1e-14,
            "delta_pa = delta - V/S",
        );
    }
}

#[test]
fn put_deltas_negative_call_deltas_positive() {
    let (s, k, t, rd, rf, sig) = MKT;
    for conv in DeltaConvention::ALL {
        assert!(delta(s, k, t, rd, rf, sig, OptionType::Call, conv).unwrap() > 0.0);
        assert!(delta(s, k, t, rd, rf, sig, OptionType::Put, conv).unwrap() < 0.0);
    }
}

#[test]
fn strike_from_delta_round_trips_all_four_conventions_to_1e8() {
    let (s, _, t, rd, rf, sig) = MKT;
    for conv in DeltaConvention::ALL {
        for &(ty, target) in &[
            (OptionType::Call, 0.25),
            (OptionType::Call, 0.10),
            (OptionType::Put, -0.25),
            (OptionType::Put, -0.10),
        ] {
            let k = strike_from_delta(target, s, t, rd, rf, sig, ty, conv).unwrap();
            let d = delta(s, k, t, rd, rf, sig, ty, conv).unwrap();
            assert_close(
                d,
                target,
                1e-8,
                &format!("round trip {conv:?} {ty:?} {target}"),
            );
        }
    }
}

#[test]
fn pa_call_takes_the_larger_strike_branch() {
    // Market convention: the 25d PA call strike sits on the decreasing
    // (high-strike, OTM) branch — above the PA delta-neutral ATM strike.
    let (s, _, t, rd, rf, sig) = MKT;
    let k25 = strike_from_delta(
        0.25,
        s,
        t,
        rd,
        rf,
        sig,
        OptionType::Call,
        DeltaConvention::ForwardPa,
    )
    .unwrap();
    let k_dns_pa = atm_dns_strike(s, t, rd, rf, sig, DeltaConvention::ForwardPa).unwrap();
    assert!(k25 > k_dns_pa, "PA 25d call strike {k25} !> DNS {k_dns_pa}");
}

#[test]
fn pa_call_delta_above_maximum_attainable_errs() {
    let (s, _, t, rd, rf, sig) = MKT;
    let res = strike_from_delta(
        0.95,
        s,
        t,
        rd,
        rf,
        sig,
        OptionType::Call,
        DeltaConvention::ForwardPa,
    );
    assert!(matches!(res, Err(FxError::InvalidInput(_))));
}

#[test]
fn wrong_sign_and_out_of_range_delta_err() {
    let (s, _, t, rd, rf, sig) = MKT;
    for conv in DeltaConvention::ALL {
        assert!(strike_from_delta(-0.25, s, t, rd, rf, sig, OptionType::Call, conv).is_err());
        assert!(strike_from_delta(0.25, s, t, rd, rf, sig, OptionType::Put, conv).is_err());
    }
    assert!(
        strike_from_delta(1.5, s, t, rd, rf, sig, OptionType::Call, DeltaConvention::Forward)
            .is_err()
    );
    assert!(strike_from_delta(
        f64::NAN,
        s,
        t,
        rd,
        rf,
        sig,
        OptionType::Call,
        DeltaConvention::Spot
    )
    .is_err());
}

#[test]
fn strike_from_delta_requires_positive_t_and_sigma() {
    let (s, _, _, rd, rf, sig) = MKT;
    assert!(
        strike_from_delta(0.25, s, 0.0, rd, rf, sig, OptionType::Call, DeltaConvention::Spot)
            .is_err()
    );
    assert!(
        strike_from_delta(0.25, s, 0.5, rd, rf, 0.0, OptionType::Call, DeltaConvention::Spot)
            .is_err()
    );
}

#[test]
fn dns_strike_zeroes_the_straddle_delta_in_all_conventions() {
    let (s, _, t, rd, rf, sig) = MKT;
    for conv in DeltaConvention::ALL {
        let k = atm_dns_strike(s, t, rd, rf, sig, conv).unwrap();
        let dc = delta(s, k, t, rd, rf, sig, OptionType::Call, conv).unwrap();
        let dp = delta(s, k, t, rd, rf, sig, OptionType::Put, conv).unwrap();
        assert_close(dc + dp, 0.0, 1e-12, &format!("DNS straddle delta {conv:?}"));
    }
}

#[test]
fn dns_formulae_and_ordering_around_the_forward() {
    let (s, _, t, rd, rf, sig) = MKT;
    let f = cip_forward(s, t, rd, rf).unwrap();
    let k_un = atm_dns_strike(s, t, rd, rf, sig, DeltaConvention::Spot).unwrap();
    let k_pa = atm_dns_strike(s, t, rd, rf, sig, DeltaConvention::SpotPa).unwrap();
    assert_close(k_un, f * (0.5 * sig * sig * t).exp(), 1e-14, "DNS unadjusted");
    assert_close(k_pa, f * (-0.5 * sig * sig * t).exp(), 1e-14, "DNS pa");
    assert!(k_un > f && k_pa < f, "DNS strikes straddle the forward");
}

#[test]
fn atm_forward_strike_is_cip_forward() {
    let (s, _, t, rd, rf, _) = MKT;
    assert_close(
        atm_forward_strike(s, t, rd, rf).unwrap(),
        cip_forward(s, t, rd, rf).unwrap(),
        1e-15,
        "ATM-forward strike",
    );
}

#[test]
fn boundary_delta_values_are_rejected_not_silently_clamped() {
    // The attainable open range is (0, 1) in forward-equivalent terms.
    // The CLOSED boundaries have no finite strike: delta = 0 needs
    // K = +inf, delta = 1 needs K = 0. Both must be errors, and so must
    // anything outside. Silently clamping would hand the desk a strike
    // that does not reprice to the requested delta.
    let (s, _, t, rd, rf, sig) = MKT;
    for conv in DeltaConvention::ALL {
        for &bad in &[0.0, 1.0, 1.0 + 1e-12, 2.0, 1e300] {
            assert!(
                strike_from_delta(bad, s, t, rd, rf, sig, OptionType::Call, conv).is_err(),
                "call delta {bad} under {conv:?} should be rejected"
            );
        }
        for &bad in &[0.0, -1.0, -1.0 - 1e-12, -2.0, -1e300] {
            assert!(
                strike_from_delta(bad, s, t, rd, rf, sig, OptionType::Put, conv).is_err(),
                "put delta {bad} under {conv:?} should be rejected"
            );
        }
        // Just inside the boundary still solves, and round-trips.
        for &(target, ot) in &[(1e-4, OptionType::Call), (-1e-4, OptionType::Put)] {
            if let Ok(k) = strike_from_delta(target, s, t, rd, rf, sig, ot, conv) {
                assert!(k > 0.0 && k.is_finite(), "{conv:?} {ot:?}: strike {k}");
                let back = delta(s, k, t, rd, rf, sig, ot, conv).unwrap();
                assert_close(
                    back,
                    target,
                    1e-8,
                    &format!("near-boundary round trip {conv:?} {ot:?}"),
                );
            }
        }
    }
    // Spot-delta targets that are attainable as spot deltas but NOT as
    // forward-equivalent deltas (|delta e^{r_f T}| >= 1) are rejected
    // with a message naming the forward-equivalent value.
    let big = (-rf * t).exp() * 0.999_999;
    let res = strike_from_delta(big, s, t, rd, rf, sig, OptionType::Call, DeltaConvention::Spot);
    assert!(res.is_ok(), "just-attainable spot delta should solve, got {res:?}");
    let over = (-rf * t).exp() * 1.000_001;
    let res = strike_from_delta(over, s, t, rd, rf, sig, OptionType::Call, DeltaConvention::Spot);
    assert!(matches!(res, Err(FxError::InvalidInput(_))), "got {res:?}");
}

#[test]
fn delta_stays_inside_its_theoretical_bounds_at_extreme_moneyness() {
    // 100x moneyness in both directions: every convention's delta must
    // stay inside its bound and never become NaN. Deep ITM call spot
    // delta -> e^{-r_f T}; deep OTM -> 0.
    let (_, _, t, rd, rf, sig) = MKT;
    let df_f = (-rf * t).exp();
    for &(s, k) in &[(1.10, 110.0), (110.0, 1.10), (1.10, 1.12)] {
        let f = cip_forward(s, t, rd, rf).unwrap();
        for conv in DeltaConvention::ALL {
            let dc = delta(s, k, t, rd, rf, sig, OptionType::Call, conv).unwrap();
            let dp = delta(s, k, t, rd, rf, sig, OptionType::Put, conv).unwrap();
            assert!(dc.is_finite() && dp.is_finite(), "S={s} K={k} {conv:?}");
            assert!(dc >= 0.0 && dp <= 0.0, "S={s} K={k} {conv:?}: {dc}, {dp}");
            // Upper bound: unadjusted deltas are capped at the discount
            // factor (1 for forward deltas); premium-adjusted deltas at
            // (K/F) times the same factor.
            let cap = match conv {
                DeltaConvention::Spot => df_f,
                DeltaConvention::Forward => 1.0,
                DeltaConvention::SpotPa => df_f * (k / f),
                DeltaConvention::ForwardPa => k / f,
            };
            assert!(dc <= cap * (1.0 + 1e-12), "S={s} K={k} {conv:?}: {dc} > {cap}");
            assert!(-dp <= cap * (1.0 + 1e-12), "S={s} K={k} {conv:?}: {dp}");
        }
    }
    // Deep ITM call spot delta pins to e^{-r_f T}; deep OTM to 0.
    let itm = delta(110.0, 1.10, t, rd, rf, sig, OptionType::Call, DeltaConvention::Spot).unwrap();
    assert_close(itm, df_f, 1e-12, "deep ITM spot delta");
    let otm = delta(1.10, 110.0, t, rd, rf, sig, OptionType::Call, DeltaConvention::Spot).unwrap();
    assert!(otm >= 0.0 && otm < 1e-30, "deep OTM spot delta {otm}");
}

#[test]
fn premium_adjustment_rejects_corrupt_inputs() {
    // A NaN premium (a stale/void quote) must not turn a hedge ratio into
    // a silent NaN: `NaN <= limit` and `NaN > limit` are BOTH false, so a
    // NaN delta passes every risk check without being seen.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(premium_adjust_spot_delta(bad, 0.02, 1.10).is_err(), "delta {bad}");
        assert!(premium_adjust_spot_delta(0.5, bad, 1.10).is_err(), "price {bad}");
        assert!(premium_adjust_spot_delta(0.5, 0.02, bad).is_err(), "spot {bad}");
        assert!(spot_to_forward_delta(bad, 0.5, 0.02).is_err(), "s2f {bad}");
        assert!(forward_to_spot_delta(bad, 0.5, 0.02).is_err(), "f2s {bad}");
    }
    assert!(premium_adjust_spot_delta(0.5, 0.02, 0.0).is_err());
    assert!(premium_adjust_spot_delta(0.5, 0.02, -1.0).is_err());
    // Clean inputs still work and satisfy the defining identity.
    let (s, k, t, rd, rf, sig) = MKT;
    let ds = delta(s, k, t, rd, rf, sig, OptionType::Call, DeltaConvention::Spot).unwrap();
    let px = gk_price(s, k, t, rd, rf, sig, OptionType::Call).unwrap();
    let pa = premium_adjust_spot_delta(ds, px, s).unwrap();
    let direct = delta(s, k, t, rd, rf, sig, OptionType::Call, DeltaConvention::SpotPa).unwrap();
    assert_close(pa, direct, 1e-14, "premium-adjustment identity");
}
