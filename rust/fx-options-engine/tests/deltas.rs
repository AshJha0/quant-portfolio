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
