//! Regression tests for wrong numbers and silent failures at the domain edges.

use fx_options_engine::iv_core::black_implied_vol;
use fx_options_engine::{
    binomial_price, delta, gk_price, implied_vol, mc_price, strike_from_delta, DeltaConvention,
    Exercise, FxError, OptionType,
};

#[test]
fn sigma_zero_with_overflowing_forward_is_finite() {
    // (r_d - r_f) T = 800 overflows the forward; the old branch gave NaN.
    assert_eq!(gk_price(1.0, 1.0, 1000.0, 0.8, 0.0, 0.0, OptionType::Call).unwrap(), 1.0);
    assert_eq!(gk_price(1.0, 1.0, 1000.0, 0.8, 0.0, 0.0, OptionType::Put).unwrap(), 0.0);
    let a = binomial_price(1.0, 1.0, 1000.0, 0.8, 0.0, 0.0, OptionType::Call, 100, Exercise::American)
        .unwrap();
    assert_eq!(a, 1.0);
}

#[test]
fn overflowing_discount_factor_is_an_error_not_inf() {
    assert!(gk_price(1.0, 1.0, 1000.0, -0.8, 0.0, 0.2, OptionType::Put).is_err());
    assert!(implied_vol(0.5, 1.0, 1.0, 1000.0, -0.8, 0.0, OptionType::Call).is_err());
}

#[test]
fn mc_with_overflowing_drift_is_an_error() {
    let r = mc_price(1.0, 1.0, 1000.0, 0.8, 0.0, 0.2, OptionType::Call, 1000, 1, true, true);
    assert!(r.is_err(), "got {r:?}");
}

#[test]
fn pa_call_strike_solves_for_total_vol_above_37() {
    // The fold bracket used to be [-37, 20], which fails to bracket once
    // sigma sqrt(T) > 37. The derived bracket works for every v. The fold
    // strike is about F e^{v^2/2}, representable up to v ~ 37.6; beyond
    // that the engine must say so (InvalidInput), not fail in Brent.
    for &(sigma, t) in &[(37.5, 1.0), (7.5, 25.0), (0.1, 1.0), (40.0, 1.0)] {
        let target = 1e-3;
        match strike_from_delta(target, 1.1, t, 0.03, 0.01, sigma, OptionType::Call, DeltaConvention::ForwardPa) {
            // Above v ~ 30 the PA delta at the strike, (K/F) N(d2), is a
            // product of e^{+v^2/2} and e^{-v^2/2} that underflows; only
            // check that the solve itself does not fail there.
            Ok(_) if sigma * t.sqrt() > 30.0 => {}
            Ok(k) => {
                let d = delta(1.1, k, t, 0.03, 0.01, sigma, OptionType::Call, DeltaConvention::ForwardPa)
                    .unwrap();
                assert!((d - target).abs() <= 1e-10, "sigma={sigma}: {d}");
            }
            Err(FxError::InvalidInput(msg)) => {
                assert!(msg.contains("exceeds") || sigma == 40.0, "{msg}")
            }
            Err(e) => panic!("sigma={sigma} t={t}: {e}"),
        }
    }
}

#[test]
fn k90_otm_put_recovers_vol_to_1e12_and_no_premature_exit() {
    for sigma in [0.10, 0.14] {
        let put = gk_price(100.0, 90.0, 0.02, 0.03, 0.0, sigma, OptionType::Put).unwrap();
        let iv = implied_vol(put, 100.0, 90.0, 0.02, 0.03, 0.0, OptionType::Put).unwrap();
        assert!((iv - sigma).abs() <= 1e-12, "{iv} vs {sigma}");
    }
}

#[test]
fn bounds_contract() {
    let (s, k, t, rd, rf) = (1.1, 1.0, 0.5, 0.04, 0.02);
    let lower = gk_price(s, k, t, rd, rf, 0.0, OptionType::Call).unwrap();
    let upper = s * (-rf * t).exp();
    assert_eq!(implied_vol(lower, s, k, t, rd, rf, OptionType::Call).unwrap(), 0.0);
    assert!(implied_vol(upper, s, k, t, rd, rf, OptionType::Call).is_err());
    assert!(implied_vol(lower - 1e-9, s, k, t, rd, rf, OptionType::Call).is_err());
}

#[test]
fn iv_uses_at_most_ten_evaluations() {
    let df: f64 = 0.97;
    for &lk in &[-3.0, -1.0, -0.05, 0.0, 0.05, 1.0, 3.0] {
        for &t in &[1e-4, 0.25, 10.0] {
            for &sigma in &[0.01, 0.2, 2.0] {
                let k = f64::exp(lk);
                let is_call = k >= 1.0;
                let ot = if is_call { OptionType::Call } else { OptionType::Put };
                let r = -df.ln() / t;
                let price = gk_price(1.0, k, t, r, r, sigma, ot).unwrap();
                if price.is_nan() || price <= 1e-300 {
                    continue;
                }
                let mut evals = 0;
                let _ = black_implied_vol(price, 1.0, k, t, df, is_call, &mut evals);
                assert!(evals <= 10, "lk={lk} t={t} sigma={sigma}: {evals} evaluations");
            }
        }
    }
}
