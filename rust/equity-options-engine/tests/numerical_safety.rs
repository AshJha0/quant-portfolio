//! Regression tests for wrong numbers at the edges of the input domain.

use eq_options_engine::iv_core::black_implied_vol;
use eq_options_engine::{bs_price, crr_price, implied_vol, mc_price, Exercise, OptionType};

#[test]
fn sigma_zero_with_overflowing_forward_is_finite() {
    // (r - q) T = 800: the forward overflows; the old branch returned NaN.
    let c = bs_price(100.0, 100.0, 1000.0, 0.8, 0.0, 0.0, OptionType::Call).unwrap();
    assert_eq!(c, 100.0);
    let p = bs_price(100.0, 100.0, 1000.0, 0.8, 0.0, 0.0, OptionType::Put).unwrap();
    assert_eq!(p, 0.0);
    let a = crr_price(100.0, 100.0, 1000.0, 0.8, 0.0, 0.0, OptionType::Call, Exercise::American, 100)
        .unwrap();
    assert_eq!(a, 100.0);
}

#[test]
fn overflowing_discount_factor_is_an_error_not_inf() {
    assert!(bs_price(100.0, 100.0, 1000.0, -0.8, 0.2, 0.0, OptionType::Put).is_err());
    assert!(implied_vol(5.0, 100.0, 100.0, 1000.0, -0.8, 0.0, OptionType::Call).is_err());
}

#[test]
fn mc_with_overflowing_drift_is_an_error() {
    let r = mc_price(100.0, 100.0, 1000.0, 0.8, 0.2, 0.0, OptionType::Call, 1000, true, true, 1);
    assert!(r.is_err(), "got {r:?}");
}

#[test]
fn american_put_at_zero_spot_waits_when_rates_are_negative() {
    for r in [-0.02, -0.005] {
        let v = crr_price(0.0, 100.0, 1.0, r, 0.2, 0.0, OptionType::Put, Exercise::American, 100)
            .unwrap();
        let expected = 100.0 * (-r * 1.0_f64).exp();
        assert!((v - expected).abs() <= 4.0 * f64::EPSILON * expected, "{v} vs {expected}");
    }
    let v = crr_price(0.0, 100.0, 1.0, 0.03, 0.2, 0.0, OptionType::Put, Exercise::American, 100)
        .unwrap();
    assert_eq!(v, 100.0);
}

#[test]
fn k90_otm_put_recovers_vol_to_1e12() {
    let (s, k, t, r, sigma) = (100.0, 90.0, 0.02, 0.03, 0.10);
    let put = bs_price(s, k, t, r, sigma, 0.0, OptionType::Put).unwrap();
    let iv = implied_vol(put, s, k, t, r, 0.0, OptionType::Put).unwrap();
    assert!((iv - sigma).abs() <= 1e-12, "{iv}");
}

#[test]
fn k90_itm_call_is_backward_stable() {
    // One ulp of the call (1.8e-15) over vega (3.5e-12) is a vol
    // uncertainty of about 5e-4, so the quote cannot give 1e-12. The
    // returned vol must reprice the quote to a few ulps.
    let (s, k, t, r, sigma) = (100.0, 90.0, 0.02, 0.03, 0.10);
    let call = bs_price(s, k, t, r, sigma, 0.0, OptionType::Call).unwrap();
    let iv = implied_vol(call, s, k, t, r, 0.0, OptionType::Call).unwrap();
    let back = bs_price(s, k, t, r, iv, 0.0, OptionType::Call).unwrap();
    assert!((back - call).abs() <= 16.0 * f64::EPSILON * call, "{back} vs {call}");
}

#[test]
fn iv_uses_at_most_ten_evaluations() {
    let df: f64 = 0.97;
    for &lk in &[-3.0, -1.0, -0.05, 0.0, 0.05, 1.0, 3.0] {
        for &t in &[1e-4, 0.25, 10.0] {
            for &sigma in &[0.01, 0.2, 2.0] {
                let k = f64::exp(lk);
                // OTM option on a forward of 1: call above the forward, put below.
                let is_call = k >= 1.0;
                let ot = if is_call { OptionType::Call } else { OptionType::Put };
                let r = -df.ln() / t;
                let price = bs_price(df, k, t, r, sigma, 0.0, ot).unwrap();
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
