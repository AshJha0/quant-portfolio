//! Property-based tests (proptest): parity, monotonicity, convexity, delta
//! bounds and the implied-vol round trip.

use eq_options_engine::{bs_greeks, bs_price, implied_vol, OptionType};
use proptest::prelude::*;

const EPS: f64 = f64::EPSILON;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn put_call_parity(s in 1.0..1000.0f64, m in 0.5..2.0f64, t in 0.01..5.0f64,
                       r in -0.02..0.10f64, q in 0.0..0.08f64, sigma in 0.05..1.0f64) {
        let k = s * m;
        let c = bs_price(s, k, t, r, sigma, q, OptionType::Call).unwrap();
        let p = bs_price(s, k, t, r, sigma, q, OptionType::Put).unwrap();
        let rhs = s * (-q * t).exp() - k * (-r * t).exp();
        prop_assert!((c - p - rhs).abs() <= 64.0 * EPS * s.max(k));
    }

    #[test]
    fn price_increases_with_vol(s in 1.0..1000.0f64, m in 0.5..2.0f64, t in 0.01..5.0f64,
                                r in -0.02..0.10f64, q in 0.0..0.08f64, sigma in 0.05..1.0f64,
                                bump in 1.001..1.5f64) {
        let k = s * m;
        for ot in [OptionType::Call, OptionType::Put] {
            let lo = bs_price(s, k, t, r, sigma, q, ot).unwrap();
            let hi = bs_price(s, k, t, r, sigma * bump, q, ot).unwrap();
            prop_assert!(hi >= lo - 16.0 * EPS * s.max(k));
        }
    }

    #[test]
    fn monotone_and_convex_in_strike(s in 1.0..1000.0f64, m in 0.5..2.0f64, t in 0.01..5.0f64,
                                     r in -0.02..0.10f64, q in 0.0..0.08f64, sigma in 0.05..1.0f64,
                                     step in 1.001..1.3f64) {
        let k1 = s * m;
        let k2 = k1 * step;
        let k3 = 2.0 * k2 - k1;
        let noise = 64.0 * EPS * s.max(k3);
        for ot in [OptionType::Call, OptionType::Put] {
            let p: Vec<f64> = [k1, k2, k3]
                .iter()
                .map(|&k| bs_price(s, k, t, r, sigma, q, ot).unwrap())
                .collect();
            if ot == OptionType::Call {
                prop_assert!(p[1] <= p[0] + noise);
            } else {
                prop_assert!(p[1] >= p[0] - noise);
            }
            prop_assert!(p[0] + p[2] - 2.0 * p[1] >= -4.0 * noise);
        }
    }

    #[test]
    fn delta_bounds(s in 1.0..1000.0f64, m in 0.5..2.0f64, t in 0.01..5.0f64,
                    r in -0.02..0.10f64, q in 0.0..0.08f64, sigma in 0.05..1.0f64) {
        let k = s * m;
        let dq = (-q * t).exp();
        let c = bs_greeks(s, k, t, r, sigma, q, OptionType::Call).unwrap().delta;
        let p = bs_greeks(s, k, t, r, sigma, q, OptionType::Put).unwrap().delta;
        prop_assert!((0.0..=dq * (1.0 + 4.0 * EPS)).contains(&c));
        prop_assert!((-dq * (1.0 + 4.0 * EPS)..=0.0).contains(&p));
        prop_assert!(((c - p) - dq).abs() <= 4.0 * EPS);
    }

    #[test]
    fn iv_round_trip(s in 1.0..1000.0f64, m in 0.5..2.0f64, t in 0.01..5.0f64,
                     r in -0.02..0.10f64, q in 0.0..0.08f64, sigma in 0.05..1.0f64,
                     call in any::<bool>()) {
        let k = s * m;
        let ot = if call { OptionType::Call } else { OptionType::Put };
        let price = bs_price(s, k, t, r, sigma, q, ot).unwrap();
        let lower = bs_price(s, k, t, r, 0.0, q, ot).unwrap();
        prop_assume!(price - lower > 1e-12 * s.max(k));
        let vega = bs_greeks(s, k, t, r, sigma, q, ot).unwrap().vega;
        let iv = implied_vol(price, s, k, t, r, q, ot).unwrap();
        prop_assert!((iv - sigma).abs() <= 8.0 * EPS * s.max(k) / vega + 2e-12 * sigma);
    }
}
