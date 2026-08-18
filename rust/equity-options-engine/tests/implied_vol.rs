//! Implied volatility: round trips, arbitrage-bound rejection, edge cases.

mod common;

use eq_options_engine::{bs_price, implied_vol, OptionType, PricingError};

#[test]
fn round_trip_recovers_sigma_to_1e8_across_grid() {
    // price(sigma) -> implied_vol -> sigma, across moneyness 0.5x-2.0x,
    // expiries from ~2 weeks to 2 years, vols 5%-80%.
    for &s in &[100.0] {
        for &k in &[50.0, 80.0, 100.0, 120.0, 200.0] {
            for &t in &[0.04, 0.25, 1.0, 2.0] {
                for &sigma in &[0.05, 0.2, 0.5, 0.8] {
                    for &(r, q) in &[(0.03, 0.0), (0.0, 0.02), (-0.01, 0.01)] {
                        for ot in [OptionType::Call, OptionType::Put] {
                            let price = bs_price(s, k, t, r, sigma, q, ot).unwrap();
                            // Identifiability guard: recovering sigma to
                            // 1e-8 from a price solved to 1e-10 requires
                            // vega > 1e-10 / 1e-8 = 1e-2. Deep-ITM wings
                            // below that carry no vol information (the
                            // Python reference behaves identically at its
                            // own tolerance floor).
                            let vega = eq_options_engine::bs_greeks(s, k, t, r, sigma, q, ot)
                                .unwrap()
                                .vega;
                            if vega < 1e-2 {
                                continue;
                            }
                            let iv = implied_vol(price, s, k, t, r, q, ot)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "IV failed at (K={k}, T={t}, sigma={sigma}, {ot:?}): {e}"
                                    )
                                });
                            assert!(
                                (iv - sigma).abs() < 1e-8,
                                "round trip (K={k}, T={t}, r={r}, q={q}, {ot:?}): \
                                 {iv} vs {sigma} (diff {:.3e})",
                                (iv - sigma).abs()
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn sub_intrinsic_prices_are_rejected() {
    // Deep ITM call quoted below its discounted-forward intrinsic.
    let (s, k, t, r, q) = (150.0, 100.0, 1.0, 0.05, 0.0);
    let lower = bs_price(s, k, t, r, 0.0, q, OptionType::Call).unwrap();
    for bad in [0.0, lower - 1.0, lower] {
        let res = implied_vol(bad, s, k, t, r, q, OptionType::Call);
        assert!(
            matches!(res, Err(PricingError::ArbitrageBound(_))),
            "expected ArbitrageBound for price {bad}, got {res:?}"
        );
    }
}

#[test]
fn above_upper_bound_prices_are_rejected() {
    // A call can never be worth more than S e^{-qT}.
    let res = implied_vol(101.0, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call);
    assert!(matches!(res, Err(PricingError::ArbitrageBound(_))));
    // A put can never be worth more than K e^{-rT}.
    let res = implied_vol(100.0, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Put);
    assert!(matches!(res, Err(PricingError::ArbitrageBound(_))));
}

#[test]
fn expired_or_degenerate_inputs_are_rejected() {
    assert!(implied_vol(5.0, 100.0, 100.0, 0.0, 0.05, 0.0, OptionType::Call).is_err());
    assert!(implied_vol(5.0, 0.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call).is_err());
    assert!(implied_vol(5.0, 100.0, 0.0, 1.0, 0.05, 0.0, OptionType::Call).is_err());
    assert!(implied_vol(f64::NAN, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call).is_err());
    assert!(implied_vol(5.0, -1.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call).is_err());
}

#[test]
fn very_high_vol_premiums_are_bracketed() {
    // sigma = 3.0 sits above the initial [1e-9, 10] midpoint region and
    // exercises the Newton path near the top of the bracket.
    let price = bs_price(100.0, 100.0, 1.0, 0.05, 3.0, 0.0, OptionType::Call).unwrap();
    let iv = implied_vol(price, 100.0, 100.0, 1.0, 0.05, 0.0, OptionType::Call).unwrap();
    assert!((iv - 3.0).abs() < 1e-7, "recovered {iv}");
}

#[test]
fn deep_wings_still_converge() {
    // Tiny-vega regions force the bisection fallback.
    let (s, t, r, q) = (100.0, 0.1, 0.02, 0.0);
    for (k, ot) in [(140.0, OptionType::Call), (60.0, OptionType::Put)] {
        let sigma = 0.35;
        let price = bs_price(s, k, t, r, sigma, q, ot).unwrap();
        let iv = implied_vol(price, s, k, t, r, q, ot).unwrap();
        assert!(
            (iv - sigma).abs() < 1e-6,
            "wing (K={k}, {ot:?}): {iv} vs {sigma}"
        );
    }
}
