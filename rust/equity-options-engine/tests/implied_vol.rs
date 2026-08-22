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

#[test]
fn vol_above_initial_bracket_top_is_recovered() {
    // sigma = 15 sits above the initial [1e-9, 10] bracket, forcing the
    // doubling expansion toward the 1e3 cap before the solve.
    let price = bs_price(100.0, 100.0, 0.5, 0.02, 15.0, 0.0, OptionType::Call).unwrap();
    let iv = implied_vol(price, 100.0, 100.0, 0.5, 0.02, 0.0, OptionType::Call).unwrap();
    assert!((iv - 15.0).abs() < 1e-6, "recovered {iv}");
}

#[test]
fn non_finite_price_and_rates_are_rejected() {
    let ot = OptionType::Call;
    assert!(implied_vol(f64::INFINITY, 100.0, 100.0, 1.0, 0.05, 0.0, ot).is_err());
    assert!(implied_vol(5.0, 100.0, 100.0, 1.0, f64::NAN, 0.0, ot).is_err());
    assert!(implied_vol(5.0, 100.0, 100.0, 1.0, 0.05, f64::INFINITY, ot).is_err());
}

#[test]
fn tiny_vol_and_near_expiry_are_recovered() {
    // vol -> 0 and T -> 0 both push the premium onto the sigma->0
    // arbitrage floor, where vega collapses and Newton must hand over to
    // bisection. Anything the solver *accepts* here has to round-trip;
    // anything it cannot identify has to be a clean ArbitrageBound /
    // NoConvergence error, never a silent wrong vol.
    let (s, k, r, q) = (100.0, 100.0, 0.02, 0.0);
    for &t in &[1.0 / 365.0, 1.0 / 52.0, 0.25, 1.0] {
        for &sigma in &[1e-3, 1e-2, 0.05] {
            let price = bs_price(s, k, t, r, sigma, q, OptionType::Call).unwrap();
            match implied_vol(price, s, k, t, r, q, OptionType::Call) {
                Ok(iv) => {
                    // Price tolerance is absolute (1e-10), so the vol
                    // tolerance it buys scales as 1e-10 / vega.
                    let vega = eq_options_engine::bs_greeks(s, k, t, r, sigma, q,
                                                            OptionType::Call)
                        .unwrap()
                        .vega;
                    let vol_tol = (1e-10 / vega).max(1e-12) * 10.0;
                    assert!(
                        (iv - sigma).abs() < vol_tol.max(1e-8),
                        "T={t}, sigma={sigma}: recovered {iv} (tol {vol_tol:.2e})"
                    );
                }
                Err(PricingError::ArbitrageBound(_)) | Err(PricingError::NoConvergence(_)) => {}
                Err(e) => panic!("T={t}, sigma={sigma}: unexpected error {e}"),
            }
        }
    }
    // Right at the solver's lower cap the premium is numerically
    // indistinguishable from the sigma->0 floor: that must be rejected as
    // an arbitrage bound, not solved to a fictitious vol.
    let floor = bs_price(s, k, 1.0, r, 0.0, q, OptionType::Call).unwrap();
    let res = implied_vol(floor, s, k, 1.0, r, q, OptionType::Call);
    assert!(matches!(res, Err(PricingError::ArbitrageBound(_))), "got {res:?}");
    // Just inside the floor the solver returns a small positive vol that
    // reprices back to the quote within the solver's price tolerance.
    let quote = floor + 1e-6;
    let iv = implied_vol(quote, s, k, 1.0, r, q, OptionType::Call).unwrap();
    assert!(iv > 0.0 && iv < 0.05, "expected a small positive vol, got {iv}");
    let repriced = bs_price(s, k, 1.0, r, iv, q, OptionType::Call).unwrap();
    assert!((repriced - quote).abs() < 1e-9, "reprice {repriced} vs quote {quote}");
}

#[test]
fn long_dated_high_vol_flat_vega_regime_stays_accurate() {
    // S=K, T=25y, sigma=300%: |d1| ~ 7.7, so vega ~ exp(-d1^2/2)
    // underflows towards zero and the price sits within double-precision
    // noise of the sigma->inf arbitrage bound (K exp(-rT) for the put).
    // A solver that stops its Newton loop the moment the *price* residual
    // is below `IV_PRICE_TOL` can declare convergence while sigma is
    // still off by whole vol points, because that tiny price residual
    // maps through a near-zero vega to a large sigma residual. The
    // bracket must be refined by bisection all the way to double
    // precision width instead of trusting the price tolerance alone.
    let (s, k, t, r, q, sigma) = (100.0, 100.0, 25.0, 0.10, 0.0, 3.0);
    let price = bs_price(s, k, t, r, sigma, q, OptionType::Put).unwrap();
    let iv = implied_vol(price, s, k, t, r, q, OptionType::Put).unwrap();
    assert!((iv - sigma).abs() < 2e-4, "recovered {iv}, expected ~{sigma}");
}
