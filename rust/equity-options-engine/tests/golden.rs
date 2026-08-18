//! Cross-language golden validation: every case in `src/golden.rs`
//! (generated from the Python project's committed golden_vectors.json)
//! must match the Python reference to 1e-9 on price and all five Greeks.
//!
//! Measured agreement is at the few-ulp level (< 1e-13); the 1e-9 gate
//! matches the C++ engine's.

mod common;

use eq_options_engine::{bs_greeks, bs_price, golden};

const TOL: f64 = 1e-9;

#[test]
fn golden_has_32_cases() {
    assert_eq!(golden::CASES.len(), 32);
}

#[test]
fn golden_prices_match_python_reference() {
    for (i, c) in golden::CASES.iter().enumerate() {
        let price = bs_price(c.s, c.k, c.t, c.r, c.sigma, c.q, c.option_type)
            .unwrap_or_else(|e| panic!("case {i}: {e}"));
        assert!(
            (price - c.price).abs() <= TOL,
            "case {i} ({:?} S={} K={} T={}): price {price} vs golden {} (diff {:.3e})",
            c.option_type,
            c.s,
            c.k,
            c.t,
            c.price,
            (price - c.price).abs()
        );
    }
}

#[test]
fn golden_greeks_match_python_reference() {
    for (i, c) in golden::CASES.iter().enumerate() {
        let g = bs_greeks(c.s, c.k, c.t, c.r, c.sigma, c.q, c.option_type)
            .unwrap_or_else(|e| panic!("case {i}: {e}"));
        for (name, got, want) in [
            ("price", g.price, c.price),
            ("delta", g.delta, c.delta),
            ("gamma", g.gamma, c.gamma),
            ("vega", g.vega, c.vega),
            ("theta", g.theta, c.theta),
            ("rho", g.rho, c.rho),
        ] {
            assert!(
                (got - want).abs() <= TOL,
                "case {i} ({:?} S={} K={} T={}): {name} {got} vs golden {want} (diff {:.3e})",
                c.option_type,
                c.s,
                c.k,
                c.t,
                (got - want).abs()
            );
        }
    }
}

#[test]
fn golden_worst_case_deviation_is_sub_picolevel() {
    // Track the actual agreement level so a regression that stays inside
    // 1e-9 but degrades accuracy by orders of magnitude is still caught.
    let mut worst = 0.0_f64;
    for c in golden::CASES.iter() {
        let g = bs_greeks(c.s, c.k, c.t, c.r, c.sigma, c.q, c.option_type).unwrap();
        for (got, want) in [
            (g.price, c.price),
            (g.delta, c.delta),
            (g.gamma, c.gamma),
            (g.vega, c.vega),
            (g.theta, c.theta),
            (g.rho, c.rho),
        ] {
            worst = worst.max((got - want).abs());
        }
    }
    println!("worst golden deviation: {worst:.3e}");
    assert!(
        worst < 1e-12,
        "worst golden deviation {worst:.3e} exceeds 1e-12"
    );
}
