//! Monte Carlo: statistical agreement, variance reduction, determinism.

mod common;
use common::assert_close;

use fx_options_engine::{gk_price, mc_price, OptionType};

const MKT: (f64, f64, f64, f64, f64, f64) = (1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925);

#[test]
fn price_within_three_standard_errors_of_gk() {
    let (s, k, t, rd, rf, sig) = MKT;
    for ty in [OptionType::Call, OptionType::Put] {
        let exact = gk_price(s, k, t, rd, rf, sig, ty).unwrap();
        let mc = mc_price(s, k, t, rd, rf, sig, ty, 200_000, 0, true, true).unwrap();
        assert!(
            (mc.price - exact).abs() < 3.0 * mc.std_error,
            "{ty:?}: |{} - {exact}| > 3 SE ({:.2e})",
            mc.price,
            mc.std_error
        );
    }
}

#[test]
fn negative_rates_within_three_standard_errors() {
    let (s, k, t, rd, rf, sig) = (1.08, 1.10, 1.0, -0.0075, -0.0050, 0.07);
    let exact = gk_price(s, k, t, rd, rf, sig, OptionType::Call).unwrap();
    let mc = mc_price(s, k, t, rd, rf, sig, OptionType::Call, 200_000, 1, true, true).unwrap();
    assert!((mc.price - exact).abs() < 3.0 * mc.std_error);
}

#[test]
fn variance_reduction_shrinks_the_standard_error() {
    let (s, k, t, rd, rf, sig) = MKT;
    let n = 100_000;
    let plain = mc_price(s, k, t, rd, rf, sig, OptionType::Call, n, 3, false, false).unwrap();
    let full = mc_price(s, k, t, rd, rf, sig, OptionType::Call, n, 3, true, true).unwrap();
    assert!(
        full.std_error < 0.5 * plain.std_error,
        "antithetic+CV SE {:.3e} not well below plain SE {:.3e}",
        full.std_error,
        plain.std_error
    );
    assert_eq!(plain.method, "plain");
    assert_eq!(full.method, "antithetic+control_variate");
}

#[test]
fn same_seed_is_bitwise_reproducible() {
    let (s, k, t, rd, rf, sig) = MKT;
    let a = mc_price(s, k, t, rd, rf, sig, OptionType::Call, 50_000, 42, true, true).unwrap();
    let b = mc_price(s, k, t, rd, rf, sig, OptionType::Call, 50_000, 42, true, true).unwrap();
    // Bit-identical, not merely close.
    assert_eq!(a.price.to_bits(), b.price.to_bits());
    assert_eq!(a.std_error.to_bits(), b.std_error.to_bits());
    assert_eq!(a, b);
}

#[test]
fn different_seeds_differ() {
    let (s, k, t, rd, rf, sig) = MKT;
    let a = mc_price(s, k, t, rd, rf, sig, OptionType::Call, 50_000, 1, true, true).unwrap();
    let b = mc_price(s, k, t, rd, rf, sig, OptionType::Call, 50_000, 2, true, true).unwrap();
    assert_ne!(a.price.to_bits(), b.price.to_bits());
}

#[test]
fn confidence_interval_brackets_the_price() {
    let (s, k, t, rd, rf, sig) = MKT;
    let mc = mc_price(s, k, t, rd, rf, sig, OptionType::Put, 20_000, 9, true, true).unwrap();
    assert!(mc.ci_low <= mc.price && mc.price <= mc.ci_high);
    assert_close(
        mc.ci_high - mc.ci_low,
        2.0 * 1.96 * mc.std_error,
        1e-15,
        "CI width",
    );
    assert_eq!(mc.n_paths, 20_000);
}

#[test]
fn antithetic_rounds_odd_path_counts_up_to_even() {
    let (s, k, t, rd, rf, sig) = MKT;
    let mc = mc_price(s, k, t, rd, rf, sig, OptionType::Call, 9_999, 0, true, false).unwrap();
    assert_eq!(mc.n_paths, 10_000);
    assert_eq!(mc.method, "antithetic+plain");
}

#[test]
fn invalid_inputs_err() {
    let (s, k, _, rd, rf, sig) = MKT;
    assert!(mc_price(s, k, 0.0, rd, rf, sig, OptionType::Call, 1000, 0, true, true).is_err());
    assert!(mc_price(s, k, 0.5, rd, rf, sig, OptionType::Call, 1, 0, true, true).is_err());
    assert!(mc_price(-s, k, 0.5, rd, rf, sig, OptionType::Call, 1000, 0, true, true).is_err());
}
