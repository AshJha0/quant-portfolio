//! Monte Carlo: statistical validity, variance reduction, reproducibility.

mod common;

use eq_options_engine::rng::Xoshiro256PlusPlus;
use eq_options_engine::{bs_price, mc_price, OptionType};

#[test]
fn mc_price_within_three_standard_errors_of_bs() {
    for &(s, k, t, r, q, sigma, ot) in &[
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.2, OptionType::Call),
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.2, OptionType::Put),
        (100.0, 120.0, 0.5, 0.03, 0.01, 0.3, OptionType::Call),
        (100.0, 80.0, 2.0, 0.02, 0.03, 0.15, OptionType::Put),
    ] {
        let bs = bs_price(s, k, t, r, sigma, q, ot).unwrap();
        let mc = mc_price(s, k, t, r, sigma, q, ot, 200_000, true, true, 42).unwrap();
        assert!(
            (mc.price - bs).abs() <= 3.0 * mc.std_error.max(1e-12),
            "MC {} vs BS {} exceeds 3 SE ({:.3e}) at (S={s}, K={k}, {ot:?})",
            mc.price,
            bs,
            mc.std_error
        );
        assert!(mc.ci_low <= mc.price && mc.price <= mc.ci_high);
        assert_eq!(mc.n_paths, 200_000);
    }
}

#[test]
fn variance_reduction_reduces_standard_error() {
    let (s, k, t, r, q, sigma) = (100.0, 100.0, 1.0, 0.05, 0.0, 0.2);
    let plain = mc_price(s, k, t, r, sigma, q, OptionType::Call, 100_000, false, false, 7)
        .unwrap();
    let anti = mc_price(s, k, t, r, sigma, q, OptionType::Call, 100_000, true, false, 7)
        .unwrap();
    let cv = mc_price(s, k, t, r, sigma, q, OptionType::Call, 100_000, false, true, 7)
        .unwrap();
    let both = mc_price(s, k, t, r, sigma, q, OptionType::Call, 100_000, true, true, 7)
        .unwrap();
    assert!(
        anti.std_error < plain.std_error,
        "antithetic did not reduce SE: {} vs {}",
        anti.std_error,
        plain.std_error
    );
    assert!(
        cv.std_error < plain.std_error,
        "control variate did not reduce SE: {} vs {}",
        cv.std_error,
        plain.std_error
    );
    // For the ATM call the measured combined reduction is ~0.54x; gate at
    // 0.75x so the test asserts a material improvement without being
    // seed-sensitive.
    assert!(
        both.std_error < 0.75 * plain.std_error,
        "combined VR should materially reduce the SE: {} vs {}",
        both.std_error,
        plain.std_error
    );
}

#[test]
fn same_seed_is_bitwise_identical() {
    let a = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, 50_000, true, true, 123)
        .unwrap();
    let b = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, 50_000, true, true, 123)
        .unwrap();
    assert_eq!(a.price.to_bits(), b.price.to_bits());
    assert_eq!(a.std_error.to_bits(), b.std_error.to_bits());
    assert_eq!(a.ci_low.to_bits(), b.ci_low.to_bits());
    assert_eq!(a.ci_high.to_bits(), b.ci_high.to_bits());
}

#[test]
fn different_seeds_differ() {
    let a = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, 50_000, true, true, 1)
        .unwrap();
    let b = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, 50_000, true, true, 2)
        .unwrap();
    assert_ne!(a.price.to_bits(), b.price.to_bits());
}

#[test]
fn deterministic_limits_have_zero_std_error() {
    // T = 0 and sigma = 0 return the exact BS value with SE = 0.
    let at_expiry =
        mc_price(105.0, 100.0, 0.0, 0.05, 0.2, 0.0, OptionType::Call, 1000, true, true, 42)
            .unwrap();
    assert_eq!(at_expiry.price, 5.0);
    assert_eq!(at_expiry.std_error, 0.0);
    let no_vol =
        mc_price(100.0, 95.0, 1.0, 0.05, 0.0, 0.02, OptionType::Call, 1000, true, true, 42)
            .unwrap();
    let bs = bs_price(100.0, 95.0, 1.0, 0.05, 0.0, 0.02, OptionType::Call).unwrap();
    assert_eq!(no_vol.price, bs);
    assert_eq!(no_vol.ci_low, no_vol.ci_high);
}

#[test]
fn invalid_inputs_return_err() {
    assert!(
        mc_price(-1.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, 1000, true, true, 42)
            .is_err()
    );
    assert!(
        mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, 1, true, true, 42).is_err()
    );
}

#[test]
fn rng_stream_is_reproducible_and_well_behaved() {
    // Bit-reproducibility of the raw stream.
    let mut a = Xoshiro256PlusPlus::new(2024);
    let mut b = Xoshiro256PlusPlus::new(2024);
    for _ in 0..1000 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
    // Uniforms live strictly inside (0, 1).
    let mut rng = Xoshiro256PlusPlus::new(5);
    for _ in 0..100_000 {
        let u = rng.next_uniform();
        assert!(u > 0.0 && u < 1.0);
    }
    // Normal moments: mean ~ 0, var ~ 1 (3-sigma bands for n = 1e6).
    let n = 1_000_000;
    let mut sum = 0.0;
    let mut sumsq = 0.0;
    for _ in 0..n {
        let z = rng.standard_normal();
        sum += z;
        sumsq += z * z;
    }
    let mean = sum / n as f64;
    let var = sumsq / n as f64 - mean * mean;
    assert!(mean.abs() < 3.0 / (n as f64).sqrt(), "normal mean {mean}");
    assert!(
        (var - 1.0).abs() < 3.0 * (2.0 / n as f64).sqrt(),
        "normal variance {var}"
    );
}
