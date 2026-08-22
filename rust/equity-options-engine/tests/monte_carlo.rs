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
fn std_error_scales_as_inverse_sqrt_n_fitted() {
    // Plain (no variance reduction) MC has statistical error O(1/sqrt(n))
    // by the CLT. Rather than trust the estimator's own reported
    // std_error formula, measure the *empirical* spread of independent
    // replications of the price at each path count and fit the exponent
    // of empirical_std vs n by log-log regression -- an actual
    // measurement of the realized rate, not a single eyeballed ratio.
    let (s, k, t, r, q, sigma) = (100.0, 100.0, 1.0, 0.05, 0.0, 0.2);
    let path_counts = [2_000usize, 4_000, 8_000, 16_000, 32_000, 64_000];
    const REPS: u64 = 40;
    let mut log_n = Vec::with_capacity(path_counts.len());
    let mut log_std = Vec::with_capacity(path_counts.len());
    for &n in &path_counts {
        let reps: Vec<f64> = (0..REPS)
            .map(|rep| {
                let seed = 1_000_003u64.wrapping_mul(n as u64).wrapping_add(rep);
                mc_price(s, k, t, r, sigma, q, OptionType::Call, n, false, false, seed)
                    .unwrap()
                    .price
            })
            .collect();
        let mean = reps.iter().sum::<f64>() / reps.len() as f64;
        let var =
            reps.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (reps.len() as f64 - 1.0);
        log_n.push((n as f64).ln());
        log_std.push(0.5 * var.ln());
    }
    let mean_x = log_n.iter().sum::<f64>() / log_n.len() as f64;
    let mean_y = log_std.iter().sum::<f64>() / log_std.len() as f64;
    let num: f64 = log_n
        .iter()
        .zip(&log_std)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = log_n.iter().map(|x| (x - mean_x).powi(2)).sum();
    let slope = num / den;
    assert!(
        (-0.65..-0.35).contains(&slope),
        "fitted MC exponent {slope:.3}; CLT predicts -0.5 (std ~ C/sqrt(n))"
    );
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
    // Non-finite inputs (including r/q) are rejected before any simulation.
    assert!(
        mc_price(100.0, 100.0, 1.0, f64::NAN, 0.2, 0.0, OptionType::Call, 1000, true, true, 42)
            .is_err()
    );
    assert!(
        mc_price(100.0, 100.0, f64::INFINITY, 0.05, 0.2, 0.0, OptionType::Call, 1000, true,
                 true, 42)
            .is_err()
    );
    // Path counts that would ask the allocator for tens of GB are
    // rejected up front rather than aborting the process.
    use eq_options_engine::monte_carlo::MAX_PATHS;
    assert!(
        mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, MAX_PATHS + 1, true,
                 true, 42)
            .is_err()
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

#[test]
fn minimum_path_counts_produce_a_finite_standard_error() {
    // n_paths = 2 with antithetic sampling leaves exactly ONE independent
    // sample (the single antithetic pair average). The sample-variance
    // denominator (n - 1) is then 0: a naive estimator computes 0/0 and
    // reports a NaN standard error, which silently poisons every
    // downstream confidence check. The estimator must instead report
    // SE = 0 and a degenerate (point) confidence interval.
    for &n in &[2usize, 3, 4] {
        let r = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, n, true, true, 11)
            .unwrap();
        assert!(r.price.is_finite(), "n={n}: price {} not finite", r.price);
        assert!(
            r.std_error.is_finite() && r.std_error >= 0.0,
            "n={n}: std_error {} must be a finite non-negative number",
            r.std_error
        );
        assert!(r.ci_low.is_finite() && r.ci_high.is_finite(), "n={n}: CI not finite");
        assert!(r.ci_low <= r.ci_high, "n={n}: inverted CI");
        assert!(r.contains(r.price), "n={n}: CI must contain its own point estimate");
    }
    // The single-pair case specifically: one sample => SE is exactly 0,
    // not NaN, and the interval degenerates to the point estimate.
    let one_pair =
        mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, 2, true, false, 11).unwrap();
    assert_eq!(one_pair.std_error, 0.0);
    assert_eq!(one_pair.ci_low, one_pair.price);
    assert_eq!(one_pair.ci_high, one_pair.price);
    assert_eq!(one_pair.n_paths, 2);
    // Without antithetic sampling, 2 paths give 2 samples and a genuine
    // (strictly positive, finite) standard error.
    let two =
        mc_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call, 2, false, false, 11).unwrap();
    assert!(two.std_error.is_finite());
    assert!(two.ci_low < two.ci_high || two.std_error == 0.0);
}

#[test]
fn extreme_moneyness_and_long_expiry_estimates_are_sane() {
    // Deep OTM (100x strike) with a long expiry: the estimator must stay
    // non-negative, finite, and bracket the analytic value within 3 SE.
    for &(s, k, t) in &[(100.0, 10_000.0, 1.0), (10_000.0, 100.0, 1.0), (100.0, 100.0, 30.0)] {
        let mc = mc_price(s, k, t, 0.03, 0.2, 0.01, OptionType::Call, 200_000, true, true, 99)
            .unwrap();
        let bs = bs_price(s, k, t, 0.03, 0.2, 0.01, OptionType::Call).unwrap();
        assert!(mc.price.is_finite() && mc.std_error.is_finite(), "S={s}, K={k}, T={t}");
        assert!(mc.price >= -1e-12, "negative MC price {} at S={s}, K={k}", mc.price);
        // Tolerance is 3 SE plus a SCALE-RELATIVE floor: deep ITM the
        // control variate drives the SE to ~1e-11, well below the
        // floating-point accumulation noise of a 200k-term sum on a
        // ~1e4-sized price, so an absolute-only tolerance would reject a
        // perfectly healthy estimate on a large-notional trade.
        assert!(
            (mc.price - bs).abs() <= 3.0 * mc.std_error + 1e-9 * bs.abs().max(1.0),
            "S={s}, K={k}, T={t}: MC {} vs BS {bs} (SE {})",
            mc.price,
            mc.std_error
        );
    }
}
