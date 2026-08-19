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

#[test]
fn minimum_path_counts_never_produce_a_nan_standard_error() {
    // `n_paths = 2` with antithetic pairing collapses to exactly ONE
    // independent sample, so the (n - 1) sample-variance denominator is
    // zero. A naive estimator computes 0/0, reports `std_error = NaN`,
    // and every downstream `|mc - analytic| < 3 SE` check then silently
    // passes (`x < NaN` is false, `x > NaN` is false — neither branch
    // fires). The estimator must report SE = 0 and a degenerate interval
    // instead.
    let (s, k, t, rd, rf, sig) = MKT;
    for &(n, anti, cv) in &[
        (2u64, true, false),
        (2, true, true),
        (2, false, false),
        (2, false, true),
        (3, true, true),
        (4, true, true),
        (5, false, true),
    ] {
        let mc = mc_price(s, k, t, rd, rf, sig, OptionType::Call, n, 3, anti, cv).unwrap();
        assert!(
            mc.price.is_finite(),
            "n={n} anti={anti} cv={cv}: price {} not finite",
            mc.price
        );
        assert!(
            mc.std_error.is_finite() && mc.std_error >= 0.0,
            "n={n} anti={anti} cv={cv}: std_error {} must be finite and >= 0",
            mc.std_error
        );
        assert!(
            mc.ci_low.is_finite() && mc.ci_high.is_finite() && mc.ci_low <= mc.ci_high,
            "n={n} anti={anti} cv={cv}: CI [{}, {}]",
            mc.ci_low,
            mc.ci_high
        );
        // Without a control variate the estimator is an average of
        // non-negative payoffs and cannot go negative. WITH one it can:
        // the CV adjustment `X - beta (C - E[C])` is unbiased but not
        // pathwise non-negative, and at n = 2 the fitted beta is pure
        // noise. That is a property of the estimator, not a bug — but it
        // is exactly why a desk should not run a CV estimator at
        // single-digit path counts.
        if !cv {
            assert!(mc.price >= -1e-15, "n={n} plain: negative price {}", mc.price);
        }
    }
    // The single-pair case specifically: SE is exactly zero, and the
    // interval degenerates to the point estimate.
    let one = mc_price(s, k, t, rd, rf, sig, OptionType::Call, 2, 3, true, false).unwrap();
    assert_eq!(one.std_error, 0.0);
    assert_eq!(one.ci_low, one.price);
    assert_eq!(one.ci_high, one.price);
    assert_eq!(one.n_paths, 2);
}

#[test]
fn extreme_moneyness_and_long_tenor_estimates_stay_inside_three_se() {
    // Deep OTM (100x strike), deep ITM (1/100x) and a 30-year tenor: the
    // estimator must stay non-negative and statistically consistent with
    // the analytic GK value.
    for &(s, k, t) in &[(1.10, 110.0, 0.5), (110.0, 1.10, 0.5), (1.10, 1.12, 30.0)] {
        let exact = gk_price(s, k, t, 0.03, 0.01, 0.12, OptionType::Call).unwrap();
        let mc = mc_price(s, k, t, 0.03, 0.01, 0.12, OptionType::Call, 200_000, 21, true, true)
            .unwrap();
        assert!(
            mc.price >= -1e-12 * exact.abs().max(1.0),
            "S={s} K={k} T={t}: materially negative price {}",
            mc.price
        );
        assert!(mc.std_error.is_finite(), "S={s} K={k} T={t}: SE not finite");
        // Scale-relative floor on top of 3 SE: with a control variate the
        // deep-ITM SE falls below the f64 summation noise of a 2e5-term
        // sum on a ~1e2-sized price, so an absolute-only gate would
        // reject a perfectly healthy large-notional estimate.
        let tol = 3.0 * mc.std_error + 1e-10 * exact.abs().max(1.0);
        assert!(
            (mc.price - exact).abs() <= tol,
            "S={s} K={k} T={t}: MC {} vs GK {exact} (SE {}, tol {tol:.3e})",
            mc.price,
            mc.std_error
        );
    }
}
