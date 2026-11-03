//! Antithetic plus control variate must beat the control variate alone.
//!
//! Beta is fitted on the antithetic pair averages (this crate already did
//! so; the test guards the shared estimator across seeds).

use fx_options_engine::{OptionType, gk_price, mc_price};

#[test]
fn antithetic_plus_cv_beats_cv_alone_across_seeds() {
    let (s, k, t, rd, rf, sig) = (1.10, 1.10, 1.0, 0.04, 0.02, 0.10);
    for seed in 1..=7u64 {
        for ot in [OptionType::Call, OptionType::Put] {
            let both = mc_price(s, k, t, rd, rf, sig, ot, 100_000, seed, true, true).unwrap();
            let cv = mc_price(s, k, t, rd, rf, sig, ot, 100_000, seed, false, true).unwrap();
            assert!(
                both.std_error <= cv.std_error,
                "seed {seed}: {} vs {}",
                both.std_error,
                cv.std_error
            );
            let exact = gk_price(s, k, t, rd, rf, sig, ot).unwrap();
            assert!((both.price - exact).abs() < 4.0 * both.std_error);
        }
    }
}
