//! Antithetic plus control variate must beat the control variate alone.
//!
//! Beta is fitted on the antithetic pair averages. Fitting it on raw draws
//! and pairing afterwards gave SE 0.0247 at 100k paths for this contract,
//! worse than the control variate alone (0.0177).

use eq_options_engine::{bs_price, mc_price, OptionType};

const OLD_COMBINED_SE: f64 = 0.0246;

#[test]
fn antithetic_plus_cv_beats_cv_alone_across_seeds() {
    for seed in 1..=7u64 {
        for ot in [OptionType::Call, OptionType::Put] {
            let both = mc_price(
                100.0, 100.0, 1.0, 0.05, 0.2, 0.0, ot, 100_000, true, true, seed,
            )
            .unwrap();
            let cv = mc_price(
                100.0, 100.0, 1.0, 0.05, 0.2, 0.0, ot, 100_000, false, true, seed,
            )
            .unwrap();
            assert!(
                both.std_error <= cv.std_error,
                "seed {seed}: {} vs {}",
                both.std_error,
                cv.std_error
            );
            let exact = bs_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, ot).unwrap();
            assert!((both.price - exact).abs() < 4.0 * both.std_error);
            if ot == OptionType::Call {
                assert!(
                    both.std_error < 0.6 * OLD_COMBINED_SE,
                    "seed {seed}: {}",
                    both.std_error
                );
            }
        }
    }
}
