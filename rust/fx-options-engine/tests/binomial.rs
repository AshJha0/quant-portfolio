//! CRR tree: convergence to GK, American exercise premia, degeneracies.

mod common;
use common::assert_close;

use fx_options_engine::{binomial_price, gk_price, Exercise, OptionType};

const MKT: (f64, f64, f64, f64, f64, f64) = (1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925);

#[test]
fn european_tree_converges_to_gk() {
    let (s, k, t, rd, rf, sig) = MKT;
    for ty in [OptionType::Call, OptionType::Put] {
        let exact = gk_price(s, k, t, rd, rf, sig, ty).unwrap();
        let tree = binomial_price(s, k, t, rd, rf, sig, ty, 500, Exercise::European).unwrap();
        assert_close(tree, exact, 5e-5, &format!("500-step tree {ty:?}"));

        // Error shrinks as the tree refines (O(1/n) up to oscillation:
        // compare 50 vs 800 steps for a robust ordering).
        let e_coarse =
            (binomial_price(s, k, t, rd, rf, sig, ty, 50, Exercise::European).unwrap() - exact)
                .abs();
        let e_fine =
            (binomial_price(s, k, t, rd, rf, sig, ty, 800, Exercise::European).unwrap() - exact)
                .abs();
        assert!(
            e_fine < e_coarse,
            "{ty:?}: 800-step error {e_fine} !< 50-step error {e_coarse}"
        );
    }
}

#[test]
fn american_at_least_european_everywhere() {
    let (s, _, t, _, _, sig) = MKT;
    for &k in &[0.95, 1.10, 1.25] {
        for &(rd, rf) in &[(0.0425, 0.0290), (0.01, 0.08), (0.08, 0.01)] {
            for ty in [OptionType::Call, OptionType::Put] {
                let eur = binomial_price(s, k, t, rd, rf, sig, ty, 300, Exercise::European).unwrap();
                let amer = binomial_price(s, k, t, rd, rf, sig, ty, 300, Exercise::American).unwrap();
                assert!(
                    amer >= eur - 1e-12,
                    "American {amer} < European {eur} (K={k} rd={rd} rf={rf} {ty:?})"
                );
            }
        }
    }
}

#[test]
fn american_call_early_exercise_premium_when_rf_exceeds_rd() {
    // High-yielding foreign currency: the foreign carry lost by holding
    // the option (not the currency) makes early exercise valuable.
    let (s, k, t, sig) = (1.10, 1.00, 1.0, 0.10);
    let (rd, rf) = (0.01, 0.08);
    let eur = binomial_price(s, k, t, rd, rf, sig, OptionType::Call, 500, Exercise::European).unwrap();
    let amer = binomial_price(s, k, t, rd, rf, sig, OptionType::Call, 500, Exercise::American).unwrap();
    assert!(
        amer - eur > 1e-4,
        "ITM American call premium {:.2e} not positive with r_f > r_d",
        amer - eur
    );
}

#[test]
fn american_put_early_exercise_premium_when_rd_exceeds_rf() {
    let (s, k, t, sig) = (1.10, 1.20, 1.0, 0.10);
    let (rd, rf) = (0.08, 0.01);
    let eur = binomial_price(s, k, t, rd, rf, sig, OptionType::Put, 500, Exercise::European).unwrap();
    let amer = binomial_price(s, k, t, rd, rf, sig, OptionType::Put, 500, Exercise::American).unwrap();
    assert!(amer - eur > 1e-4, "ITM American put premium not positive with r_d > r_f");
}

#[test]
fn t_zero_returns_intrinsic() {
    assert_close(
        binomial_price(1.10, 1.00, 0.0, 0.04, 0.02, 0.10, OptionType::Call, 100, Exercise::American)
            .unwrap(),
        0.10,
        1e-15,
        "T=0 intrinsic",
    );
}

#[test]
fn sigma_zero_european_matches_gk_limit() {
    let (s, k, t, rd, rf) = (1.10, 1.05, 0.75, 0.0425, 0.0290);
    let tree =
        binomial_price(s, k, t, rd, rf, 0.0, OptionType::Call, 200, Exercise::European).unwrap();
    let gk = gk_price(s, k, t, rd, rf, 0.0, OptionType::Call).unwrap();
    assert_close(tree, gk, 1e-15, "sigma=0 European tree");
}

#[test]
fn sigma_zero_american_optimises_deterministic_exercise() {
    // r_f > r_d drifts the spot down: an ITM call is best exercised now.
    let (s, k, t, rd, rf) = (1.10, 1.00, 1.0, 0.01, 0.08);
    let amer =
        binomial_price(s, k, t, rd, rf, 0.0, OptionType::Call, 200, Exercise::American).unwrap();
    assert_close(amer, s - k, 1e-12, "sigma=0 American immediate exercise");
    let eur = binomial_price(s, k, t, rd, rf, 0.0, OptionType::Call, 200, Exercise::European).unwrap();
    assert!(amer > eur);
}

#[test]
fn invalid_steps_and_coarse_tree_err() {
    let (s, k, t, rd, rf, sig) = MKT;
    assert!(binomial_price(s, k, t, rd, rf, sig, OptionType::Call, 0, Exercise::European).is_err());
    // dt too large for |r_d - r_f| vs sigma: p outside [0, 1].
    assert!(
        binomial_price(s, k, 1.0, 0.50, 0.0, 0.01, OptionType::Call, 1, Exercise::European)
            .is_err()
    );
}
