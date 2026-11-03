//! Regression tests for the shared conventions S1-S11 and the H-1 panic
//! paths (ws-var round 1). Each failed on the code before this change
//! (wrong number, panic, or a missing entry point) and passes now.
#![allow(clippy::needless_range_loop)]

use fx_var_engine::McDist;
use fx_var_engine::backtest::*;
use fx_var_engine::book::{Book, Market, Position, SpotPosition};
use fx_var_engine::expected_shortfall::*;
use fx_var_engine::historical::{HistoricalOptions, historical_var};
use fx_var_engine::matrix::Matrix;
use fx_var_engine::monte_carlo::*;
use fx_var_engine::parametric::*;
use fx_var_engine::psd::*;
use fx_var_engine::returns::{FactorCov, ReturnsMatrix, ewma_volatility};
use fx_var_engine::rng::Rng;
use fx_var_engine::stats::*;

// ---------------------------------------------------------------- S1 alpha

#[test]
fn confidence_levels_are_rejected_not_reinterpreted() {
    let pnl: Vec<f64> = (0..100).map(|i| (i as f64 * 0.7).sin()).collect();
    for bad in [0.99, 0.975, 0.95, 0.5] {
        let e = empirical_var(&pnl, bad, None).unwrap_err();
        assert!(
            e.to_string().contains("alpha is the tail probability"),
            "{bad}"
        );
        assert!(normal_var(1.0, bad, 0.0).is_err());
        assert!(
            kupiec_pof(&KupiecInput {
                n_obs: 250,
                n_exceptions: 3,
                alpha: bad
            })
            .is_err()
        );
    }
    // Tail convention: alpha = 0.01 is 99 % VaR.
    assert!((normal_var(1.0, 0.01, 0.0).unwrap() - 2.326_347_874_040_840_8).abs() < 1e-9);
    assert!((normal_es(1.0, 0.01, 0.0).unwrap() - 2.665_214_220_345_808).abs() < 1e-9);
    assert_eq!(HistoricalOptions::default().alpha, 0.01);
    assert_eq!(MonteCarloOptions::default().alpha, 0.01);
    assert_eq!(ParametricOptions::default().alpha, 0.01);
}

#[test]
#[allow(deprecated)]
fn deprecated_confidence_shims_convert() {
    let new = kupiec_pof(&KupiecInput {
        n_obs: 250,
        n_exceptions: 8,
        alpha: 0.01,
    })
    .unwrap();
    let old = kupiec_pof_confidence(8, 250, 0.99).unwrap();
    assert!((new.lr - old.lr).abs() < 1e-12);
    let t = basel_traffic_light_confidence(5, 250, 0.99).unwrap();
    assert_eq!(t.zone, Zone::Yellow);
}

// ------------------------------------------------------------- S2 estimator

#[test]
fn order_statistic_and_fractional_es_by_hand() {
    let pnl: Vec<f64> = (1..=10).map(|i| -(i as f64)).collect();
    let (v, es) = empirical_var_es(&pnl, 0.25, None).unwrap();
    assert_eq!(v, 8.0);
    assert!((es - 9.2).abs() < 1e-14);
}

// ------------------------------------------------------------------- S4 Basel

#[test]
fn basel_plus_factor_uses_the_equivalent_250_day_count() {
    // Old code read the add-on from the raw count (15 -> 0.85).
    for (n, x) in [(1000_i64, 15_i64), (500, 9), (500, 12)] {
        let t = basel_traffic_light(&BaselInput {
            n_obs: n,
            n_exceptions: x,
            alpha: 0.01,
        })
        .unwrap();
        let f = binom_cdf(x, n, 0.01).unwrap();
        assert!((t.cumulative_prob - f).abs() < 1e-12);
        if t.zone == Zone::Yellow {
            let mut x_eq = 9;
            for xp in 5..=9 {
                if binom_cdf(xp, 250, 0.01).unwrap() >= f - 1e-12 {
                    x_eq = xp;
                    break;
                }
            }
            let table = [0.40, 0.50, 0.65, 0.75, 0.85][(x_eq - 5) as usize];
            assert_eq!(t.plus_factor, table, "n={n} x={x}");
        }
    }
    let t = basel_traffic_light(&BaselInput {
        n_obs: 1000,
        n_exceptions: 15,
        alpha: 0.01,
    })
    .unwrap();
    assert_eq!(t.zone, Zone::Yellow);
    assert!(t.plus_factor < 0.85);
    assert!(
        basel_traffic_light(&BaselInput {
            n_obs: 250,
            n_exceptions: 251,
            alpha: 0.01
        })
        .is_err()
    );
}

// --------------------------------------------------------------------- S5 Z2

#[test]
fn acerbi_szekely_z2_value() {
    let z2 = acerbi_szekely_z2(&[-2.0, 0.0, -2.0, 1.0], &[1.0], &[2.0], 0.25).unwrap();
    assert!((z2 + 1.0).abs() < 1e-15);
}

// ---------------------------------------------------------------------- S6 CF

#[test]
fn cornish_fisher_counterexample_is_valid_at_99() {
    // The old fixed |z| <= 4 interval rejected (0.122, -0.427) at 99 %.
    assert!(cornish_fisher_valid(0.122, -0.427, 0.01).unwrap());
    assert!(cornish_fisher_var(1.0, 0.122, -0.427, 0.01, 0.0, 1.0, true).is_ok());
    assert!(cornish_fisher_valid(f64::NAN, 0.0, 0.01).is_err());
}

// ---------------------------------------------------------------- S7 Cholesky

#[test]
fn indefinite_covariance_errors_and_repairs() {
    let a = Matrix::from_rows(&[
        vec![1.0, 0.9, 0.9],
        vec![0.9, 1.0, -0.9],
        vec![0.9, -0.9, 1.0],
    ])
    .unwrap();
    let e = cholesky_repair(&a, PsdRepair::None).unwrap_err();
    assert!(e.to_string().contains("most negative eigenvalue"));
    let h = cholesky_repair(&a, PsdRepair::Higham)
        .unwrap()
        .repaired
        .unwrap();
    for i in 0..3 {
        assert!((h.get(i, i) - 1.0).abs() < 1e-12);
    }
    let c = cholesky_repair(&a, PsdRepair::Clip)
        .unwrap()
        .repaired
        .unwrap();
    for i in 0..3 {
        assert!((c.get(i, i) - 1.0).abs() < 1e-15);
    }
    let cov = FactorCov {
        factors: vec!["FX:EUR".into(), "FX:JPY".into(), "FX:GBP".into()],
        cov: a,
    };
    assert!(
        simulate_factor_returns(&cov, 100, McDist::Normal, 6.0, &JumpSpec::default(), 1, 1.0)
            .is_err()
    );
}

// -------------------------------------------------------------------- S8 EWMA

#[test]
fn ewma_seed_is_first_30_observations() {
    let n = 60;
    let mut data = Matrix::zeros(n, 1);
    for t in 0..n {
        data.set(
            t,
            0,
            0.01 * ((t as f64) * 0.9).sin() * if t >= 30 { 10.0 } else { 1.0 },
        );
    }
    let r = ReturnsMatrix::new(vec!["FX:EUR".to_string()], data.clone()).unwrap();
    let ev = ewma_volatility(&r, 0.94).unwrap();
    let seed: f64 = (0..30).map(|t| data.get(t, 0).powi(2)).sum::<f64>() / 30.0;
    // Old seed was the full-sample variance, which saw the 10x regime.
    assert!((ev.sigma.get(0, 0) - seed.sqrt()).abs() < 1e-17);
}

// ---------------------------------------------------------------------- S10 MC

#[test]
fn monte_carlo_reports_bootstrap_se_and_jitter() {
    let market = Market::new([("EUR", 1.10)], [("USD", 0.05), ("EUR", 0.03)]).unwrap();
    let book = Book::new(
        vec![Position::Spot(SpotPosition::new("EURUSD", 10e6, None))],
        "USD",
    )
    .unwrap();
    let cov = FactorCov {
        factors: vec!["FX:EUR".to_string()],
        cov: Matrix::from_rows(&[vec![1e-4]]).unwrap(),
    };
    let opts = MonteCarloOptions {
        n_scenarios: 20_000,
        seed: 3,
        ..Default::default()
    };
    let r = monte_carlo_var(&book, &market, &cov, &opts).unwrap();
    let boot =
        var_standard_error_bootstrap(&r.pnl, 0.01, opts.n_boot, opts.seed ^ 0x9E37_79B9_7F4A_7C15)
            .unwrap();
    assert_eq!(r.se_var, boot);
    assert_eq!(r.jitter, 0.0);
    let bad = MonteCarloOptions { n_boot: 1, ..opts };
    assert!(monte_carlo_var(&book, &market, &cov, &bad).is_err());
}

// ------------------------------------------------------------- S11 empty book

#[test]
fn empty_book_error_message() {
    let market = Market::new([("EUR", 1.10)], [("USD", 0.05), ("EUR", 0.03)]).unwrap();
    let rets = ReturnsMatrix::new(vec!["FX:EUR".to_string()], Matrix::zeros(100, 1)).unwrap();
    let e = historical_var(
        &Book::empty(),
        &market,
        &rets,
        &HistoricalOptions::default(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("book has no positions"));
}

// --------------------------------------------------------- H-1 panic paths

#[test]
fn former_panics_are_errors() {
    // Each of these was an assert!/expect panic reachable from user input.
    assert!(inv_norm_cdf(1.5).is_err());
    assert!(inv_norm_cdf(f64::NAN).is_err());
    assert!(reg_inc_beta(-1.0, 1.0, 0.5).is_err());
    assert!(reg_inc_beta(1.0, 1.0, 2.0).is_err());
    assert!(reg_lower_gamma(-1.0, 1.0).is_err());
    assert!(reg_upper_gamma(1.0, f64::NAN).is_err());
    assert!(chi2_sf(1.0, f64::NAN).is_err());
    // Defensive: the old partial_cmp().expect() was shielded by an earlier
    // finiteness check; the sort now uses total_cmp and cannot panic.
    assert!(var_standard_error(&[f64::NAN; 20], 0.01).is_err());
    let cov1 = FactorCov {
        factors: vec!["FX:EUR".to_string()],
        cov: Matrix::from_rows(&[vec![1e-4]]).unwrap(),
    };
    assert!(
        simulate_factor_returns(
            &cov1,
            10,
            McDist::StudentT,
            f64::NAN,
            &JumpSpec::default(),
            1,
            1.0
        )
        .is_err()
    );
    assert!(student_t_quantile(0.3, f64::NAN).is_err());
    let mut rng = Rng::new(1);
    assert!(rng.gamma(-1.0).is_nan());
    assert!(rng.gamma(f64::NAN).is_nan());
    #[allow(deprecated)]
    let ok = cornish_fisher_domain_ok(0.0, 0.0, 4.0, 0); // n_grid < 3 used to assert!
    assert!(ok);
}
