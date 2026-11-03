#![allow(clippy::needless_range_loop)]
//! Regression tests for the shared conventions S1-S11 (ws-var round 1).
//!
//! Every test here failed on the code before this change (wrong number,
//! panic, hang, or a missing entry point) and passes now.

use eq_var_engine::backtest::*;
use eq_var_engine::historical::*;
use eq_var_engine::matrix::Matrix;
use eq_var_engine::monte_carlo::*;
use eq_var_engine::parametric::*;
use eq_var_engine::psd::*;
use eq_var_engine::rng::Rng;
use eq_var_engine::stats::*;
use eq_var_engine::tail::*;
use eq_var_engine::{EqVarError, TailModel};

fn msg(e: EqVarError) -> String {
    e.to_string()
}

// ---------------------------------------------------------------- S1 alpha

#[test]
fn alpha_is_tail_probability_and_confidence_levels_are_rejected() {
    let pnl: Vec<f64> = (0..100).map(|i| (i as f64 * 0.7).sin()).collect();
    for bad in [0.5, 0.95, 0.99, 0.0, -0.01, f64::NAN, f64::INFINITY] {
        let e = historical_var(&pnl, bad).unwrap_err();
        assert!(
            msg(e).contains("alpha is the tail probability"),
            "alpha={bad}"
        );
    }
    let e = cornish_fisher_valid(0.0, 0.0, 0.99).unwrap_err();
    assert!(msg(e).contains("tail probability"));
}

// ------------------------------------------------------------- S2 estimator

#[test]
fn order_statistic_var_and_fractional_es_by_hand() {
    // Losses 1..=10, alpha = 0.25: k = 2.5, m = 3, VaR = 8,
    // ES = (10 + 9 + 0.5 * 8) / 2.5 = 9.2.
    let pnl: Vec<f64> = (1..=10).map(|i| -(i as f64)).collect();
    let (v, es) = empirical_var_es(&pnl, 0.25).unwrap();
    assert_eq!(v, 8.0);
    assert!((es - 9.2).abs() < 1e-14);
    // Integer k up to rounding stays on the integer rank: 250 * 0.02 = 5.000000000000001.
    assert_eq!(tail_rank(250, 0.02).unwrap(), 5);
    // Old code: type-7 interpolation gave 95.05 here.
    let p100: Vec<f64> = (1..=100).map(|i| -(i as f64)).collect();
    assert_eq!(historical_var(&p100, 0.05).unwrap(), 96.0);
}

#[test]
fn weighted_estimator_matches_unweighted_with_equal_weights() {
    let pnl: Vec<f64> = (0..250)
        .map(|i| 100.0 * ((0.37 * i as f64) + 0.1).sin())
        .collect();
    for a in [0.01, 0.025, 0.05] {
        let (v, e) = empirical_var_es(&pnl, a).unwrap();
        let (vw, ew) = weighted_var_es(&pnl, &vec![1.0; 250], a).unwrap();
        assert_eq!(v, vw);
        assert!((e - ew).abs() <= 1e-12 * e.abs(), "a={a}: {e} vs {ew}");
    }
}

// ------------------------------------------------------------------ S3 Kupiec

#[test]
fn kupiec_struct_matches_formula_and_rejects_bad_counts() {
    let r = kupiec_pof(&KupiecInput {
        n_obs: 250,
        n_exceptions: 5,
        alpha: 0.01,
    })
    .unwrap();
    let (n, x, a) = (250.0_f64, 5.0_f64, 0.01_f64);
    let p = x / n;
    let lr = -2.0 * ((n - x) * (1.0 - a).ln() + x * a.ln())
        + 2.0 * ((n - x) * (1.0 - p).ln() + x * p.ln());
    assert!((r.lr - lr).abs() < 1e-12);
    assert!(
        kupiec_pof(&KupiecInput {
            n_obs: 250,
            n_exceptions: 251,
            alpha: 0.01
        })
        .is_err()
    );
    assert!(
        kupiec_pof(&KupiecInput {
            n_obs: 250,
            n_exceptions: 5,
            alpha: 0.99
        })
        .is_err()
    );
    #[allow(deprecated)]
    let old = kupiec_pof_positional(250, 5, 0.01).unwrap();
    assert_eq!(old, r);
}

// ------------------------------------------------------------------- S4 Basel

#[test]
fn basel_zone_follows_the_window_not_the_raw_count() {
    let z = |n, x, a| {
        basel_traffic_light(&BaselInput {
            n_obs: n,
            n_exceptions: x,
            alpha: a,
        })
        .unwrap()
    };
    // Old code: 5 raw exceptions were yellow whatever the window.
    assert_eq!(z(500, 5, 0.01).zone, BaselZone::Green);
    assert_eq!(z(250, 4, 0.01).zone, BaselZone::Green);
    for (x, plus) in [(5, 0.40), (6, 0.50), (7, 0.65), (8, 0.75), (9, 0.85)] {
        let r = z(250, x, 0.01);
        assert_eq!(r.zone, BaselZone::Yellow);
        assert_eq!(r.plus_factor, plus);
        assert!((r.multiplier - 3.0 - plus).abs() < 1e-15);
    }
    assert_eq!(z(250, 10, 0.01).zone, BaselZone::Red);
    // Old code clamped x > n; now an error.
    assert!(
        basel_traffic_light(&BaselInput {
            n_obs: 250,
            n_exceptions: 300,
            alpha: 0.01
        })
        .is_err()
    );
    // Zone thresholds on F(x) for other windows.
    for (n, x, a) in [
        (500, 8, 0.01),
        (500, 9, 0.01),
        (500, 12, 0.01),
        (1000, 15, 0.01),
        (250, 12, 0.025),
    ] {
        let r = z(n, x, a);
        let f = r.cumulative_prob;
        let want = if f < 0.95 {
            BaselZone::Green
        } else if f < 0.9999 {
            BaselZone::Yellow
        } else {
            BaselZone::Red
        };
        assert_eq!(r.zone, want, "n={n} x={x}");
    }
}

#[test]
fn basel_cdf_matches_incomplete_beta() {
    for (n, x, a) in [
        (250_i64, 4_i64, 0.01),
        (500, 9, 0.01),
        (1000, 15, 0.01),
        (250, 12, 0.025),
    ] {
        let r = basel_traffic_light(&BaselInput {
            n_obs: n,
            n_exceptions: x,
            alpha: a,
        })
        .unwrap();
        let beta = binomial_cdf(x as u64, n as u64, a).unwrap();
        assert!((r.cumulative_prob - beta).abs() < 1e-12, "n={n} x={x}");
    }
}

// --------------------------------------------------------------------- S5 Z2

#[test]
fn acerbi_szekely_z2_sign_and_value() {
    // Exceptions exactly at forecast ES give Z2 = 1 - x / (T alpha).
    let pnl = [-2.0, 0.0, -2.0, 1.0];
    let z2 = acerbi_szekely_z2(&pnl, &[1.0], &[2.0], 0.25).unwrap();
    assert!((z2 - (1.0 - 2.0 / 1.0)).abs() < 1e-15);
    // Losses beyond ES make Z2 more negative (ES understated).
    let worse = acerbi_szekely_z2(&[-4.0, 0.0, -4.0, 1.0], &[1.0], &[2.0], 0.25).unwrap();
    assert!(worse < z2);
    assert!(acerbi_szekely_z2(&pnl, &[1.0], &[0.0], 0.25).is_err());
    assert!(acerbi_szekely_z2(&[], &[1.0], &[2.0], 0.25).is_err());
}

// ---------------------------------------------------------------------- S6 CF

#[test]
fn cornish_fisher_validity_follows_alpha() {
    // The old fixed |z| <= 3.5 check is replaced by [-z_a, z_a].
    assert!(cornish_fisher_valid(0.122, -0.427, 0.01).unwrap());
    assert!(cornish_fisher_valid(0.122, -0.427, 0.001).unwrap());
    // z_a > 4 (alpha = 1e-5, z_a = 4.26): the endpoint derivative is negative.
    assert!(!cornish_fisher_valid(0.122, -0.427, 1e-5).unwrap());
    assert!(cornish_fisher_valid(0.0, 0.0, 0.01).unwrap());
    assert!(cornish_fisher_valid(f64::NAN, 0.0, 0.01).is_err());
    assert!(cornish_fisher_valid(0.0, f64::INFINITY, 0.01).is_err());
}

// ---------------------------------------------------------------- S7 Cholesky

fn bad3() -> Matrix {
    Matrix::from_vec(3, 3, vec![1.0, 0.9, 0.9, 0.9, 1.0, -0.9, 0.9, -0.9, 1.0]).unwrap()
}

fn max_abs_diff(a: &Matrix, b: &Matrix) -> f64 {
    let mut m = 0.0_f64;
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            m = m.max((a.get(i, j) - b.get(i, j)).abs());
        }
    }
    m
}

#[test]
fn indefinite_matrix_errors_with_eigenvalue_and_repairs_work() {
    let a = bad3();
    let e = cholesky_repair(&a, PsdRepair::None).unwrap_err();
    assert!(msg(e).contains("most negative eigenvalue"));
    let (vals, _) = symmetric_eigen(&a).unwrap();
    assert!(
        (vals[0] - (1.0 - 1.8)).abs() < 1e-12,
        "lambda_min = {}",
        vals[0]
    );

    let h = cholesky_repair(&a, PsdRepair::Higham).unwrap();
    let rh = h.repaired.unwrap();
    for i in 0..3 {
        assert!((rh.get(i, i) - 1.0).abs() < 1e-12);
    }
    let (vh, _) = symmetric_eigen(&rh).unwrap();
    assert!(vh[0] > -1e-10);

    let c = cholesky_repair(&a, PsdRepair::Clip).unwrap();
    let rc = c.repaired.unwrap();
    for i in 0..3 {
        assert!((rc.get(i, i) - a.get(i, i)).abs() < 1e-15);
    }
    // Higham is the nearest unit-diagonal PSD matrix: closer than clipping.
    assert!(max_abs_diff(&rh, &a) <= max_abs_diff(&rc, &a) + 1e-12);
}

#[test]
fn eigen_reconstructs_the_matrix() {
    let a = Matrix::from_vec(3, 3, vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0]).unwrap();
    let (vals, v) = symmetric_eigen(&a).unwrap();
    let mut b = Matrix::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            let s: f64 = (0..3).map(|k| v.get(i, k) * vals[k] * v.get(j, k)).sum();
            b.set(i, j, s);
        }
    }
    assert!(max_abs_diff(&a, &b) < 1e-13);
}

#[test]
fn singular_matrix_reports_jitter_within_cap() {
    // Exactly representable rank-1 matrix: second pivot is exactly zero.
    let a = Matrix::from_vec(2, 2, vec![6.25e-2, 3.125e-2, 3.125e-2, 1.5625e-2]).unwrap();
    let f = cholesky_repair(&a, PsdRepair::None).unwrap();
    let md = (6.25e-2 + 1.5625e-2) / 2.0;
    assert!(f.jitter > 0.0 && f.jitter <= 1e-6 * md);
    assert!(!f.warnings.is_empty());
}

#[test]
fn relative_symmetry_check() {
    let tiny = Matrix::from_vec(2, 2, vec![1e-4, 1e-13, 0.0, 1e-4]).unwrap();
    // |diff| = 1e-13 > 1e-10 * 1e-4 = 1e-14: asymmetric under the relative rule.
    assert!(cholesky_repair(&tiny, PsdRepair::None).is_err());
}

// -------------------------------------------------------------------- S8 EWMA

#[test]
fn ewma_seed_uses_first_30_observations_only() {
    let mut x: Vec<f64> = (0..60).map(|i| 0.01 * ((i as f64) * 0.9).sin()).collect();
    let (s, _) = ewma_volatility_forecast(&x, 0.94).unwrap();
    // Changing data after day 30 must not move sigma[0] (no full-sample seed).
    for v in x.iter_mut().skip(30) {
        *v *= 10.0;
    }
    let (s2, _) = ewma_volatility_forecast(&x, 0.94).unwrap();
    assert_eq!(s[0], s2[0]);
    let seed: f64 = x[..30].iter().map(|v| v * v).sum::<f64>() / 30.0;
    assert!((s[0] - seed.sqrt()).abs() < 1e-17);
}

// --------------------------------------------------------------------- S9 FHS

#[test]
fn per_factor_fhs_matches_hand_recipe() {
    let t = 80;
    let mut r = Matrix::zeros(t, 2);
    for i in 0..t {
        let calm = 0.01 * ((i as f64) * 1.3).sin();
        let regime = if i >= 60 { 0.05 } else { 0.005 } * ((i as f64) * 0.7).cos();
        r.set(i, 0, calm);
        r.set(i, 1, regime);
    }
    let w = [1.0e6, -2.0e6];
    let lam = 0.94;
    let mut pnl = vec![0.0; t];
    for j in 0..2 {
        let col: Vec<f64> = (0..t).map(|i| r.get(i, j)).collect();
        let mut s2 = col[..30].iter().map(|v| v * v).sum::<f64>() / 30.0;
        let mut path = Vec::new();
        for v in &col {
            path.push(s2);
            s2 = lam * s2 + (1.0 - lam) * v * v;
        }
        let next = s2.sqrt();
        for i in 0..t {
            pnl[i] += w[j] * col[i] / path[i].sqrt() * next;
        }
    }
    let want = empirical_var(&pnl, 0.05).unwrap();
    let got = filtered_historical_var_factors(&r, &w, 0.05, lam).unwrap();
    assert!((got - want).abs() <= 1e-9 * want.abs());
    assert!(filtered_historical_var_factors(&r, &[], 0.05, lam).is_err());
}

// ---------------------------------------------------------------------- S10 MC

#[test]
fn monte_carlo_run_simulates_once_and_bootstrap_is_primary() {
    let cov = Matrix::from_vec(2, 2, vec![1e-4, 2e-5, 2e-5, 4e-4]).unwrap();
    let w = [1.0e6, 5.0e5];
    let run = monte_carlo_run(&w, &cov, 0.01, 20_000, TailModel::Normal, 7, 200).unwrap();
    let (v, e) = empirical_var_es(&run.pnl, 0.01).unwrap();
    assert_eq!(run.var, v);
    assert_eq!(run.es, e);
    assert_eq!(
        run.var,
        monte_carlo_var(&w, &cov, 0.01, 20_000, TailModel::Normal, 7).unwrap()
    );
    assert!(run.se > 0.0 && run.se_kde > 0.0);
    assert!(run.se < 3.0 * run.se_kde && run.se_kde < 3.0 * run.se);
    assert!(monte_carlo_run(&w, &cov, 0.01, 20_000, TailModel::Normal, 7, 1).is_err());
    assert!(var_bootstrap_se(&run.pnl, 0.01, 1, 0).is_err());
}

// ------------------------------------------------------------- S11 empty book

#[test]
fn empty_book_errors_everywhere() {
    let cov = Matrix::zeros(0, 0);
    let e = monte_carlo_pnl(&[], &cov, 1000, TailModel::Normal, 1).unwrap_err();
    assert!(msg(e).contains("book has no positions"));
    assert!(parametric_var(&[], &cov, 0.01, TailModel::Normal).is_err());
}

// --------------------------------------------------------- H-1 panic paths

#[test]
fn special_functions_never_panic_or_hang_on_non_finite_input() {
    // Old erfc(NaN) recursed without end; erfc(inf) panicked in expect().
    assert!(erfc(f64::NAN).is_nan());
    assert_eq!(erfc(f64::INFINITY), 0.0);
    assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
    assert_eq!(normal_cdf(f64::INFINITY), 1.0);
    assert!(erf(f64::NAN).is_nan());
    // Old Rng::gamma(NaN) never left its rejection loop.
    let mut rng = Rng::new(1);
    assert!(rng.gamma(f64::NAN).is_nan());
    assert!(rng.chi_square(-1.0).is_nan());
    // ln_gamma is defined on x > 0 only; it used to return inf / NaN silently.
    assert!(ln_gamma(-1.0).is_err());
    assert!(ln_gamma(0.0).is_err());
    assert!(chi2_sf(f64::NAN, 1.0).is_err());
}

#[test]
fn ewma_covariance_seed_uses_first_window_rows_only() {
    use eq_var_engine::matrix::{EWMA_COV_SEED_WINDOW, ewma_covariance};
    // Two panels identical in the seed window, wildly different afterwards.
    // With lam -> 1 the forecast stays at the seed, so both must agree.
    let w = EWMA_COV_SEED_WINDOW;
    let t = w + 20;
    let mut a = Vec::with_capacity(t * 2);
    let mut b = Vec::with_capacity(t * 2);
    for i in 0..t {
        let x = 0.01 * ((i as f64) * 0.7).sin();
        let y = 0.01 * ((i as f64) * 1.3).cos();
        a.extend_from_slice(&[x, y]);
        if i < w {
            b.extend_from_slice(&[x, y]);
        } else {
            b.extend_from_slice(&[0.0, 0.0]);
        }
    }
    let ma = Matrix::from_vec(t, 2, a.clone()).unwrap();
    let mb = Matrix::from_vec(t, 2, b).unwrap();
    let ea = ewma_covariance(&ma, 1.0 - 1e-12).unwrap();
    let eb = ewma_covariance(&mb, 1.0 - 1e-12).unwrap();
    let mut seed = [0.0_f64; 4];
    for i in 0..w {
        let (x, y) = (a[2 * i], a[2 * i + 1]);
        seed[0] += x * x;
        seed[1] += x * y;
        seed[3] += y * y;
    }
    for v in &mut seed {
        *v /= w as f64;
    }
    seed[2] = seed[1];
    for (j, k, s) in [
        (0, 0, seed[0]),
        (0, 1, seed[1]),
        (1, 0, seed[2]),
        (1, 1, seed[3]),
    ] {
        assert!((ea.get(j, k) - s).abs() < 1e-12, "a[{j},{k}]");
        assert!((eb.get(j, k) - s).abs() < 1e-12, "b[{j},{k}]");
    }
}
