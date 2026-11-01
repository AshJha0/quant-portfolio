//! The allocation-free API returns the same bits as the main API, and
//! repeated runs with one seed repeat bit for bit.

use eq_var_engine::fast::*;
use eq_var_engine::prelude::*;

fn sample(seed: u64, n: usize, fat: bool) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let z = rng.standard_normal();
            if fat {
                1.0e4 * z * z * z
            } else {
                2.5e3 * z + 10.0
            }
        })
        .collect()
}

fn book(seed: u64, n: usize) -> (Vec<f64>, Matrix) {
    let mut rng = Rng::new(seed);
    let mut cov = Matrix::zeros(n, n);
    let vols: Vec<f64> = (0..n).map(|j| 0.005 + 0.002 * j as f64).collect();
    for i in 0..n {
        for j in 0..n {
            let rho = if i == j { 1.0 } else { 0.35 };
            cov.set(i, j, rho * vols[i] * vols[j]);
        }
    }
    let w = (0..n).map(|_| 1.0e5 * rng.standard_normal()).collect();
    (w, cov)
}

const ALPHAS: [f64; 5] = [0.001, 0.01, 0.025, 0.05, 0.3];

#[test]
fn empirical_kernels_match_bits() {
    for seed in 0..40u64 {
        for &n in &[50usize, 51, 99, 250, 1000] {
            let pnl = sample(seed, n, seed % 2 == 0);
            let mut work = vec![0.0; n];
            for &a in &ALPHAS {
                let old = historical_var(&pnl, a).unwrap();
                let new = historical_var_ws(&pnl, a, &mut work).unwrap();
                assert_eq!(new.to_bits(), old.to_bits());
                let old = expected_shortfall(&pnl, a).unwrap();
                let new = expected_shortfall_ws(&pnl, a, &mut work).unwrap();
                assert_eq!(new.to_bits(), old.to_bits());
                let (ov, oe) = empirical_var_es(&pnl, a).unwrap();
                let (nv, ne) = empirical_var_es_ws(&pnl, a, &mut work).unwrap();
                assert_eq!((nv.to_bits(), ne.to_bits()), (ov.to_bits(), oe.to_bits()));
            }
            let mut est = vec![0.0; 50];
            let old = var_bootstrap_se(&pnl, 0.01, 50, seed).unwrap();
            let new = var_bootstrap_se_ws(&pnl, 0.01, 50, seed, &mut work, &mut est).unwrap();
            assert_eq!(new.to_bits(), old.to_bits());
        }
    }
}

#[test]
fn parametric_kernels_match_bits() {
    for seed in 0..30u64 {
        for &n in &[1usize, 2, 7, 30] {
            let (w, cov) = book(seed, n);
            assert_eq!(
                portfolio_sigma_fast(&w, &cov).unwrap().to_bits(),
                portfolio_sigma(&w, &cov).unwrap().to_bits()
            );
            for tail in [TailModel::Normal, TailModel::StudentT { df: 4.5 }] {
                for &a in &ALPHAS {
                    assert_eq!(
                        parametric_var_fast(&w, &cov, a, tail).unwrap().to_bits(),
                        parametric_var(&w, &cov, a, tail).unwrap().to_bits()
                    );
                    assert_eq!(
                        parametric_es_fast(&w, &cov, a, tail).unwrap().to_bits(),
                        parametric_es(&w, &cov, a, tail).unwrap().to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn mc_workspace_matches_bits_and_repeats() {
    for seed in 0..6u64 {
        for &n in &[1usize, 3, 12] {
            let (w, cov) = book(seed + 100, n);
            for tail in [TailModel::Normal, TailModel::StudentT { df: 5.0 }] {
                let paths = 3_000;
                let old = monte_carlo_run(&w, &cov, 0.01, paths, tail, seed, 40).unwrap();
                let mut ws = McWorkspace::new(n, paths, 40);
                ws.prepare(&cov, tail).unwrap();
                let s = ws.run(&w, 0.01, paths, seed, 40).unwrap();
                assert_eq!(s.var.to_bits(), old.var.to_bits());
                assert_eq!(s.es.to_bits(), old.es.to_bits());
                assert_eq!(s.se.to_bits(), old.se.to_bits());
                assert_eq!(s.se_kde.to_bits(), old.se_kde.to_bits());
                assert_eq!(s.jitter.to_bits(), old.jitter.to_bits());
                let bits: Vec<u64> = ws.pnl(paths).iter().map(|v| v.to_bits()).collect();
                let old_bits: Vec<u64> = old.pnl.iter().map(|v| v.to_bits()).collect();
                assert_eq!(bits, old_bits);
                // Same seed, same workspace, second run: same bits.
                let again = ws.run(&w, 0.01, paths, seed, 40).unwrap();
                assert_eq!(again.var.to_bits(), s.var.to_bits());
                assert_eq!(again.se.to_bits(), s.se.to_bits());
                let v = ws.var(&w, 0.01, paths, seed).unwrap();
                let ov = monte_carlo_var(&w, &cov, 0.01, paths, tail, seed).unwrap();
                assert_eq!(v.to_bits(), ov.to_bits());
            }
        }
    }
}

#[test]
fn fast_errors_are_reported_not_panics() {
    let pnl = sample(1, 100, false);
    let mut short = vec![0.0; 5];
    assert_eq!(historical_var_ws(&pnl, 0.01, &mut short), Err(VarFault::WorkspaceTooSmall));
    assert_eq!(historical_var_ws(&pnl[..10], 0.01, &mut short), Err(VarFault::TooFewObservations));
    let mut bad = pnl.clone();
    bad[3] = f64::NAN;
    let mut work = vec![0.0; 100];
    assert_eq!(historical_var_ws(&bad, 0.01, &mut work), Err(VarFault::NonFinite));
    assert_eq!(expected_shortfall_ws(&pnl, 0.0, &mut work), Err(VarFault::BadAlpha));
    let (w, cov) = book(2, 3);
    assert_eq!(parametric_var_fast(&w[..2], &cov, 0.01, TailModel::Normal), Err(VarFault::ShapeMismatch));
    assert_eq!(
        parametric_var_fast(&w, &cov, 0.01, TailModel::StudentT { df: 2.0 }),
        Err(VarFault::BadParameter)
    );
    let mut ws = McWorkspace::new(3, 200, 10);
    assert_eq!(ws.run(&w, 0.01, 200, 1, 10), Err(VarFault::NotPrepared));
    ws.prepare(&cov, TailModel::Normal).unwrap();
    assert_eq!(ws.run(&w, 0.01, 500, 1, 10), Err(VarFault::WorkspaceTooSmall));
    assert_eq!(ws.run(&w, 0.01, 200, 1, 11), Err(VarFault::WorkspaceTooSmall));
    assert_eq!(ws.run(&w, 0.01, 50, 1, 10), Err(VarFault::TooFewObservations));
    assert_eq!(ws.run(&w, 0.01, 200, 1, 1), Err(VarFault::BadParameter));
    assert_eq!(ws.var(&w[..1], 0.01, 200, 1), Err(VarFault::ShapeMismatch));
    let mut ws2 = McWorkspace::new(2, 200, 10);
    assert_eq!(ws2.prepare(&cov, TailModel::Normal), Err(VarFault::ShapeMismatch));
}
