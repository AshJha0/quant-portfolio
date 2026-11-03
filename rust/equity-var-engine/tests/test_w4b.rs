//! W4(b): per-path Philox streams and AS 241 normals for the linear MC kernel.
//!
//! * the P&L is bit-identical on 1, 2, 4 and 8 threads;
//! * the allocation-free workspace gives the same bits as the main API;
//! * fixed values pin the stream definition; the C++ twin
//!   (`cpp/equity-var-engine/tests/test_w4b.cpp`) checks the same numbers.

use eq_var_engine::fast::McWorkspace;
use eq_var_engine::prelude::*;
use eq_var_engine::rng::PathRng;

fn book(n: usize) -> (Vec<f64>, Matrix) {
    let vols: Vec<f64> = (0..n).map(|i| 0.01 + 0.0005 * i as f64).collect();
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            c[i * n + j] = if i == j { 1.0 } else { 0.3 };
        }
    }
    let corr = Matrix::from_vec(n, n, c).unwrap();
    let w = (0..n)
        .map(|i| 1e6 * (1.0 + (i % 3) as f64) * if i % 4 == 0 { -1.0 } else { 1.0 })
        .collect();
    (w, covariance_from_vols(&vols, &corr).unwrap())
}

#[test]
fn bit_identical_across_thread_counts() {
    let (w, cov) = book(12);
    for tail in [TailModel::Normal, TailModel::StudentT { df: 5.0 }] {
        let one = monte_carlo_pnl_threads(&w, &cov, 20_000, tail, 11, 1).unwrap();
        for t in [2usize, 4, 8] {
            let many = monte_carlo_pnl_threads(&w, &cov, 20_000, tail, 11, t).unwrap();
            assert!(
                one.iter()
                    .zip(&many)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{t} threads"
            );
        }
        let dflt = monte_carlo_pnl(&w, &cov, 20_000, tail, 11).unwrap();
        assert!(
            one.iter()
                .zip(&dflt)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
}

#[test]
fn workspace_equals_main_api() {
    let (w, cov) = book(50);
    let mut ws = McWorkspace::new(50, 10_000, 50);
    for tail in [TailModel::Normal, TailModel::StudentT { df: 4.5 }] {
        ws.prepare(&cov, tail).unwrap();
        for seed in [0u64, 7, 1 << 40] {
            let a = ws.var(&w, 0.01, 10_000, seed).unwrap();
            let b = monte_carlo_var(&w, &cov, 0.01, 10_000, tail, seed).unwrap();
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

/// Pins the stream: path p of seed s is Philox4x32-10 stream (s, p), one
/// AS 241 normal per 64-bit word. Values printed by this engine; the C++
/// twin asserts the same numbers to 1e-15 relative.
#[test]
#[allow(clippy::excessive_precision)]
fn stream_definition_is_pinned() {
    let mut r = PathRng::new(42, 3);
    let z: Vec<f64> = (0..4).map(|_| r.standard_normal()).collect();
    let chi = PathRng::new(42, 3).chi_square(5.0);
    let (w, cov) = book(50);
    let v = monte_carlo_var(&w, &cov, 0.01, 10_000, TailModel::Normal, 2026).unwrap();
    let vt = monte_carlo_var(
        &w,
        &cov,
        0.01,
        10_000,
        TailModel::StudentT { df: 5.0 },
        2026,
    )
    .unwrap();
    let want_z = PIN_Z;
    for (a, b) in z.iter().zip(want_z) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    assert_eq!(chi.to_bits(), PIN_CHI.to_bits());
    assert_eq!(v.to_bits(), PIN_V.to_bits());
    assert_eq!(vt.to_bits(), PIN_VT.to_bits());
}

const PIN_Z: [f64; 4] = [
    0.4416806772623007,
    -0.3077768475777059,
    0.4693154650693704,
    -1.8602711545492243,
];
const PIN_CHI: f64 = 5.767996648372349;
const PIN_V: f64 = 1616611.6346006482;
const PIN_VT: f64 = 1679342.0463568298;

/// The re-baselined normal VaR sits within 3 order-statistic SE of the
/// closed form sigma * z_0.99 (the W5 oracle checks the same at 200k paths).
#[test]
fn rebaseline_agrees_with_closed_form() {
    let (w, cov) = book(50);
    let pnl = monte_carlo_pnl(&w, &cov, 10_000, TailModel::Normal, 2026).unwrap();
    let v = empirical_var(&pnl, 0.01).unwrap();
    let se = var_bootstrap_se(&pnl, 0.01, 200, 5).unwrap();
    let exact = parametric_var(&w, &cov, 0.01, TailModel::Normal).unwrap();
    assert!((v - exact).abs() <= 3.0 * se, "{v} vs {exact} (se {se})");
}
