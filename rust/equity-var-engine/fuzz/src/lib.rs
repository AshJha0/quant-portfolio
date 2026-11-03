//! Fuzz property for the eq-var-engine Cholesky with jitter ladder and PSD
//! repair (`psd::cholesky_repair`). Same input mapping and property as the
//! C++ target `cpp/core/fuzz/fuzz_cholesky_repair.cpp`.

use eq_var_engine::matrix::Matrix;
use eq_var_engine::psd::{PsdRepair, cholesky_repair};
use qp_core_fuzz::FuzzInput;

/// For n in 1..=6, vols in [1e-4, 10] and raw correlations in [-1.5, 1.5]:
/// `cholesky_repair` returns Ok or Err, never panics. On Ok, the factor is
/// finite and lower triangular, L L^T equals the factored matrix (input or
/// repair, plus jitter on the diagonal) to 1e-8 of the largest variance,
/// and a repaired matrix is symmetric and keeps the input variances to
/// 1e-8 relative.
pub fn check_cholesky_repair(data: &[u8]) {
    let mut fin = FuzzInput::new(data);
    let n = 1 + (fin.u64() % 6) as usize;
    let repair = match fin.u64() % 3 {
        0 => PsdRepair::None,
        1 => PsdRepair::Clip,
        _ => PsdRepair::Higham,
    };
    let vol: Vec<f64> = (0..n).map(|_| fin.in_range(1e-4, 10.0)).collect();
    let mut a = vec![0.0; n * n];
    let mut vmax: f64 = 0.0;
    for i in 0..n {
        a[i * n + i] = vol[i] * vol[i];
        vmax = vmax.max(a[i * n + i]);
        for j in 0..i {
            let c = fin.in_range(-1.5, 1.5) * vol[i] * vol[j];
            a[i * n + j] = c;
            a[j * n + i] = c;
        }
    }
    let m = Matrix::from_vec(n, n, a.clone()).expect("square");
    let Ok(f) = cholesky_repair(&m, repair) else {
        return;
    };
    let target: Vec<f64> = match &f.repaired {
        None => a.clone(),
        Some(r) => {
            let mut t = vec![0.0; n * n];
            for i in 0..n {
                assert!(
                    (r.get(i, i) - a[i * n + i]).abs() <= 1e-8 * a[i * n + i],
                    "variance moved"
                );
                for j in 0..n {
                    let asym = (r.get(i, j) - r.get(j, i)).abs();
                    assert!(asym <= 1e-12 * vmax, "repair not symmetric: {asym:e}");
                    t[i * n + j] = r.get(i, j);
                }
            }
            t
        }
    };
    for i in 0..n {
        for j in 0..n {
            let lij = f.factor.get(i, j);
            assert!(lij.is_finite());
            if j > i {
                assert_eq!(lij, 0.0);
            }
            let s: f64 = (0..=i.min(j))
                .map(|k| f.factor.get(i, k) * f.factor.get(j, k))
                .sum();
            let want = target[i * n + j] + if i == j { f.jitter } else { 0.0 };
            assert!(
                (s - want).abs() <= 1e-8 * vmax,
                "({i},{j}) {s:e} vs {want:e}"
            );
        }
    }
}
