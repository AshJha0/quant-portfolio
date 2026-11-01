//! Symmetric eigen-decomposition and Cholesky with PSD repair (S7).
//!
//! [`cholesky_repair`] is the one factorisation every simulation path uses:
//!
//! * symmetry is checked relative to the largest entry,
//!   `|A - A'| <= 1e-10 max|A|`, else an error;
//! * plain Cholesky is tried first; on failure a diagonal jitter
//!   `eps I` is added, `eps = 1e-12 mean(diag)` rising by 10x per try up to
//!   the cap `1e-6 mean(diag)` ([`MAX_RELATIVE_JITTER`]). Any jitter used is
//!   returned in [`CholeskyFactor::jitter`] and described in `warnings`;
//! * past the cap, [`PsdRepair::None`] returns an error naming the most
//!   negative eigenvalue; [`PsdRepair::Clip`] clips eigenvalues to
//!   `1e-10 mean(diag)` and restores the original diagonal;
//!   [`PsdRepair::Higham`] runs Higham (2002) alternating projections with
//!   Dykstra's correction on the correlation matrix and rescales by the
//!   original vols. The repaired matrix is returned in
//!   [`CholeskyFactor::repaired`].
//!
//! The eigen solver is cyclic Jacobi: simple, accurate to a few ulps of
//! the matrix norm, and fast enough for the factor counts used here.

use crate::matrix::Matrix;
use crate::{FxVarError, Result};

/// Largest diagonal jitter, as a multiple of the mean diagonal.
///
/// `1e-6` is a part per million of the average variance: enough to repair a
/// covariance that is singular or indefinite only by rounding, and far too
/// small to hide a materially indefinite one.
pub const MAX_RELATIVE_JITTER: f64 = 1e-6;

/// First jitter rung, as a multiple of the mean diagonal.
pub const MIN_RELATIVE_JITTER: f64 = 1e-12;

/// Relative symmetry tolerance: `|a_ij - a_ji| <= 1e-10 max|A|`.
pub const SYMMETRY_RTOL: f64 = 1e-10;

/// What [`cholesky_repair`] may do with a materially indefinite matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsdRepair {
    /// No repair: error past the jitter cap.
    None,
    /// Clip eigenvalues to `1e-10 mean(diag)`, keep the diagonal.
    Clip,
    /// Higham (2002) nearest correlation matrix, rescaled by the vols.
    Higham,
}

/// Result of [`cholesky_repair`].
#[derive(Debug, Clone, PartialEq)]
pub struct CholeskyFactor {
    /// Lower-triangular `L` with `L L' = A_used + jitter I`.
    pub factor: Matrix,
    /// Diagonal jitter added (0 if none was needed).
    pub jitter: f64,
    /// The repaired matrix when a repair ran, else `None`.
    pub repaired: Option<Matrix>,
    /// Human-readable notes on any jitter or repair applied.
    pub warnings: Vec<String>,
}

fn check_square_finite(a: &Matrix) -> Result<usize> {
    if a.rows() != a.cols() {
        return Err(FxVarError::Invalid(format!(
            "matrix must be square, got {} x {}",
            a.rows(),
            a.cols()
        )));
    }
    if a.rows() == 0 {
        return Err(FxVarError::Invalid("matrix is empty".to_string()));
    }
    if !a.all_finite() {
        return Err(FxVarError::Invalid(
            "matrix contains NaN or infinite entries".to_string(),
        ));
    }
    Ok(a.rows())
}

fn max_abs(a: &Matrix) -> f64 {
    let n = a.rows();
    let mut m = 0.0_f64;
    for i in 0..n {
        for j in 0..a.cols() {
            m = m.max(a.get(i, j).abs());
        }
    }
    m
}

/// Relative symmetry check `|a_ij - a_ji| <= 1e-10 max|A|`.
///
/// # Errors
/// Non-square, empty or non-finite input, or an asymmetric matrix.
pub fn check_symmetric(a: &Matrix) -> Result<()> {
    let n = check_square_finite(a)?;
    let tol = SYMMETRY_RTOL * max_abs(a);
    for i in 0..n {
        for j in (i + 1)..n {
            if (a.get(i, j) - a.get(j, i)).abs() > tol {
                return Err(FxVarError::Invalid(format!(
                    "matrix is not symmetric: |a[{i},{j}] - a[{j},{i}]| = {:e} > {tol:e} \
                     (1e-10 x max|A|)",
                    (a.get(i, j) - a.get(j, i)).abs()
                )));
            }
        }
    }
    Ok(())
}

/// Eigenvalues (ascending) and eigenvectors (columns of the returned
/// matrix) of a symmetric matrix, by cyclic Jacobi rotations.
///
/// # Errors
/// Non-square, empty, non-finite or asymmetric input; a numerical error if
/// 100 sweeps do not converge.
pub fn symmetric_eigen(a: &Matrix) -> Result<(Vec<f64>, Matrix)> {
    check_symmetric(a)?;
    let n = a.rows();
    let mut m = a.clone();
    // Symmetrise exactly so the rotations act on a symmetric matrix.
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (m.get(i, j) + m.get(j, i));
            m.set(i, j, v);
            m.set(j, i, v);
        }
    }
    let mut v = Matrix::identity(n);
    let mut frob2 = 0.0;
    for i in 0..n {
        for j in 0..n {
            frob2 += m.get(i, j) * m.get(i, j);
        }
    }
    let mut converged = false;
    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m.get(i, j) * m.get(i, j);
            }
        }
        if off <= 1e-32 * frob2 {
            converged = true;
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m.get(p, q);
                if apq == 0.0 {
                    continue;
                }
                let app = m.get(p, p);
                let aqq = m.get(q, q);
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / t.hypot(1.0);
                let s = t * c;
                for k in 0..n {
                    if k != p && k != q {
                        let mkp = m.get(k, p);
                        let mkq = m.get(k, q);
                        let np = c * mkp - s * mkq;
                        let nq = s * mkp + c * mkq;
                        m.set(k, p, np);
                        m.set(p, k, np);
                        m.set(k, q, nq);
                        m.set(q, k, nq);
                    }
                }
                m.set(p, p, app - t * apq);
                m.set(q, q, aqq + t * apq);
                m.set(p, q, 0.0);
                m.set(q, p, 0.0);
                for k in 0..n {
                    let vkp = v.get(k, p);
                    let vkq = v.get(k, q);
                    v.set(k, p, c * vkp - s * vkq);
                    v.set(k, q, s * vkp + c * vkq);
                }
            }
        }
    }
    if !converged {
        return Err(FxVarError::Numerical(
            "Jacobi eigen solver did not converge in 100 sweeps".to_string(),
        ));
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&x, &y| m.get(x, x).total_cmp(&m.get(y, y)));
    let vals: Vec<f64> = idx.iter().map(|&i| m.get(i, i)).collect();
    let mut vecs = Matrix::zeros(n, n);
    for (newc, &oldc) in idx.iter().enumerate() {
        for k in 0..n {
            vecs.set(k, newc, v.get(k, oldc));
        }
    }
    Ok((vals, vecs))
}

/// `V diag(lam) V'`.
fn rebuild(vals: &[f64], vecs: &Matrix) -> Matrix {
    let n = vals.len();
    let mut out = Matrix::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for (k, l) in vals.iter().enumerate() {
                s += vecs.get(i, k) * l * vecs.get(j, k);
            }
            out.set(i, j, s);
            out.set(j, i, s);
        }
    }
    out
}

fn mean_diag(a: &Matrix) -> f64 {
    let n = a.rows();
    (0..n).map(|i| a.get(i, i)).sum::<f64>() / n as f64
}

/// Eigenvalue clipping: eigenvalues floored at `1e-10 mean(diag)`, then the
/// original diagonal restored by `D^{1/2}` scaling (variances kept).
///
/// # Errors
/// Invalid input (see [`check_symmetric`]) or a non-positive mean diagonal.
pub fn clip_psd(a: &Matrix) -> Result<Matrix> {
    check_symmetric(a)?;
    let md = mean_diag(a);
    if !(md > 0.0) {
        return Err(FxVarError::Invalid(format!(
            "mean diagonal must be > 0, got {md}"
        )));
    }
    let (vals, vecs) = symmetric_eigen(a)?;
    let floor = 1e-10 * md;
    let clipped: Vec<f64> = vals.iter().map(|l| l.max(floor)).collect();
    let mut b = rebuild(&clipped, &vecs);
    let n = a.rows();
    let d: Vec<f64> = (0..n)
        .map(|i| {
            if b.get(i, i) > 0.0 {
                (a.get(i, i).max(0.0) / b.get(i, i)).sqrt()
            } else {
                0.0
            }
        })
        .collect();
    for i in 0..n {
        for j in 0..n {
            let v = b.get(i, j) * d[i] * d[j];
            b.set(i, j, v);
        }
    }
    for (i, di) in d.iter().enumerate() {
        if *di > 0.0 {
            b.set(i, i, a.get(i, i));
        }
    }
    Ok(b)
}

/// Higham (2002) nearest correlation matrix by alternating projections with
/// Dykstra's correction, applied to the correlation of `a` and rescaled by
/// the original vols.
///
/// Iterates `R = Y - dS`, `X = P_psd(R)`, `dS = X - R`, `Y = P_unit(X)`
/// until `||Y_k - Y_{k-1}||_F <= 1e-12 ||Y_k||_F` (max 1000 iterations).
///
/// # Errors
/// Invalid input, a non-positive variance, or no convergence.
pub fn higham_nearest(a: &Matrix) -> Result<Matrix> {
    check_symmetric(a)?;
    let n = a.rows();
    let mut vol = vec![0.0; n];
    for (i, v) in vol.iter_mut().enumerate() {
        let d = a.get(i, i);
        if !(d > 0.0) {
            return Err(FxVarError::Invalid(format!(
                "Higham repair needs positive variances, a[{i},{i}] = {d}"
            )));
        }
        *v = d.sqrt();
    }
    let mut y = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            y.set(
                i,
                j,
                if i == j {
                    1.0
                } else {
                    a.get(i, j) / (vol[i] * vol[j])
                },
            );
        }
    }
    let mut ds = Matrix::zeros(n, n);
    let mut converged = false;
    for _ in 0..1000 {
        let mut r = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                r.set(i, j, y.get(i, j) - ds.get(i, j));
            }
        }
        let (vals, vecs) = symmetric_eigen(&r)?;
        let pos: Vec<f64> = vals.iter().map(|l| l.max(0.0)).collect();
        let x = rebuild(&pos, &vecs);
        let mut ynew = x.clone();
        for i in 0..n {
            for j in 0..n {
                ds.set(i, j, x.get(i, j) - r.get(i, j));
            }
            ynew.set(i, i, 1.0);
        }
        let mut diff2 = 0.0;
        let mut norm2 = 0.0;
        for i in 0..n {
            for j in 0..n {
                let d = ynew.get(i, j) - y.get(i, j);
                diff2 += d * d;
                norm2 += ynew.get(i, j) * ynew.get(i, j);
            }
        }
        y = ynew;
        if diff2.sqrt() <= 1e-12 * norm2.sqrt() {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(FxVarError::Numerical(
            "Higham nearest-correlation iteration did not converge in 1000 iterations".to_string(),
        ));
    }
    let mut out = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            out.set(i, j, y.get(i, j) * vol[i] * vol[j]);
        }
    }
    Ok(out)
}

/// Plain Cholesky, then the jitter ladder `1e-12 .. 1e-6` x mean(diag).
fn jitter_ladder(a: &Matrix) -> Option<(Matrix, f64)> {
    if let Ok(l) = a.cholesky() {
        return Some((l, 0.0));
    }
    let md = mean_diag(a);
    if !(md > 0.0) {
        return None;
    }
    let n = a.rows();
    for k in 0..=6 {
        let eps = MIN_RELATIVE_JITTER * 10f64.powi(k) * md;
        let mut b = a.clone();
        for i in 0..n {
            b.set(i, i, a.get(i, i) + eps);
        }
        if let Ok(l) = b.cholesky() {
            return Some((l, eps));
        }
    }
    None
}

/// Cholesky with jitter and optional PSD repair (see the module docs).
///
/// # Errors
/// Invalid input (non-square, empty, non-finite, asymmetric); with
/// [`PsdRepair::None`], a numerical error naming the most negative
/// eigenvalue when the jitter cap is not enough.
///
/// # Examples
///
/// ```
/// # use fx_var_engine::matrix::Matrix;
/// # use fx_var_engine::psd::{cholesky_repair, PsdRepair};
/// let a = Matrix::from_rows(&[vec![1.0, 0.9, 0.9], vec![0.9, 1.0, -0.9], vec![0.9, -0.9, 1.0]]).unwrap();
/// assert!(cholesky_repair(&a, PsdRepair::None).is_err());
/// let f = cholesky_repair(&a, PsdRepair::Higham).unwrap();
/// assert!(f.repaired.is_some());
/// ```
pub fn cholesky_repair(a: &Matrix, repair: PsdRepair) -> Result<CholeskyFactor> {
    check_symmetric(a)?;
    let mut warnings = Vec::new();
    if let Some((factor, jitter)) = jitter_ladder(a) {
        if jitter > 0.0 {
            warnings.push(format!(
                "matrix was not positive definite; added diagonal jitter {jitter:e} \
                 ({:e} x mean(diag))",
                jitter / mean_diag(a)
            ));
        }
        return Ok(CholeskyFactor {
            factor,
            jitter,
            repaired: None,
            warnings,
        });
    }
    let (vals, _) = symmetric_eigen(a)?;
    let lam_min = vals.first().copied().unwrap_or(f64::NAN);
    let repaired = match repair {
        PsdRepair::None => {
            return Err(FxVarError::Numerical(format!(
                "matrix is materially indefinite: most negative eigenvalue {lam_min:e}; \
                 a diagonal jitter up to {MAX_RELATIVE_JITTER:e} x mean(diag) did not \
                 make it factorisable. Fix the covariance or pass PsdRepair::Clip / Higham."
            )));
        }
        PsdRepair::Clip => clip_psd(a)?,
        PsdRepair::Higham => higham_nearest(a)?,
    };
    warnings.push(format!(
        "matrix repaired with {repair:?} (most negative eigenvalue was {lam_min:e})"
    ));
    let (factor, jitter) = jitter_ladder(&repaired).ok_or_else(|| {
        FxVarError::Numerical(
            "repaired matrix still not factorisable within the jitter cap".to_string(),
        )
    })?;
    if jitter > 0.0 {
        warnings.push(format!("repaired matrix needed diagonal jitter {jitter:e}"));
    }
    Ok(CholeskyFactor {
        factor,
        jitter,
        repaired: Some(repaired),
        warnings,
    })
}
