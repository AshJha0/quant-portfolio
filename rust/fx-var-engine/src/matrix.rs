//! Dense row-major matrix with the small set of operations a VaR engine
//! needs: products, quadratic forms, and a Cholesky factorisation with
//! escalating diagonal jitter.
//!
//! The jitter path is not a numerical nicety: an FX factor covariance that
//! contains pegged currencies (near-zero-variance factors, or two
//! currencies pegged to the same anchor and hence perfectly correlated) is
//! routinely singular or numerically indefinite, and is a *legitimate*
//! input to this engine. The factorisation reports the jitter it used so
//! callers can surface a [`crate::FxVarError`]-free diagnostic.

use crate::{FxVarError, Result};

/// Largest diagonal jitter [`Matrix::cholesky_with_jitter`] will apply, as
/// a multiple of the mean diagonal (mean variance).
///
/// `1e-6` is a part per million of the average variance: enough to repair a
/// covariance that is only singular or indefinite through accumulated
/// floating-point noise (a pegged currency, two currencies pegged to the
/// same anchor, an EWMA update that lost the last bit of positivity), and
/// far too small to paper over a genuinely indefinite matrix.
pub const MAX_RELATIVE_JITTER: f64 = 1e-6;

/// Dense `rows x cols` matrix of `f64`, row-major storage.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Zero-filled `rows x cols` matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Identity matrix of dimension `n`.
    pub fn identity(n: usize) -> Self {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Build from row slices; every row must have equal length.
    pub fn from_rows(rows: &[Vec<f64>]) -> Result<Self> {
        let r = rows.len();
        let c = rows.first().map_or(0, |v| v.len());
        let mut data = Vec::with_capacity(r * c);
        for row in rows {
            if row.len() != c {
                return Err(FxVarError::invalid("matrix rows have unequal lengths"));
            }
            data.extend_from_slice(row);
        }
        Ok(Matrix { rows: r, cols: c, data })
    }

    /// Number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Element at `(i, j)` (bounds-checked in debug builds).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    /// Set element at `(i, j)`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    /// Borrow row `i` as a slice.
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Mutably borrow row `i`.
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        &mut self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Whole storage as a flat row-major slice.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// True when every entry is finite (no NaN, no infinity).
    ///
    /// Used as an explicit pre-check by the factorisations: a NaN entry
    /// passes every ordering test (`NaN <= 0.0` is false, `|NaN| > atol`
    /// is false), so without this check a NaN covariance would factorise
    /// into an all-NaN `L` and hand a silent NaN VaR to the desk.
    pub fn all_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }

    /// Largest absolute entry (0 for an empty matrix) — the natural scale
    /// for relative tolerances.
    pub fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    /// True when the matrix is square and symmetric to `atol`.
    ///
    /// A matrix containing NaN is **not** symmetric by this definition:
    /// `|NaN - NaN| > atol` is false, so a naive comparison would call it
    /// symmetric. The NaN check is explicit.
    ///
    /// `atol` is an **absolute** tolerance, which is the right thing for
    /// the caller who knows the units. Callers that receive a matrix of
    /// unknown scale (e.g. a P&L covariance for a multi-billion book,
    /// whose entries are ~1e12) should scale it — see
    /// [`Matrix::is_symmetric_rel`] — because a fixed absolute tolerance
    /// spuriously rejects a perfectly symmetric large-notional matrix
    /// whose two triangles were accumulated in different orders.
    pub fn is_symmetric(&self, atol: f64) -> bool {
        if self.rows != self.cols {
            return false;
        }
        if !self.all_finite() {
            return false;
        }
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if (self.get(i, j) - self.get(j, i)).abs() > atol {
                    return false;
                }
            }
        }
        true
    }

    /// Symmetry check with a **scale-relative** tolerance:
    /// `|a_ij - a_ji| <= rtol * max(1, max|a|)`.
    ///
    /// This is what covariance inputs of unknown magnitude want. With an
    /// absolute tolerance, a genuinely symmetric covariance expressed in
    /// currency units (entries ~1e12 for a multi-billion book) is rejected
    /// purely because its two triangles were summed in different orders and
    /// differ in the last ulp — roughly 1e-4 in absolute terms at that
    /// scale, i.e. eight orders of magnitude above a 1e-12 absolute gate.
    pub fn is_symmetric_rel(&self, rtol: f64) -> bool {
        self.is_symmetric(rtol * self.max_abs().max(1.0))
    }

    /// Matrix-vector product `A x`.
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.cols {
            return Err(FxVarError::invalid("matvec dimension mismatch"));
        }
        let mut out = vec![0.0; self.rows];
        for i in 0..self.rows {
            let row = self.row(i);
            let mut acc = 0.0;
            for (a, b) in row.iter().zip(x) {
                acc += a * b;
            }
            out[i] = acc;
        }
        Ok(out)
    }

    /// Quadratic form `w' A w` (requires a square matrix).
    pub fn quad_form(&self, w: &[f64]) -> Result<f64> {
        if self.rows != self.cols || w.len() != self.cols {
            return Err(FxVarError::invalid("quad_form dimension mismatch"));
        }
        let aw = self.matvec(w)?;
        Ok(w.iter().zip(&aw).map(|(a, b)| a * b).sum())
    }

    /// Lower-triangular Cholesky factor `L` with `L L' = A`.
    ///
    /// Fails (like `numpy.linalg.cholesky`) when the matrix is not
    /// numerically positive definite; use [`Matrix::cholesky_with_jitter`]
    /// for covariance inputs that may be singular.
    pub fn cholesky(&self) -> Result<Matrix> {
        if self.rows != self.cols {
            return Err(FxVarError::invalid("cholesky requires a square matrix"));
        }
        if !self.all_finite() {
            return Err(FxVarError::invalid(
                "cholesky requires finite entries; the matrix contains NaN or \
                 infinity (a NaN covariance factorises into a NaN factor and \
                 then into a silent NaN VaR)",
            ));
        }
        let n = self.rows;
        let mut l = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let mut s = self.get(i, j);
                for k in 0..j {
                    s -= l.get(i, k) * l.get(j, k);
                }
                if i == j {
                    // `!(s > 0.0)` rather than `s <= 0.0`: the latter is
                    // FALSE for a NaN pivot, which would then propagate
                    // through `sqrt` into an all-NaN factor.
                    if !(s > 0.0) {
                        return Err(FxVarError::numerical(
                            "matrix is not positive definite (Cholesky pivot <= 0)",
                        ));
                    }
                    l.set(i, j, s.sqrt());
                } else {
                    l.set(i, j, s / l.get(j, j));
                }
            }
        }
        Ok(l)
    }

    /// Cholesky with escalating diagonal jitter for singular covariances.
    ///
    /// On failure, jitter `1e-12 * mean(diag)` is added to the diagonal
    /// and escalated by x10 up to `max_tries` times (mirrors the Python
    /// `robust_cholesky`). Returns `(L, jitter_used)`; `jitter_used == 0`
    /// means the clean factorisation succeeded. Callers should surface a
    /// diagnostic when jitter was needed (pegged-currency covariance
    /// blocks are the expected trigger).
    ///
    /// # Materiality cap
    ///
    /// The escalation stops once the jitter would exceed
    /// [`MAX_RELATIVE_JITTER`] times the mean diagonal, and returns
    /// [`FxVarError::Numerical`] naming that cap. Without it the ladder
    /// `1e-12 * 10^k` reaches a *multiple* of the mean variance for the
    /// `max_tries` values used in this crate — at which point the factor
    /// no longer describes the caller's covariance and the simulated risk
    /// is materially inflated with no diagnostic. Singular-but-PSD
    /// covariances (pegged currencies, perfectly correlated factors) are
    /// repaired at the first rung and are unaffected.
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] for non-square/non-symmetric input;
    /// [`FxVarError::Numerical`] if factorisation still fails at maximum
    /// jitter.
    pub fn cholesky_with_jitter(&self, max_tries: usize) -> Result<(Matrix, f64)> {
        if self.rows != self.cols {
            return Err(FxVarError::invalid("cov must be a square matrix"));
        }
        if !self.all_finite() {
            return Err(FxVarError::invalid(
                "cov must have finite entries (NaN or infinity present)",
            ));
        }
        // SCALE-RELATIVE symmetry gate. The old absolute 1e-12 rejected a
        // perfectly symmetric covariance expressed in currency units — a
        // multi-billion book's P&L covariance has entries ~1e12, where the
        // last ulp is ~1e-4, eight orders of magnitude above 1e-12.
        if !self.is_symmetric_rel(1e-12) {
            return Err(FxVarError::invalid("cov must be symmetric"));
        }
        if let Ok(l) = self.cholesky() {
            return Ok((l, 0.0));
        }
        let n = self.rows;
        let mut base = 0.0;
        for i in 0..n {
            base += self.get(i, i);
        }
        base /= n.max(1) as f64;
        if base <= 0.0 {
            base = 1.0;
        }
        let cap = MAX_RELATIVE_JITTER * base;
        let mut jitter = 1e-12 * base;
        for _ in 0..max_tries {
            if jitter > cap {
                break;
            }
            let mut a = self.clone();
            for i in 0..n {
                a.set(i, i, a.get(i, i) + jitter);
            }
            if let Ok(l) = a.cholesky() {
                return Ok((l, jitter));
            }
            jitter *= 10.0;
        }
        Err(FxVarError::numerical(format!(
            "covariance matrix is not factorisable with a diagonal jitter of up to \
             {MAX_RELATIVE_JITTER:e} x mean(diag) = {cap:e}; it is materially \
             indefinite, not merely singular. A larger jitter would inflate the \
             simulated risk with no diagnostic - fix the covariance instead \
             (shrinkage, nearest-PSD projection, or dropping the offending factor)."
        )))
    }

    /// Moore-Penrose-free symmetric linear solve `A x = b` by Cholesky with
    /// jitter (adequate for the reverse-stress ellipsoid projection).
    pub fn solve_spd(&self, b: &[f64]) -> Result<Vec<f64>> {
        let (l, _) = self.cholesky_with_jitter(8)?;
        let n = self.rows;
        if b.len() != n {
            return Err(FxVarError::invalid("solve dimension mismatch"));
        }
        // forward: L y = b
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= l.get(i, k) * y[k];
            }
            y[i] = s / l.get(i, i);
        }
        // backward: L' x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= l.get(k, i) * x[k];
            }
            x[i] = s / l.get(i, i);
        }
        Ok(x)
    }
}
