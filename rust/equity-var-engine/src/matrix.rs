//! Dense row-major matrices and the covariance estimators.
//!
//! [`Matrix`] is a minimal dense `Vec<f64>` matrix: exactly the linear
//! algebra a linear-portfolio VaR engine needs and nothing more:
//! matrix–vector products, a Cholesky factorization with a diagonal-jitter
//! fallback for singular covariances, and the two covariance estimators
//! used desk-side (unbiased sample covariance and RiskMetrics EWMA).
//!
//! Conventions match the Python reference (`eq_var.parametric_var`):
//! return panels are `(T, n)`: one row per day, one column per factor:
//! and EWMA is seeded with the zero-mean second moment `mean(r r')` of the
//! first `min(30, T)` rows (convention S8) then iterated over every row: `Sigma_{t+1} = lam * Sigma_t + (1 - lam) * r_t r_t'`.

use crate::{EqVarError, Result};

pub use crate::psd::MAX_RELATIVE_JITTER;
use crate::psd::{CholeskyFactor, PsdRepair};

/// Dense row-major matrix of `f64`.
///
/// # Examples
///
/// ```
/// use eq_var_engine::matrix::Matrix;
/// let m = Matrix::from_vec(2, 2, vec![4.0, 2.0, 2.0, 3.0]).unwrap();
/// let l = m.cholesky().unwrap();
/// assert!((l.get(0, 0) - 2.0).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, PartialEq)]
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

    /// Build from a row-major buffer; errors if `data.len() != rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(EqVarError::InvalidInput(format!(
                "matrix buffer has {} entries, expected {rows} x {cols} = {}",
                data.len(),
                rows * cols
            )));
        }
        Ok(Matrix { rows, cols, data })
    }

    /// `n x n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
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

    /// Element `(i, j)` (row, column). Panics on out-of-range indices:
    /// index bugs are programmer errors, not recoverable model inputs.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    /// Set element `(i, j)`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    /// Row `i` as a contiguous slice.
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Underlying row-major buffer.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Matrix–vector product `A x`; errors on dimension mismatch.
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.cols {
            return Err(EqVarError::InvalidInput(format!(
                "matvec: vector has {} entries, matrix has {} columns",
                x.len(),
                self.cols
            )));
        }
        let mut out = vec![0.0; self.rows];
        for i in 0..self.rows {
            let row = self.row(i);
            let mut acc = 0.0;
            for (a, b) in row.iter().zip(x.iter()) {
                acc += a * b;
            }
            out[i] = acc;
        }
        Ok(out)
    }

    /// `true` if every entry is finite (no NaN, no infinity).
    ///
    /// Used as an explicit pre-check by [`Matrix::cholesky`]: a NaN entry
    /// would otherwise pass every ordering test (`NaN <= 0.0` is false,
    /// `|NaN| > tol` is false), factorise into an all-NaN `L`, and hand a
    /// silent NaN VaR to the desk.
    ///
    /// # Examples
    ///
    /// ```
    /// use eq_var_engine::matrix::Matrix;
    /// assert!(Matrix::identity(2).all_finite());
    /// assert!(!Matrix::from_vec(1, 1, vec![f64::NAN]).unwrap().all_finite());
    /// ```
    pub fn all_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }

    /// Relative symmetry check shared by every engine:
    /// `|a_ij - a_ji| <= 1e-10 * max|a|` ([`crate::psd::check_symmetric`]).
    ///
    /// A matrix containing NaN is **not** symmetric by this definition:
    /// `|NaN - NaN| > tol` is false, so a naive comparison would call it
    /// symmetric. The NaN check is explicit.
    pub fn is_symmetric(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }
        if !self.all_finite() {
            return false;
        }
        let scale = self.data.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let tol = crate::psd::SYMMETRY_RTOL * scale;
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if (self.get(i, j) - self.get(j, i)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Plain Cholesky factorization: lower-triangular `L` with `L L' = A`.
    ///
    /// Requires a symmetric **positive-definite** matrix; a non-positive
    /// pivot (indefinite or exactly singular input, e.g. a zero-variance
    /// factor or perfectly correlated factors) returns
    /// [`EqVarError::Numerical`]: use [`Matrix::cholesky_jitter`] for the
    /// production fallback.
    pub fn cholesky(&self) -> Result<Matrix> {
        if self.rows != self.cols {
            return Err(EqVarError::InvalidInput(format!(
                "Cholesky needs a square matrix, got {} x {}",
                self.rows, self.cols
            )));
        }
        if !self.all_finite() {
            return Err(EqVarError::InvalidInput(
                "Cholesky needs a matrix of finite entries; the input contains                  NaN or infinity (a NaN covariance factorises into a NaN                  factor and then into a silent NaN VaR)"
                    .to_string(),
            ));
        }
        if !self.is_symmetric() {
            return Err(EqVarError::InvalidInput(
                "Cholesky needs a symmetric matrix".to_string(),
            ));
        }
        let n = self.rows;
        let mut l = Matrix::zeros(n, n);
        // Shared kernel; fails on a pivot that is not > 0, NaN included.
        if let Err(e) = qp_core::linalg::cholesky_lower(&self.data, n, &mut l.data) {
            let (sum, i) = (e.pivot, e.row);
            return Err(EqVarError::Numerical(format!(
                "matrix is not positive definite (pivot {sum:e} at row {i})"
            )));
        }
        Ok(l)
    }

    /// Cholesky with jitter and optional PSD repair: see
    /// [`crate::psd::cholesky_repair`]. The jitter actually used is in the
    /// returned [`CholeskyFactor::jitter`].
    ///
    /// # Errors
    /// See [`crate::psd::cholesky_repair`].
    pub fn cholesky_repair(&self, repair: PsdRepair) -> Result<CholeskyFactor> {
        crate::psd::cholesky_repair(self, repair)
    }
}

/// Unbiased sample covariance (`ddof = 1`) of a `(T, n)` return panel.
///
/// Errors if the panel has fewer than 2 rows or no columns.
pub fn sample_covariance(returns: &Matrix) -> Result<Matrix> {
    let (t, n) = (returns.rows(), returns.cols());
    if t < 2 {
        return Err(EqVarError::InvalidInput(format!(
            "need at least 2 observations for a covariance, got {t}"
        )));
    }
    if n == 0 {
        return Err(EqVarError::InvalidInput(
            "return panel has no columns (assets)".to_string(),
        ));
    }
    if !returns.all_finite() {
        return Err(EqVarError::InvalidInput(
            "return panel contains NaN or infinite values; a single missing              observation would otherwise turn the whole covariance, and every              VaR derived from it, into a silent NaN"
                .to_string(),
        ));
    }
    let mut means = vec![0.0; n];
    for i in 0..t {
        for (j, m) in means.iter_mut().enumerate() {
            *m += returns.get(i, j);
        }
    }
    for m in &mut means {
        *m /= t as f64;
    }
    // Centre once, then a cache-blocked D^T D that adds the products for
    // each (j, k) in row order: the same bits as the row-streaming loop.
    let mut centred = Vec::with_capacity(t * n);
    for i in 0..t {
        centred.extend(returns.row(i).iter().zip(&means).map(|(x, m)| x - m));
    }
    let mut upper = vec![0.0; n * n];
    qp_core::gemm::syrk_upper_blocked(&centred, &mut upper, t, n);
    let mut cov = Matrix::from_vec(n, n, upper)?;
    let denom = (t - 1) as f64;
    for j in 0..n {
        for k in j..n {
            let v = cov.get(j, k) / denom;
            cov.set(j, k, v);
            cov.set(k, j, v);
        }
    }
    Ok(cov)
}

/// Number of leading rows used for the EWMA covariance seed (convention S8).
pub const EWMA_COV_SEED_WINDOW: usize = 30;

/// RiskMetrics EWMA covariance forecast for the day after the sample.
///
/// `Sigma_{t+1} = lam * Sigma_t + (1 - lam) * r_t r_t'`, seeded with the
/// zero-mean second moment `mean(r r')` of the first
/// `min(EWMA_COV_SEED_WINDOW, T)` rows (convention S8, so the seed uses no
/// later data) and iterated over **every** row of the panel (zero
/// mean assumed, standard at daily horizon): identical to the Python
/// reference `eq_var.parametric_var.ewma_covariance`.
pub fn ewma_covariance(returns: &Matrix, lam: f64) -> Result<Matrix> {
    if !(lam > 0.0 && lam < 1.0) {
        return Err(EqVarError::InvalidInput(format!(
            "decay lam must be in (0, 1), got {lam}"
        )));
    }
    sample_covariance(returns)?; // validates T >= 2 and finiteness
    let n = returns.cols();
    let m = EWMA_COV_SEED_WINDOW.min(returns.rows());
    let mut cov = Matrix::zeros(n, n);
    for j in 0..n {
        for k in 0..n {
            let mut acc = 0.0;
            for i in 0..m {
                let r = returns.row(i);
                acc += r[j] * r[k];
            }
            cov.set(j, k, acc / m as f64);
        }
    }
    for i in 0..returns.rows() {
        let r = returns.row(i);
        for j in 0..n {
            for k in 0..n {
                let v = lam * cov.get(j, k) + (1.0 - lam) * r[j] * r[k];
                cov.set(j, k, v);
            }
        }
    }
    Ok(cov)
}

/// Assemble a covariance matrix from per-asset vols and a correlation
/// matrix: `Sigma_ij = vol_i * vol_j * corr_ij`.
///
/// Errors on dimension mismatch, negative vols, or `|corr_ij| > 1`.
pub fn covariance_from_vols(vols: &[f64], corr: &Matrix) -> Result<Matrix> {
    let n = vols.len();
    if corr.rows() != n || corr.cols() != n {
        return Err(EqVarError::InvalidInput(format!(
            "correlation matrix is {} x {}, expected {n} x {n}",
            corr.rows(),
            corr.cols()
        )));
    }
    if vols.iter().any(|v| *v < 0.0 || !v.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "vols must be finite and non-negative".to_string(),
        ));
    }
    let mut cov = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let c = corr.get(i, j);
            if !c.is_finite() {
                return Err(EqVarError::InvalidInput(format!(
                    "correlation entry ({i}, {j}) = {c} is not finite"
                )));
            }
            if c.abs() > 1.0 + 1e-12 {
                return Err(EqVarError::InvalidInput(format!(
                    "correlation entry ({i}, {j}) = {c} outside [-1, 1]"
                )));
            }
            cov.set(i, j, vols[i] * vols[j] * c);
        }
    }
    Ok(cov)
}
