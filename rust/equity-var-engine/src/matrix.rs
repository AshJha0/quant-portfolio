//! Dense row-major matrices and the covariance estimators.
//!
//! [`Matrix`] is a minimal dense `Vec<f64>` matrix — exactly the linear
//! algebra a linear-portfolio VaR engine needs and nothing more:
//! matrix–vector products, a Cholesky factorization with a diagonal-jitter
//! fallback for singular covariances, and the two covariance estimators
//! used desk-side (unbiased sample covariance and RiskMetrics EWMA).
//!
//! Conventions match the Python reference (`eq_var.parametric_var`):
//! return panels are `(T, n)` — one row per day, one column per factor —
//! and EWMA is seeded with the sample covariance then iterated over every
//! row: `Sigma_{t+1} = lam * Sigma_t + (1 - lam) * r_t r_t'`.

use crate::{EqVarError, Result};

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
        Matrix { rows, cols, data: vec![0.0; rows * cols] }
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

    /// Element `(i, j)` (row, column). Panics on out-of-range indices —
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

    /// Symmetry check with the tolerance used by the Python reference:
    /// `|a_ij - a_ji| <= 1e-12 * max(1, max|a|)`.
    pub fn is_symmetric(&self) -> bool {
        if self.rows != self.cols {
            return false;
        }
        let scale = self.data.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        let tol = 1e-12 * scale;
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
    /// [`EqVarError::Numerical`] — use [`Matrix::cholesky_jitter`] for the
    /// production fallback.
    pub fn cholesky(&self) -> Result<Matrix> {
        if self.rows != self.cols {
            return Err(EqVarError::InvalidInput(format!(
                "Cholesky needs a square matrix, got {} x {}",
                self.rows, self.cols
            )));
        }
        if !self.is_symmetric() {
            return Err(EqVarError::InvalidInput(
                "Cholesky needs a symmetric matrix".to_string(),
            ));
        }
        let n = self.rows;
        let mut l = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let mut sum = self.get(i, j);
                for k in 0..j {
                    sum -= l.get(i, k) * l.get(j, k);
                }
                if i == j {
                    if sum <= 0.0 {
                        return Err(EqVarError::Numerical(format!(
                            "matrix is not positive definite (pivot {sum:e} at row {i})"
                        )));
                    }
                    l.set(i, j, sum.sqrt());
                } else {
                    l.set(i, j, sum / l.get(j, j));
                }
            }
        }
        Ok(l)
    }

    /// Cholesky with diagonal-jitter fallback (the `safe_cholesky` of the
    /// Python reference).
    ///
    /// If plain Cholesky fails, `jitter * mean(diag)` is added to the
    /// diagonal, escalating by 10x up to `max_tries` times. The
    /// perturbation is tiny relative to the variances, so simulated moments
    /// are unchanged to within Monte Carlo noise. Defaults used by the MC
    /// module: `jitter = 1e-10`, `max_tries = 12`.
    pub fn cholesky_jitter(&self, jitter: f64, max_tries: usize) -> Result<Matrix> {
        match self.cholesky() {
            Ok(l) => return Ok(l),
            Err(EqVarError::InvalidInput(msg)) => {
                return Err(EqVarError::InvalidInput(msg));
            }
            Err(EqVarError::Numerical(_)) => {}
        }
        let n = self.rows;
        let mut scale = 0.0;
        for i in 0..n {
            scale += self.get(i, i);
        }
        scale /= n.max(1) as f64;
        if scale <= 0.0 {
            scale = 1.0;
        }
        let mut eps = jitter * scale;
        for _ in 0..max_tries {
            let mut jittered = self.clone();
            for i in 0..n {
                jittered.set(i, i, self.get(i, i) + eps);
            }
            if let Ok(l) = jittered.cholesky() {
                return Ok(l);
            }
            eps *= 10.0;
        }
        Err(EqVarError::Numerical(
            "Cholesky failed even with jitter; covariance matrix is badly indefinite"
                .to_string(),
        ))
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
    let mut means = vec![0.0; n];
    for i in 0..t {
        for (j, m) in means.iter_mut().enumerate() {
            *m += returns.get(i, j);
        }
    }
    for m in &mut means {
        *m /= t as f64;
    }
    let mut cov = Matrix::zeros(n, n);
    for i in 0..t {
        let row = returns.row(i);
        for j in 0..n {
            let dj = row[j] - means[j];
            for k in j..n {
                let dk = row[k] - means[k];
                let v = cov.get(j, k) + dj * dk;
                cov.set(j, k, v);
            }
        }
    }
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

/// RiskMetrics EWMA covariance forecast for the day after the sample.
///
/// `Sigma_{t+1} = lam * Sigma_t + (1 - lam) * r_t r_t'`, seeded with the
/// sample covariance and iterated over **every** row of the panel (zero
/// mean assumed, standard at daily horizon) — identical to the Python
/// reference `eq_var.parametric_var.ewma_covariance`.
pub fn ewma_covariance(returns: &Matrix, lam: f64) -> Result<Matrix> {
    if !(lam > 0.0 && lam < 1.0) {
        return Err(EqVarError::InvalidInput(format!(
            "decay lam must be in (0, 1), got {lam}"
        )));
    }
    let mut cov = sample_covariance(returns)?;
    let n = returns.cols();
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
