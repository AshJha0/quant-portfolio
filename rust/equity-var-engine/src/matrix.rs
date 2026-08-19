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

/// Largest diagonal jitter [`Matrix::cholesky_jitter`] will apply, as a
/// multiple of the mean diagonal (mean variance).
///
/// `1e-6` is roughly a part per million of the average variance: enough to
/// repair a covariance that is only singular or indefinite by accumulated
/// floating-point noise (a riskless leg, perfectly correlated factors, an
/// EWMA update that lost the last bit of positivity), and far too small to
/// paper over a genuinely indefinite matrix.
pub const MAX_RELATIVE_JITTER: f64 = 1e-6;

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

    /// Symmetry check with the tolerance used by the Python reference:
    /// `|a_ij - a_ji| <= 1e-12 * max(1, max|a|)`.
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
        for i in 0..n {
            for j in 0..=i {
                let mut sum = self.get(i, j);
                for k in 0..j {
                    sum -= l.get(i, k) * l.get(j, k);
                }
                if i == j {
                    // `!(sum > 0.0)` rather than `sum <= 0.0`: the latter is
                    // FALSE for a NaN pivot, which would then propagate
                    // through `sqrt` into an all-NaN factor.
                    if !(sum > 0.0) {
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
    ///
    /// # Materiality cap
    ///
    /// The escalation stops once the added diagonal would exceed
    /// [`MAX_RELATIVE_JITTER`] times the mean variance, and a
    /// [`EqVarError::Numerical`] naming the required jitter is returned
    /// instead. Without that cap the ladder `1e-10 * 10^k`, `k < 12`,
    /// reaches ten *times* the mean variance — at which point the factor
    /// being returned is no longer a factor of the caller's covariance in
    /// any useful sense, and the simulated risk would be materially
    /// inflated with no diagnostic. A matrix that needs more than a
    /// rounding-noise repair is a data problem (a stale correlation
    /// block, a mis-signed factor loading, a shrinkage step that was
    /// skipped) and must be surfaced, not patched.
    ///
    /// PSD-but-singular covariances — a riskless leg, two perfectly
    /// correlated factors — are repaired at the very first rung
    /// (`1e-10 * mean(diag)`) and are unaffected.
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
        if !(jitter > 0.0) || !jitter.is_finite() {
            return Err(EqVarError::InvalidInput(format!(
                "cholesky_jitter: jitter must be finite and positive, got {jitter}"
            )));
        }
        let cap = MAX_RELATIVE_JITTER * scale;
        let mut eps = jitter * scale;
        for _ in 0..max_tries {
            if eps > cap {
                break;
            }
            let mut jittered = self.clone();
            for i in 0..n {
                jittered.set(i, i, self.get(i, i) + eps);
            }
            if let Ok(l) = jittered.cholesky() {
                return Ok(l);
            }
            eps *= 10.0;
        }
        Err(EqVarError::Numerical(format!(
            "Cholesky failed with a diagonal jitter of up to {MAX_RELATIVE_JITTER:e} x              mean(diag) = {cap:e}; the covariance matrix is materially indefinite, not              merely singular. Repairing it with a larger jitter would inflate the              simulated risk without any diagnostic — fix the covariance instead              (shrinkage, nearest-PSD projection, or removing the offending factor)."
        )))
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
            "return panel contains NaN or infinite values; a single missing              observation would otherwise turn the whole covariance — and every              VaR derived from it — into a silent NaN"
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
