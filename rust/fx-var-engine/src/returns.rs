//! Factor-return histories: validation, covariance estimators, peg screen.
//!
//! Factor conventions (mirrors Python `fx_var.common` / C++ `returns.hpp`):
//!
//! * `"FX:CCY"` columns are daily *log returns* of CCYUSD (USD price of one
//!   unit of CCY). USD itself has no FX factor.
//! * `"IR:CCY"` columns are *absolute* daily changes (decimal p.a.) of the
//!   continuously compounded ACT/365 zero rate.
//!
//! NaN policy: the engine refuses NaNs outright ([`FxVarError::Invalid`])
//! rather than silently dropping or filling — missing FX fixings must be
//! handled upstream (holiday calendars differ across time zones).

use crate::matrix::Matrix;
use crate::{FxVarError, Result};

/// Daily log-return standard deviation below which an FX factor is treated
/// as "peg-like" (~0.8% annualised — an order of magnitude below any free
/// float; HKD inside its band realises roughly this level).
pub const PEG_VOL_THRESHOLD: f64 = 5e-4;

/// Trading days per year used for annualisation.
pub const TRADING_DAYS_PER_YEAR: u32 = 252;

/// A labelled factor-return history: rows = days, columns = factors.
#[derive(Clone, Debug, PartialEq)]
pub struct ReturnsMatrix {
    /// Column names, e.g. `"FX:EUR"`, in the same order as `data`'s columns.
    pub factors: Vec<String>,
    /// `n_obs x n_factors` daily factor returns.
    pub data: Matrix,
}

impl ReturnsMatrix {
    /// Build a [`ReturnsMatrix`], checking `factors.len() == data.cols()`.
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] on a factor/column count mismatch.
    pub fn new(factors: Vec<String>, data: Matrix) -> Result<Self> {
        if factors.len() != data.cols() {
            return Err(FxVarError::invalid(
                "returns: factor labels do not match the data column count",
            ));
        }
        Ok(ReturnsMatrix { factors, data })
    }

    /// Number of observations (rows).
    pub fn n_obs(&self) -> usize {
        self.data.rows()
    }

    /// Number of factors (columns).
    pub fn n_factors(&self) -> usize {
        self.factors.len()
    }

    /// Column index of `factor`, or `None` if absent.
    pub fn column_index(&self, factor: &str) -> Option<usize> {
        self.factors.iter().position(|f| f == factor)
    }

    /// Restrict to (and reorder as) `wanted`.
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] listing any missing factor columns.
    pub fn select(&self, wanted: &[String]) -> Result<ReturnsMatrix> {
        let mut idx = Vec::with_capacity(wanted.len());
        let mut missing = Vec::new();
        for f in wanted {
            match self.column_index(f) {
                Some(j) => idx.push(j),
                None => missing.push(f.clone()),
            }
        }
        if !missing.is_empty() {
            return Err(FxVarError::invalid(format!(
                "returns is missing required factor columns: {missing:?}"
            )));
        }
        let n = self.n_obs();
        let mut data = Matrix::zeros(n, wanted.len());
        for i in 0..n {
            for (j, &src) in idx.iter().enumerate() {
                data.set(i, j, self.data.get(i, src));
            }
        }
        Ok(ReturnsMatrix { factors: wanted.to_vec(), data })
    }
}

/// A labelled covariance matrix over named factors.
#[derive(Clone, Debug, PartialEq)]
pub struct FactorCov {
    /// Factor names, aligned with `cov`'s rows/columns.
    pub factors: Vec<String>,
    /// `n_factors x n_factors` daily covariance.
    pub cov: Matrix,
}

impl FactorCov {
    /// Restrict to (and reorder as) `wanted`.
    ///
    /// # Errors
    /// [`FxVarError::Invalid`] listing any missing factors.
    pub fn select(&self, wanted: &[String]) -> Result<FactorCov> {
        let mut idx = Vec::with_capacity(wanted.len());
        let mut missing = Vec::new();
        for f in wanted {
            match self.factors.iter().position(|x| x == f) {
                Some(j) => idx.push(j),
                None => missing.push(f.clone()),
            }
        }
        if !missing.is_empty() {
            return Err(FxVarError::invalid(format!(
                "cov is missing required factor columns: {missing:?}"
            )));
        }
        let n = wanted.len();
        let mut cov = Matrix::zeros(n, n);
        for (i, &ri) in idx.iter().enumerate() {
            for (j, &rj) in idx.iter().enumerate() {
                cov.set(i, j, self.cov.get(ri, rj));
            }
        }
        Ok(FactorCov { factors: wanted.to_vec(), cov })
    }
}

/// Validate a history for use by any VaR method: consistent shape, at
/// least `min_obs` rows, every `required` factor present, no NaNs.
///
/// # Errors
/// [`FxVarError::Invalid`] with an informative message otherwise.
pub fn validate_returns(
    returns: &ReturnsMatrix,
    required: &[String],
    min_obs: usize,
) -> Result<()> {
    if returns.data.cols() != returns.factors.len() {
        return Err(FxVarError::invalid(
            "returns: factor labels do not match the data column count",
        ));
    }
    if returns.n_obs() < min_obs {
        return Err(FxVarError::invalid(format!(
            "insufficient history: {} rows < min_obs={min_obs}; VaR quantiles are \
             meaningless on this sample",
            returns.n_obs()
        )));
    }
    let missing: Vec<&String> =
        required.iter().filter(|f| returns.column_index(f).is_none()).collect();
    if !missing.is_empty() {
        return Err(FxVarError::invalid(format!(
            "returns is missing required factor columns: {missing:?}"
        )));
    }
    // `is_finite`, not `!is_nan`: an infinite return is just as corrosive
    // as a NaN (it produces an infinite covariance and an infinite VaR, and
    // `inf - inf` in the demeaning step is a NaN), and an `is_nan`-only
    // screen waves it straight through.
    if returns.data.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(FxVarError::invalid(
            "returns contains NaN or infinite values; clean or drop them explicitly \
             before calling the engine (NaN policy: refuse, never impute silently)",
        ));
    }
    Ok(())
}

/// Unbiased (ddof=1) sample covariance of the daily factor returns.
///
/// # Errors
/// [`FxVarError::Invalid`] if fewer than 2 observations are given.
pub fn sample_cov(returns: &ReturnsMatrix) -> Result<FactorCov> {
    let n = returns.n_obs();
    let k = returns.n_factors();
    if n < 2 {
        return Err(FxVarError::invalid("sample_cov: need at least 2 observations"));
    }
    if !returns.data.all_finite() {
        return Err(FxVarError::invalid(
            "sample_cov: returns contain NaN or infinite values; a single bad \
             observation would otherwise turn the whole covariance - and every VaR \
             derived from it - into a silent NaN",
        ));
    }
    let mut mean = vec![0.0; k];
    for i in 0..n {
        for j in 0..k {
            mean[j] += returns.data.get(i, j);
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    let mut cov = Matrix::zeros(k, k);
    for i in 0..n {
        for a in 0..k {
            let da = returns.data.get(i, a) - mean[a];
            for b in a..k {
                let db = returns.data.get(i, b) - mean[b];
                cov.set(a, b, cov.get(a, b) + da * db);
            }
        }
    }
    let denom = n as f64 - 1.0;
    for a in 0..k {
        for b in a..k {
            let v = cov.get(a, b) / denom;
            cov.set(a, b, v);
            cov.set(b, a, v);
        }
    }
    Ok(FactorCov { factors: returns.factors.clone(), cov })
}

/// RiskMetrics EWMA covariance forecast after the last observation:
/// `S_t = lam * S_{t-1} + (1 - lam) * r_{t-1} r_{t-1}'`, seeded with the
/// sample covariance.
///
/// # Errors
/// [`FxVarError::Invalid`] unless `0 < lam < 1`, or on the same conditions
/// as [`sample_cov`].
pub fn ewma_cov(returns: &ReturnsMatrix, lam: f64) -> Result<FactorCov> {
    if !(lam > 0.0 && lam < 1.0) {
        return Err(FxVarError::invalid(format!("lambda must be in (0, 1), got {lam}")));
    }
    let mut out = sample_cov(returns)?; // also rejects non-finite input
    let n = returns.n_obs();
    let k = returns.n_factors();
    for i in 0..n {
        let r = returns.data.row(i);
        for a in 0..k {
            for b in 0..k {
                let v = lam * out.cov.get(a, b) + (1.0 - lam) * r[a] * r[b];
                out.cov.set(a, b, v);
            }
        }
    }
    Ok(out)
}

/// RiskMetrics EWMA per-factor volatility.
#[derive(Clone, Debug, PartialEq)]
pub struct EwmaVolatility {
    /// Per-day forecast vols (`n_obs` rows): `sigma[t]` is the forecast for
    /// day `t` made with information up to `t - 1` (no look-ahead).
    pub sigma: Matrix,
    /// One-step-ahead forecast per factor, for the day after the last
    /// observation.
    pub sigma_next: Vec<f64>,
}

/// Compute [`EwmaVolatility`]: `sigma2[t] = lam * sigma2[t-1] + (1 - lam) *
/// r[t-1]^2`, seeded with the full-sample ddof=1 variance (floored at
/// `1e-18`).
///
/// # Errors
/// [`FxVarError::Invalid`] unless `0 < lam < 1`, or fewer than 2
/// observations are given.
pub fn ewma_volatility(returns: &ReturnsMatrix, lam: f64) -> Result<EwmaVolatility> {
    if !(lam > 0.0 && lam < 1.0) {
        return Err(FxVarError::invalid(format!("lambda must be in (0, 1), got {lam}")));
    }
    let n = returns.n_obs();
    let k = returns.n_factors();
    if n < 2 {
        return Err(FxVarError::invalid("ewma_volatility: need at least 2 observations"));
    }
    if !returns.data.all_finite() {
        return Err(FxVarError::invalid(
            "ewma_volatility: returns contain NaN or infinite values",
        ));
    }
    let mut mean = vec![0.0; k];
    for i in 0..n {
        for j in 0..k {
            mean[j] += returns.data.get(i, j);
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    let mut sig2 = vec![0.0; k];
    for i in 0..n {
        for j in 0..k {
            let d = returns.data.get(i, j) - mean[j];
            sig2[j] += d * d;
        }
    }
    for s in sig2.iter_mut() {
        *s = (*s / (n as f64 - 1.0)).max(1e-18);
    }

    let mut sigma = Matrix::zeros(n, k);
    for i in 0..n {
        for j in 0..k {
            sigma.set(i, j, sig2[j].sqrt());
            let r = returns.data.get(i, j);
            sig2[j] = lam * sig2[j] + (1.0 - lam) * r * r;
        }
    }
    let sigma_next: Vec<f64> = sig2.iter().map(|v| v.sqrt()).collect();
    Ok(EwmaVolatility { sigma, sigma_next })
}

/// Screen `FX:*` factors for near-zero realised vol (pegged/managed ccys).
///
/// Returns the flagged factor names (empty if none). Only FX factors are
/// screened — rate factors are legitimately quiet. Historical and
/// parametric VaR are blind to peg-break risk; the caller must surface the
/// returned list as a warning and add the peg-break stress add-on
/// ([`crate::stress::peg_break_scenario`]).
pub fn flag_peg_factors(returns: &ReturnsMatrix, threshold: f64) -> Vec<String> {
    let mut flagged = Vec::new();
    let n = returns.n_obs();
    if n < 2 {
        return flagged;
    }
    for j in 0..returns.n_factors() {
        let f = &returns.factors[j];
        if !f.starts_with("FX:") {
            continue;
        }
        let mut mean = 0.0;
        for i in 0..n {
            mean += returns.data.get(i, j);
        }
        mean /= n as f64;
        let mut ss = 0.0;
        for i in 0..n {
            let d = returns.data.get(i, j) - mean;
            ss += d * d;
        }
        let sd = (ss / (n as f64 - 1.0)).sqrt();
        if sd < threshold {
            flagged.push(f.clone());
        }
    }
    flagged
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_returns() -> ReturnsMatrix {
        let mut data = Matrix::zeros(5, 2);
        let vals = [
            [0.01, -0.02],
            [-0.005, 0.015],
            [0.02, -0.01],
            [-0.015, 0.005],
            [0.008, -0.003],
        ];
        for (i, row) in vals.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                data.set(i, j, *v);
            }
        }
        ReturnsMatrix::new(vec!["FX:EUR".to_string(), "FX:JPY".to_string()], data).unwrap()
    }

    #[test]
    fn select_reorders_and_restricts() {
        let r = small_returns();
        let sel = r.select(&["FX:JPY".to_string()]).unwrap();
        assert_eq!(sel.n_factors(), 1);
        for i in 0..5 {
            assert_eq!(sel.data.get(i, 0), r.data.get(i, 1));
        }
    }

    #[test]
    fn select_missing_factor_errors() {
        let r = small_returns();
        assert!(r.select(&["FX:GBP".to_string()]).is_err());
    }

    #[test]
    fn sample_cov_symmetric_psd() {
        let r = small_returns();
        let cov = sample_cov(&r).unwrap();
        assert!(cov.cov.is_symmetric(1e-12));
        assert!(cov.cov.get(0, 0) > 0.0);
        assert!(cov.cov.get(1, 1) > 0.0);
    }

    #[test]
    fn ewma_cov_rejects_bad_lambda() {
        let r = small_returns();
        assert!(ewma_cov(&r, 1.5).is_err());
        assert!(ewma_cov(&r, 0.0).is_err());
    }

    #[test]
    fn nan_is_refused() {
        let mut r = small_returns();
        r.data.set(0, 0, f64::NAN);
        let err = validate_returns(&r, &[], 2).unwrap_err();
        assert!(matches!(err, FxVarError::Invalid(_)));
    }

    #[test]
    fn peg_screen_flags_quiet_factor() {
        let mut data = Matrix::zeros(60, 2);
        for i in 0..60 {
            data.set(i, 0, 0.00001 * ((i % 3) as f64 - 1.0)); // pegged
            data.set(i, 1, 0.02 * ((i % 5) as f64 - 2.0)); // free float
        }
        let r =
            ReturnsMatrix::new(vec!["FX:HKD".to_string(), "FX:EUR".to_string()], data).unwrap();
        let flagged = flag_peg_factors(&r, PEG_VOL_THRESHOLD);
        assert_eq!(flagged, vec!["FX:HKD".to_string()]);
    }
}
