//! Expected Shortfall (ES / CVaR) for all three VaR families.
//!
//! `ES_alpha = -(1/alpha) * integral_0^alpha Q_u(pnl) du` — the average loss
//! in the worst `alpha` tail. ES `>=` VaR by construction, ES is coherent
//! (subadditive), and FRTB replaced 99 % VaR with 97.5 % ES as the
//! market-risk capital measure.
//!
//! Conventions: `alpha` = tail probability; ES positive for losses.

use crate::stats::{normal_pdf, normal_ppf, student_t_pdf, student_t_ppf};
use crate::{validate_alpha, EqVarError, Result};

/// Minimum P&L observations for an empirical ES estimate (a 3rd of
/// [`crate::historical::MIN_OBS`] — ES averages several order statistics, so
/// it degrades more gracefully than a single quantile on a short sample, but
/// still needs enough points that `floor(alpha * n)` is meaningful).
pub const MIN_OBS: usize = 10;

/// Empirical Expected Shortfall — exact tail integral of the step CDF.
///
/// With sorted P&L `x_(1) <= ... <= x_(n)` and `k = floor(alpha * n)`:
///
/// `ES = -(1/(alpha*n)) * [ sum_{i<=k} x_(i) + (alpha*n - k) * x_(k+1) ]`
///
/// i.e. the exact integral of the empirical quantile function over
/// `(0, alpha]`, with a fractional weight on the boundary order statistic.
/// This estimator is consistent, satisfies `ES >= VaR` (same order-statistic
/// quantile) and is exact on hand-computable arrays (unit tested). Requires
/// at least [`MIN_OBS`] finite observations.
///
/// # Examples
///
/// ```
/// use eq_var_engine::expected_shortfall::expected_shortfall;
/// // Sorted: -10 -8 -6 -4 -2 1 2 3 4 5 (n = 10).
/// let pnl = [1.0, -10.0, 2.0, -8.0, 3.0, -6.0, 4.0, -4.0, 5.0, -2.0];
/// // alpha = 0.2: an = 2 exactly -> mean of the two worst losses.
/// assert!((expected_shortfall(&pnl, 0.2).unwrap() - 9.0).abs() < 1e-12);
/// ```
pub fn expected_shortfall(pnl: &[f64], alpha: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    if pnl.len() < MIN_OBS {
        return Err(EqVarError::InvalidInput(format!(
            "need at least {MIN_OBS} P&L observations for empirical ES, got {}",
            pnl.len()
        )));
    }
    if pnl.iter().any(|v| !v.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "pnl contains NaN or infinite values".to_string(),
        ));
    }
    let mut sorted = pnl.to_vec();
    // `total_cmp` is a total order on every f64, so this sort has no panic
    // path even though the finiteness check above already rules NaN out.
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let an = alpha * n as f64;
    let k = an.floor() as usize;
    let tail_sum: f64 = sorted[..k.min(n)].iter().sum();
    let frac = an - k as f64;
    let tail_sum = if frac > 0.0 && k < n {
        tail_sum + frac * sorted[k]
    } else {
        tail_sum
    };
    Ok(-tail_sum / an)
}

/// Closed-form ES for normal P&L: `ES = sigma * phi(z_alpha) / alpha - mean`.
///
/// The identity `E[-X | X <= Q_alpha] = sigma * phi(z) / alpha - mu` with
/// `z = Phi^{-1}(alpha)` is unit-tested against numerical (Simpson)
/// quadrature of the tail integral to 1e-10.
pub fn normal_es(sigma: f64, alpha: f64, mean: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    // `!(sigma >= 0.0)` rather than `sigma < 0.0`: the latter is false for
    // NaN, which would return a NaN Expected Shortfall — a risk number
    // that breaches no limit because every comparison against it is false.
    if !(sigma >= 0.0) || !sigma.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "sigma must be finite and >= 0, got {sigma}"
        )));
    }
    if !mean.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "mean must be finite, got {mean}"
        )));
    }
    let z = normal_ppf(alpha)?;
    Ok(sigma * normal_pdf(z) / alpha - mean)
}

/// Closed-form ES for variance-matched Student-t P&L (`df > 2`).
///
/// For standardised t with `df = nu` (unit variance after scaling by
/// `sqrt((nu-2)/nu)`):
///
/// `ES_std = f_nu(q) * (nu + q^2) / ((nu-1) * alpha) * sqrt((nu-2)/nu)`
///
/// with `q = t_nu^{-1}(alpha)` and `f_nu` the t density.
pub fn student_t_es(sigma: f64, alpha: f64, df: f64, mean: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    if !(sigma >= 0.0) || !sigma.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "sigma must be finite and >= 0, got {sigma}"
        )));
    }
    if !mean.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "mean must be finite, got {mean}"
        )));
    }
    if !(df > 2.0) || !df.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "Student-t df must be finite and > 2 for finite variance, got {df}"
        )));
    }
    let q = student_t_ppf(alpha, df)?;
    let es_std = student_t_pdf(q, df)? * (df + q * q) / ((df - 1.0) * alpha);
    Ok(sigma * es_std * ((df - 2.0) / df).sqrt() - mean)
}

/// Variance–covariance ES at a 1-day horizon with zero mean — the common
/// case. See [`parametric_es_full`] for a drift and multi-day horizon.
///
/// # Examples
///
/// ```
/// use eq_var_engine::matrix::Matrix;
/// use eq_var_engine::expected_shortfall::parametric_es;
/// use eq_var_engine::TailModel;
/// let cov = Matrix::from_vec(1, 1, vec![0.0001]).unwrap();
/// let es = parametric_es(&[1.0e6], &cov, 0.025, TailModel::Normal).unwrap();
/// assert!(es > 0.0);
/// ```
pub fn parametric_es(
    exposures: &[f64],
    cov: &crate::matrix::Matrix,
    alpha: f64,
    tail: crate::TailModel,
) -> Result<f64> {
    parametric_es_full(exposures, cov, alpha, tail, 0.0, 1)
}

/// Variance–covariance ES from dollar exposures and factor covariance, with
/// expected daily P&L `mean` and a `horizon_days` horizon.
///
/// Mirrors [`crate::parametric::parametric_var_full`]'s sigma/horizon
/// scaling: `sigma_h = sigma_p(w, cov) * sqrt(horizon_days)`,
/// `mu_h = mean * horizon_days`, then [`normal_es`] or [`student_t_es`].
pub fn parametric_es_full(
    exposures: &[f64],
    cov: &crate::matrix::Matrix,
    alpha: f64,
    tail: crate::TailModel,
    mean: f64,
    horizon_days: u32,
) -> Result<f64> {
    if !mean.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "expected daily P&L must be finite, got {mean}"
        )));
    }
    if horizon_days < 1 {
        return Err(EqVarError::InvalidInput(format!(
            "horizon_days must be >= 1, got {horizon_days}"
        )));
    }
    let sigma = crate::parametric::portfolio_sigma(exposures, cov)? * (horizon_days as f64).sqrt();
    let mu = mean * horizon_days as f64;
    match tail {
        crate::TailModel::Normal => normal_es(sigma, alpha, mu),
        crate::TailModel::StudentT { df } => student_t_es(sigma, alpha, df, mu),
    }
}
