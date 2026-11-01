//! Expected Shortfall (ES / CVaR) and the shared tail-quantile machinery.
//!
//! Conventions (shared by every engine in the repository, spec S1/S2):
//!
//! * P&L arrays are profit (+) / loss (-); VaR and ES are reported as
//!   positive loss amounts in the book's base currency.
//! * `alpha` is the **tail probability** (`0.01` = 99 % VaR). It used to be
//!   a confidence level in this crate; `0.99` is now an error.
//! * Empirical VaR is the order statistic `L(m)` of the losses sorted
//!   descending, `m = ceil(n alpha - 1e-9)`; with weights, the loss at
//!   which the cumulative weight first reaches `alpha - 1e-12`
//!   ([`crate::tail`]).
//! * Empirical ES is the Acerbi-Tasche estimator with the exact fractional
//!   weight on the boundary atom; `ES >= VaR` holds for every sample.
//! * Closed forms: `Normal(mu, sigma)` P&L has `VaR = -mu + sigma z` and
//!   `ES = -mu + sigma phi(z)/alpha`, `z = Phi^{-1}(1 - alpha)`. Student-t
//!   uses the *standardised* (unit-variance) t so sigma is always the true
//!   P&L std.

use crate::stats::{inv_norm_cdf, norm_pdf, student_t_pdf, student_t_quantile};
use crate::stats::validate_alpha;
use crate::{FxVarError, Result};

fn var_es(pnl: &[f64], alpha: f64, weights: Option<&[f64]>) -> Result<(f64, f64)> {
    match weights {
        None => crate::tail::empirical_var_es(pnl, alpha),
        Some(w) => crate::tail::weighted_var_es(pnl, w, alpha),
    }
}

/// Empirical VaR (positive loss) at tail probability `alpha`. `weights`
/// (if given) are non-negative scenario weights, normalised internally;
/// `None` means uniform.
///
/// # Errors
/// [`FxVarError::Invalid`] for empty/non-finite `pnl`, bad `alpha`, or bad
/// `weights`.
pub fn empirical_var(pnl: &[f64], alpha: f64, weights: Option<&[f64]>) -> Result<f64> {
    match weights {
        None => crate::tail::empirical_var(pnl, alpha),
        Some(w) => Ok(crate::tail::weighted_var_es(pnl, w, alpha)?.0),
    }
}

/// Empirical ES (positive loss): coherent Acerbi-Tasche estimator.
///
/// # Errors
/// Same as [`empirical_var`].
pub fn empirical_es(pnl: &[f64], alpha: f64, weights: Option<&[f64]>) -> Result<f64> {
    Ok(var_es(pnl, alpha, weights)?.1)
}

/// `(VaR, ES)` from one pass over the sample.
///
/// # Errors
/// Same as [`empirical_var`].
pub fn empirical_var_es(pnl: &[f64], alpha: f64, weights: Option<&[f64]>) -> Result<(f64, f64)> {
    var_es(pnl, alpha, weights)
}

/// Closed-form Normal VaR: `-mean + sigma * z`, `z = Phi^{-1}(1 - alpha)`.
///
/// # Errors
/// [`FxVarError::Invalid`] if `sigma < 0` or `alpha` is out of range.
pub fn normal_var(sigma: f64, alpha: f64, mean: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    check_sigma_mean(sigma, mean)?;
    Ok(-mean + sigma * inv_norm_cdf(1.0 - alpha)?)
}

/// Closed-form Normal ES: `-mean + sigma * phi(z) / alpha`.
///
/// # Errors
/// Same as [`normal_var`].
pub fn normal_es(sigma: f64, alpha: f64, mean: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    check_sigma_mean(sigma, mean)?;
    let z = inv_norm_cdf(1.0 - alpha)?;
    Ok(-mean + sigma * norm_pdf(z) / alpha)
}

fn t_scale(df: f64) -> Result<f64> {
    // `!(df > 2.0)` rather than `df <= 2.0`: the latter is false for NaN.
    if !(df > 2.0) || !df.is_finite() {
        return Err(FxVarError::invalid(format!(
            "Student-t df must be finite and > 2 for finite variance, got {df}"
        )));
    }
    Ok(((df - 2.0) / df).sqrt())
}

/// Shared guard for the closed-form tail formulae.
///
/// Written with `is_finite` and `!(sigma >= 0.0)` rather than
/// `sigma < 0.0`: every comparison against NaN is false, so the naive form
/// accepts a NaN sigma and returns a NaN "VaR" — a risk number that
/// breaches no limit, because `NaN <= limit` and `NaN > limit` are both
/// false.
fn check_sigma_mean(sigma: f64, mean: f64) -> Result<()> {
    if !(sigma >= 0.0) || !sigma.is_finite() {
        return Err(FxVarError::invalid(format!(
            "sigma must be finite and >= 0, got {sigma}"
        )));
    }
    if !mean.is_finite() {
        return Err(FxVarError::invalid(format!("mean must be finite, got {mean}")));
    }
    Ok(())
}

/// Standardised Student-t VaR with true P&L std `sigma` (unit-variance
/// scaling `sqrt((df-2)/df)`); `df` must be `> 2` for finite variance.
///
/// # Errors
/// [`FxVarError::Invalid`] if `sigma < 0`, `alpha` is out of range, or
/// `df <= 2`.
pub fn student_t_var(sigma: f64, alpha: f64, df: f64, mean: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    check_sigma_mean(sigma, mean)?;
    t_scale(df)?;
    let q = student_t_quantile(1.0 - alpha, df)?;
    Ok(-mean + sigma * t_scale(df)? * q)
}

/// Standardised Student-t ES:
/// `E[X | X > q] = f(q) (df + q^2) / (alpha (df-1))` for the standard t,
/// `q = t^{-1}(1 - alpha)`.
///
/// # Errors
/// Same as [`student_t_var`].
pub fn student_t_es(sigma: f64, alpha: f64, df: f64, mean: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    check_sigma_mean(sigma, mean)?;
    t_scale(df)?;
    let q = student_t_quantile(1.0 - alpha, df)?;
    let es_std = student_t_pdf(q, df)? * (df + q * q) / (alpha * (df - 1.0));
    Ok(-mean + sigma * t_scale(df)? * es_std)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_var_es_matches_hand_computation() {
        // 100 scenarios: pnl = -1, -2, ..., -100.
        let pnl: Vec<f64> = (1..=100).map(|i| -(i as f64)).collect();
        let var = empirical_var(&pnl, 0.01, None).unwrap();
        // 1% tail on 100 obs = worst 1 -> VaR = 100.
        assert!((var - 100.0).abs() < 1e-9);
        let es = empirical_es(&pnl, 0.01, None).unwrap();
        assert!(es >= var);
    }

    #[test]
    fn es_at_least_var_always() {
        let pnl = vec![-5.0, -3.0, -1.0, 0.5, 2.0, -8.0, 1.0, -2.0, 3.0, -0.5];
        let (var, es) = empirical_var_es(&pnl, 0.1, None).unwrap();
        assert!(es >= var - 1e-12);
    }

    #[test]
    fn normal_es_ge_var() {
        let sigma = 1.0e6;
        let var = normal_var(sigma, 0.01, 0.0).unwrap();
        let es = normal_es(sigma, 0.01, 0.0).unwrap();
        assert!(es > var);
    }

    #[test]
    fn student_t_fatter_tail_than_normal_at_equal_sigma() {
        let sigma = 1.0e6;
        let nvar = normal_var(sigma, 0.01, 0.0).unwrap();
        let tvar = student_t_var(sigma, 0.01, 5.0, 0.0).unwrap();
        assert!(tvar > nvar);
    }

    #[test]
    fn invalid_alpha_errors() {
        assert!(empirical_var(&[1.0, -1.0], 1.5, None).is_err());
        // A confidence level is rejected, never reinterpreted.
        assert!(empirical_var(&[1.0, -1.0], 0.99, None).is_err());
        assert!(normal_var(-1.0, 0.01, 0.0).is_err());
        assert!(student_t_var(1.0, 0.01, 2.0, 0.0).is_err());
    }

    #[test]
    fn weighted_var_matches_uniform_special_case() {
        let pnl: Vec<f64> = (1..=200).map(|i| -(i as f64)).collect();
        let w = vec![1.0; 200];
        let var_w = empirical_var(&pnl, 0.05, Some(&w)).unwrap();
        let var_u = empirical_var(&pnl, 0.05, None).unwrap();
        assert!((var_w - var_u).abs() < 1e-9);
    }

    #[test]
    fn bad_weights_error() {
        let pnl = vec![-1.0, -2.0, -3.0];
        assert!(empirical_var(&pnl, 0.1, Some(&[1.0, -1.0, 1.0])).is_err());
        assert!(empirical_var(&pnl, 0.1, Some(&[1.0, 1.0])).is_err());
    }
}
