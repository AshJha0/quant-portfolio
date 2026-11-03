//! Special functions and sample moments: wrappers over `qp_core::stats`.
//!
//! The maths moved to [`qp_core::stats`] in Round 8 (plan W20), shared with
//! `eq-var-engine`. The wrappers keep this crate's names and its
//! [`FxVarError`] type. One numerical change came with the move: the old
//! normal CDF here was `(1 - P(1/2, x^2/2)) / 2`, which cancels in the
//! lower tail (`Phi(-10)` came out as 0, `Phi(-8)` 1.8% low), and that
//! error reached [`inv_norm_cdf`] (4.4e-7 relative at `p = 1e-12`) and the
//! Student-t quantile (1.1e-11 relative at `p = 1e-6`). The shared code
//! computes the lower tail directly. At the VaR levels the engine is used
//! for (`alpha` from 0.001 to 0.05) values moved by at most 4.8e-14
//! relative; see `tests/test_stats_corpus.rs`.
//!
//! Conventions: quantile levels are probabilities in (0, 1); Student-t
//! quantiles are for the *standard* t (unit scale), and the estimators
//! rescale by `sqrt((df-2)/df)` where a variance-matched t is needed.

use crate::{FxVarError, Result};
use qp_core::stats as core;

pub use qp_core::stats::{Moments, normal_cdf as norm_cdf, normal_pdf as norm_pdf};

/// Inverse standard normal CDF `Phi^{-1}(p)`.
///
/// # Errors
/// [`FxVarError::Invalid`] unless `p` lies in `(0, 1)`.
pub fn inv_norm_cdf(p: f64) -> Result<f64> {
    Ok(core::normal_ppf(p)?)
}

/// [`inv_norm_cdf`] for `p` already known to be in `(0, 1)`.
pub(crate) fn inv_norm_cdf_unchecked(p: f64) -> f64 {
    core::normal_ppf_unchecked(p)
}

/// Natural log of the Gamma function (Lanczos) for finite `x > 0`.
pub fn ln_gamma(x: f64) -> Result<f64> {
    Ok(core::ln_gamma(x)?)
}

/// Regularised lower incomplete gamma `P(a, x)`.
pub fn reg_lower_gamma(a: f64, x: f64) -> Result<f64> {
    Ok(core::regularized_gamma_p(a, x)?)
}

/// Regularised upper incomplete gamma `Q(a, x) = 1 - P(a, x)`.
pub fn reg_upper_gamma(a: f64, x: f64) -> Result<f64> {
    Ok(core::regularized_gamma_q(a, x)?)
}

/// Chi-square survival function `P(X > x)` with `df` degrees of freedom.
pub fn chi2_sf(x: f64, df: f64) -> Result<f64> {
    Ok(core::chi2_sf(x, df)?)
}

/// Regularised incomplete beta `I_x(a, b)`.
pub fn reg_inc_beta(a: f64, b: f64, x: f64) -> Result<f64> {
    Ok(core::regularized_beta(a, b, x)?)
}

/// Standard Student-t probability density with `df` degrees of freedom.
pub fn student_t_pdf(x: f64, df: f64) -> Result<f64> {
    Ok(core::student_t_pdf(x, df)?)
}

/// Standard Student-t CDF.
pub fn student_t_cdf(x: f64, df: f64) -> Result<f64> {
    Ok(core::student_t_cdf(x, df)?)
}

/// Standard Student-t quantile (inverse CDF) with `df > 0`.
///
/// # Errors
/// [`FxVarError::Invalid`] if `p` is outside (0, 1) or `df <= 0`.
pub fn student_t_quantile(p: f64, df: f64) -> Result<f64> {
    Ok(core::student_t_ppf(p, df)?)
}

/// Exact binomial CDF `P(X <= k)` for `X ~ Binomial(n, p)`.
pub fn binom_cdf(k: i64, n: i64, p: f64) -> Result<f64> {
    Ok(core::binomial_cdf(k, n, p)?)
}

/// Compute [`Moments`] of `x` (at least 2 finite observations).
pub fn moments(x: &[f64]) -> Result<Moments> {
    Ok(core::moments(x)?)
}

/// Validate a VaR/ES tail probability, returning it unchanged.
///
/// `alpha` is the **tail probability** (`0.01` = 99 % VaR), the convention
/// of every engine in the repository. It used to be a confidence level in
/// this crate; `0.99` is now an error, never silently reinterpreted.
///
/// # Errors
/// [`FxVarError::Invalid`] unless `alpha` is strictly inside `(0, 0.5)`.
pub fn validate_alpha(alpha: f64) -> Result<f64> {
    crate::tail::validate_alpha(alpha)
}

/// Validate a VaR horizon in trading days, returning it unchanged.
///
/// # Errors
/// [`FxVarError::Invalid`] unless `horizon_days` is finite and `> 0`
/// (`+inf` has to be excluded explicitly: `!(inf > 0.0)` is false).
pub fn validate_horizon(horizon_days: f64) -> Result<f64> {
    if !(horizon_days > 0.0) || !horizon_days.is_finite() {
        return Err(FxVarError::invalid(format!(
            "horizon_days must be > 0, got {horizon_days}"
        )));
    }
    Ok(horizon_days)
}

/// Unbiased (ddof=1) standard deviation of a slice.
///
/// # Errors
/// [`FxVarError::Invalid`] for fewer than 2 points or non-finite values.
pub fn sample_std(x: &[f64]) -> Result<f64> {
    Ok(core::stdev(x)?)
}
