//! Special functions and moment statistics: wrappers over `qp_core::stats`.
//!
//! The maths moved to [`qp_core::stats`] in Round 8 (plan W20), shared with
//! `fx-var-engine`; this crate's numbers did not change (bit-identical on
//! the W20 characterisation grid, `tests/test_stats_corpus.rs`). The
//! wrappers keep this crate's names and its [`EqVarError`] type.
//!
//! All functions are pure; invalid inputs return
//! [`EqVarError::InvalidInput`](crate::EqVarError::InvalidInput).

#[cfg(doc)]
use crate::EqVarError;
use crate::Result;
use qp_core::stats as core;

pub use qp_core::stats::{erf, erfc, normal_cdf, normal_pdf};

/// Natural log of the Gamma function for finite `x > 0` (Lanczos).
pub fn ln_gamma(x: f64) -> Result<f64> {
    Ok(core::ln_gamma(x)?)
}

/// Lower regularized incomplete gamma `P(a, x)`, `a > 0`, `x >= 0`.
pub fn regularized_gamma_p(a: f64, x: f64) -> Result<f64> {
    Ok(core::regularized_gamma_p(a, x)?)
}

/// Upper regularized incomplete gamma `Q(a, x) = 1 - P(a, x)`.
pub fn regularized_gamma_q(a: f64, x: f64) -> Result<f64> {
    Ok(core::regularized_gamma_q(a, x)?)
}

/// Inverse standard normal CDF `Phi^{-1}(p)`, `p` in `(0, 1)`.
pub fn normal_ppf(p: f64) -> Result<f64> {
    Ok(core::normal_ppf(p)?)
}

/// Regularized incomplete beta `I_x(a, b)`, `a, b > 0`, `x` in `[0, 1]`.
pub fn regularized_beta(a: f64, b: f64, x: f64) -> Result<f64> {
    Ok(core::regularized_beta(a, b, x)?)
}

/// Chi-squared survival function `P(X > x)` with `df` degrees of freedom.
pub fn chi2_sf(x: f64, df: f64) -> Result<f64> {
    Ok(core::chi2_sf(x, df)?)
}

/// Exact Binomial(n, p) CDF `P(X <= k)`; `k >= n` returns 1.
pub fn binomial_cdf(k: u64, n: u64, p: f64) -> Result<f64> {
    let (k, n) = (
        i64::try_from(k).unwrap_or(i64::MAX),
        i64::try_from(n).unwrap_or(i64::MAX),
    );
    Ok(core::binomial_cdf(k, n, p)?)
}

/// Student-t density with `df > 0` degrees of freedom.
pub fn student_t_pdf(x: f64, df: f64) -> Result<f64> {
    Ok(core::student_t_pdf(x, df)?)
}

/// Student-t CDF via the regularized incomplete beta.
pub fn student_t_cdf(x: f64, df: f64) -> Result<f64> {
    Ok(core::student_t_cdf(x, df)?)
}

/// Student-t quantile `F^{-1}(p)`, `p` in `(0, 1)`.
pub fn student_t_ppf(p: f64, df: f64) -> Result<f64> {
    Ok(core::student_t_ppf(p, df)?)
}

/// Arithmetic mean; errors on empty or non-finite input.
pub fn mean(x: &[f64]) -> Result<f64> {
    Ok(core::mean(x)?)
}

/// Sample standard deviation with `ddof = 1`; needs at least 2 points.
pub fn stdev(x: &[f64]) -> Result<f64> {
    Ok(core::stdev(x)?)
}

/// Population variance (`ddof = 0`), the EWMA seed; errors on empty input.
pub fn population_variance(x: &[f64]) -> Result<f64> {
    Ok(core::population_variance(x)?)
}

/// Sample skewness `m3 / m2^{3/2}` (biased); needs at least 3 points.
pub fn skewness(x: &[f64]) -> Result<f64> {
    Ok(core::skewness(x)?)
}

/// Sample excess kurtosis `m4 / m2^2 - 3` (biased); needs at least 4 points.
pub fn excess_kurtosis(x: &[f64]) -> Result<f64> {
    Ok(core::excess_kurtosis(x)?)
}
