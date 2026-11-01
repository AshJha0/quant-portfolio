//! The one tail estimator shared by every empirical VaR / ES path.
//!
//! Losses are `L = -pnl`, sorted descending `L(1) >= L(2) >= ... >= L(n)`.
//! With `k = n * alpha` and `m = ceil(k - 1e-9)` clamped to `[1, n]`:
//!
//! * `VaR = L(m)`, the inverse-ECDF order statistic (no interpolation);
//! * `ES = [ sum_{i<m} L(i) + (k - (m-1)) L(m) ] / k`, the Acerbi-Tasche
//!   estimator with the exact fractional weight on the boundary atom.
//!
//! The `1e-9` guard keeps `k` values that are integers up to rounding
//! (e.g. `250 * 0.02 = 5.000000000000001`) on the integer rank.
//!
//! Weighted samples (BRW age weights) use the same rule on the weighted
//! step CDF: weights normalised to sum 1, sorted with the losses (stable,
//! ties keep their original order), cumulative `W_j`; `m` is the first `j`
//! with `W_j >= alpha - 1e-12`, `VaR = L(m)` and
//! `ES = [ sum_{i<m} w_i L(i) + (alpha - W_{m-1}) L(m) ] / alpha`.
//!
//! The unweighted path selects `L(m)` with `select_nth_unstable_by` and sums
//! only the `m - 1` larger losses (sorted, so the summation order and hence
//! the rounding is fixed). It is `O(n + m log m)`, not a full sort.

use crate::{EqVarError, Result};

/// Message used by every `alpha` validation error.
pub const ALPHA_MSG: &str = "alpha is the tail probability (e.g. 0.01 for 99 %)";

/// Validate a tail probability: `alpha` must lie in `(0, 0.5)`.
///
/// # Errors
/// [`EqVarError::InvalidInput`] for NaN, infinities, or values outside
/// `(0, 0.5)`. The message states the convention.
pub fn validate_alpha(alpha: f64) -> Result<f64> {
    if !(alpha > 0.0 && alpha < 0.5) {
        return Err(EqVarError::InvalidInput(format!(
            "{ALPHA_MSG}; it must lie in (0, 0.5), got {alpha}"
        )));
    }
    Ok(alpha)
}

fn check_sample(pnl: &[f64]) -> Result<()> {
    if pnl.is_empty() {
        return Err(EqVarError::InvalidInput("P&L sample is empty".to_string()));
    }
    if pnl.iter().any(|v| !v.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "P&L sample contains NaN or infinite values".to_string(),
        ));
    }
    Ok(())
}

/// One-based rank `m` of the VaR order statistic for `n` scenarios.
///
/// `m = ceil(n * alpha - 1e-9)`, clamped to `[1, n]`.
///
/// # Errors
/// [`EqVarError::InvalidInput`] if `n == 0` or `alpha` is invalid.
pub fn tail_rank(n: usize, alpha: f64) -> Result<usize> {
    validate_alpha(alpha)?;
    if n == 0 {
        return Err(EqVarError::InvalidInput(
            "tail_rank: sample size must be >= 1".to_string(),
        ));
    }
    let k = n as f64 * alpha;
    let m = (k - 1e-9).ceil();
    Ok((m.max(1.0) as usize).min(n))
}

/// Empirical `(VaR, ES)` of a P&L sample (positive numbers are losses).
///
/// # Errors
/// [`EqVarError::InvalidInput`] for an empty or non-finite sample or an
/// invalid `alpha`.
///
/// # Examples
///
/// ```
/// use eq_var_engine::tail::empirical_var_es;
/// // Losses 1..=10; alpha = 0.25 -> k = 2.5, m = 3.
/// let pnl: Vec<f64> = (1..=10).map(|i| -(i as f64)).collect();
/// let (var, es) = empirical_var_es(&pnl, 0.25).unwrap();
/// assert_eq!(var, 8.0);
/// assert!((es - (10.0 + 9.0 + 0.5 * 8.0) / 2.5).abs() < 1e-12);
/// ```
pub fn empirical_var_es(pnl: &[f64], alpha: f64) -> Result<(f64, f64)> {
    validate_alpha(alpha)?;
    check_sample(pnl)?;
    let n = pnl.len();
    let m = tail_rank(n, alpha)?;
    let k = n as f64 * alpha;
    let mut losses: Vec<f64> = pnl.iter().map(|p| -p).collect();
    let (top, lm, _) = losses.select_nth_unstable_by(m - 1, |a, b| b.total_cmp(a));
    let var = *lm;
    top.sort_unstable_by(|a, b| b.total_cmp(a));
    let head: f64 = top.iter().sum();
    let es = (head + (k - (m - 1) as f64) * var) / k;
    Ok((var, es))
}

/// Empirical VaR (order statistic `L(m)`), see [`empirical_var_es`].
///
/// # Errors
/// Same as [`empirical_var_es`].
pub fn empirical_var(pnl: &[f64], alpha: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    check_sample(pnl)?;
    let m = tail_rank(pnl.len(), alpha)?;
    let mut losses: Vec<f64> = pnl.iter().map(|p| -p).collect();
    let (_, lm, _) = losses.select_nth_unstable_by(m - 1, |a, b| b.total_cmp(a));
    Ok(*lm)
}

/// Weighted empirical `(VaR, ES)` on the weighted step CDF.
///
/// # Errors
/// [`EqVarError::InvalidInput`] for an empty or non-finite sample, an
/// invalid `alpha`, a length mismatch, or weights that are negative,
/// non-finite or sum to zero.
pub fn weighted_var_es(pnl: &[f64], weights: &[f64], alpha: f64) -> Result<(f64, f64)> {
    validate_alpha(alpha)?;
    check_sample(pnl)?;
    if weights.len() != pnl.len() {
        return Err(EqVarError::InvalidInput(format!(
            "weights have {} entries, pnl has {}",
            weights.len(),
            pnl.len()
        )));
    }
    let mut total = 0.0;
    for &w in weights {
        if !(w >= 0.0) || !w.is_finite() {
            return Err(EqVarError::InvalidInput(
                "weights must be finite and non-negative".to_string(),
            ));
        }
        total += w;
    }
    if !(total > 0.0) || !total.is_finite() {
        return Err(EqVarError::InvalidInput(
            "weights must sum to a finite positive number".to_string(),
        ));
    }
    let mut order: Vec<usize> = (0..pnl.len()).collect();
    // Losses descending == P&L ascending; stable sort keeps tie order.
    order.sort_by(|&a, &b| pnl[a].total_cmp(&pnl[b]));
    let mut cum = 0.0;
    let mut head = 0.0;
    let mut m = order.len() - 1;
    for (j, &i) in order.iter().enumerate() {
        let w = weights[i] / total;
        if cum + w >= alpha - 1e-12 {
            m = j;
            break;
        }
        cum += w;
        head += w * -pnl[i];
    }
    let var = -pnl[order[m]];
    let es = (head + (alpha - cum).max(0.0) * var) / alpha;
    Ok((var, es))
}
