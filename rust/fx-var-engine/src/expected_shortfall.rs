//! Expected Shortfall (ES / CVaR) and the shared tail-quantile machinery.
//!
//! Conventions (mirrors Python `fx_var.expected_shortfall` / C++
//! `fxvar::expected_shortfall`):
//!
//! * P&L arrays are profit (+) / loss (-); VaR and ES are reported as
//!   positive loss amounts in the book's base currency.
//! * Empirical VaR at confidence level `alpha` on `n` scenarios with
//!   weights `w` (uniform by default) is the order-statistic / inverse-ECDF
//!   quantile: sort losses descending, accumulate weights, VaR = the loss
//!   at which the cumulative tail weight first reaches `1 - alpha`. With
//!   uniform weights this is the `m`-th worst loss, `m = ceil(n (1 -
//!   alpha))`.
//! * Empirical ES uses the Acerbi-Tasche tail-splitting estimator: the
//!   worst losses averaged over exactly `1 - alpha` of probability mass,
//!   taking only a fractional share of the atom at the VaR level. This is
//!   the coherent (subadditive) estimator; `ES >= VaR` holds for every
//!   sample.
//! * Closed forms: `Normal(mu, sigma)` P&L has `VaR = -mu + sigma z_a` and
//!   `ES = -mu + sigma phi(z_a)/(1-a)`. Student-t uses the *standardised*
//!   (unit-variance) t so sigma is always the true P&L std — normal and t
//!   figures are comparable at equal risk.

use crate::stats::{inv_norm_cdf, norm_pdf, student_t_pdf, student_t_quantile};
use crate::stats::validate_alpha;
use crate::{FxVarError, Result};

const WEIGHT_TOL: f64 = 1e-12;

struct Tail {
    losses: Vec<f64>,  // descending
    weights: Vec<f64>, // aligned, normalised
    idx: usize,        // index of the VaR atom
}

fn make_tail(pnl: &[f64], alpha: f64, weights: Option<&[f64]>) -> Result<Tail> {
    validate_alpha(alpha)?;
    if pnl.is_empty() {
        return Err(FxVarError::invalid("pnl sample is empty"));
    }
    if pnl.iter().any(|v| v.is_nan()) {
        return Err(FxVarError::invalid("pnl sample contains NaNs (NaN policy: refuse)"));
    }
    let n = pnl.len();
    let w: Vec<f64> = match weights {
        None => vec![1.0 / n as f64; n],
        Some(ws) => {
            if ws.len() != n {
                return Err(FxVarError::invalid("weights must match pnl length"));
            }
            let mut sum = 0.0;
            for &x in ws {
                if x < 0.0 {
                    return Err(FxVarError::invalid("weights must be non-negative and sum > 0"));
                }
                sum += x;
            }
            if !(sum > 0.0) {
                return Err(FxVarError::invalid("weights must be non-negative and sum > 0"));
            }
            ws.iter().map(|x| x / sum).collect()
        }
    };

    // Losses descending == P&L ascending (most negative P&L = biggest loss
    // first). A stable sort keeps ties in their original (chronological)
    // order, matching the C++/Python reference.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| pnl[a].partial_cmp(&pnl[b]).expect("NaNs already excluded"));

    let losses: Vec<f64> = order.iter().map(|&i| -pnl[i]).collect();
    let ws: Vec<f64> = order.iter().map(|&i| w[i]).collect();

    let target = 1.0 - alpha;
    let mut cum = 0.0;
    let mut idx = n - 1;
    for i in 0..n {
        cum += ws[i];
        if cum >= target - WEIGHT_TOL {
            idx = i;
            break;
        }
    }
    Ok(Tail { losses, weights: ws, idx })
}

fn es_from_tail(t: &Tail, alpha: f64) -> f64 {
    let target = 1.0 - alpha;
    let mut full = 0.0;
    let mut cum_before = 0.0;
    for i in 0..t.idx {
        full += t.losses[i] * t.weights[i];
        cum_before += t.weights[i];
    }
    let frac = (target - cum_before).max(0.0);
    (full + frac * t.losses[t.idx]) / target
}

/// Empirical VaR (positive loss) at confidence level `alpha`. `weights`
/// (if given) are non-negative scenario weights, normalised internally;
/// `None` means uniform.
///
/// # Errors
/// [`FxVarError::Invalid`] for empty/NaN `pnl`, bad `alpha`, or bad
/// `weights`.
pub fn empirical_var(pnl: &[f64], alpha: f64, weights: Option<&[f64]>) -> Result<f64> {
    let t = make_tail(pnl, alpha, weights)?;
    Ok(t.losses[t.idx])
}

/// Empirical ES (positive loss): coherent Acerbi-Tasche estimator.
///
/// # Errors
/// Same as [`empirical_var`].
pub fn empirical_es(pnl: &[f64], alpha: f64, weights: Option<&[f64]>) -> Result<f64> {
    let t = make_tail(pnl, alpha, weights)?;
    Ok(es_from_tail(&t, alpha))
}

/// `(VaR, ES)` from one pass over the sample.
///
/// # Errors
/// Same as [`empirical_var`].
pub fn empirical_var_es(pnl: &[f64], alpha: f64, weights: Option<&[f64]>) -> Result<(f64, f64)> {
    let t = make_tail(pnl, alpha, weights)?;
    let es = es_from_tail(&t, alpha);
    Ok((t.losses[t.idx], es))
}

/// Closed-form Normal VaR: `-mean + sigma * z_alpha` (positive loss).
///
/// # Errors
/// [`FxVarError::Invalid`] if `sigma < 0` or `alpha` is out of range.
pub fn normal_var(sigma: f64, alpha: f64, mean: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    if sigma < 0.0 {
        return Err(FxVarError::invalid("sigma must be >= 0"));
    }
    Ok(-mean + sigma * inv_norm_cdf(alpha))
}

/// Closed-form Normal ES: `-mean + sigma * phi(z_alpha) / (1 - alpha)`.
///
/// # Errors
/// Same as [`normal_var`].
pub fn normal_es(sigma: f64, alpha: f64, mean: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    if sigma < 0.0 {
        return Err(FxVarError::invalid("sigma must be >= 0"));
    }
    let z = inv_norm_cdf(alpha);
    Ok(-mean + sigma * norm_pdf(z) / (1.0 - alpha))
}

fn t_scale(df: f64) -> Result<f64> {
    if df <= 2.0 {
        return Err(FxVarError::invalid("Student-t df must be > 2 for finite variance"));
    }
    Ok(((df - 2.0) / df).sqrt())
}

/// Standardised Student-t VaR with true P&L std `sigma` (unit-variance
/// scaling `sqrt((df-2)/df)`); `df` must be `> 2` for finite variance.
///
/// # Errors
/// [`FxVarError::Invalid`] if `sigma < 0`, `alpha` is out of range, or
/// `df <= 2`.
pub fn student_t_var(sigma: f64, alpha: f64, df: f64, mean: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    if sigma < 0.0 {
        return Err(FxVarError::invalid("sigma must be >= 0"));
    }
    let q = student_t_quantile(alpha, df)?;
    Ok(-mean + sigma * t_scale(df)? * q)
}

/// Standardised Student-t ES:
/// `E[X | X > q_a] = f(q_a) (df + q_a^2) / ((1-a)(df-1))` for the standard
/// t.
///
/// # Errors
/// Same as [`student_t_var`].
pub fn student_t_es(sigma: f64, alpha: f64, df: f64, mean: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    if sigma < 0.0 {
        return Err(FxVarError::invalid("sigma must be >= 0"));
    }
    let q = student_t_quantile(alpha, df)?;
    let es_std = student_t_pdf(q, df) * (df + q * q) / ((1.0 - alpha) * (df - 1.0));
    Ok(-mean + sigma * t_scale(df)? * es_std)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_var_es_matches_hand_computation() {
        // 100 scenarios: pnl = -1, -2, ..., -100.
        let pnl: Vec<f64> = (1..=100).map(|i| -(i as f64)).collect();
        let var = empirical_var(&pnl, 0.99, None).unwrap();
        // 1% tail on 100 obs = worst 1 -> VaR = 100.
        assert!((var - 100.0).abs() < 1e-9);
        let es = empirical_es(&pnl, 0.99, None).unwrap();
        assert!(es >= var);
    }

    #[test]
    fn es_at_least_var_always() {
        let pnl = vec![-5.0, -3.0, -1.0, 0.5, 2.0, -8.0, 1.0, -2.0, 3.0, -0.5];
        let (var, es) = empirical_var_es(&pnl, 0.9, None).unwrap();
        assert!(es >= var - 1e-12);
    }

    #[test]
    fn normal_es_ge_var() {
        let sigma = 1.0e6;
        let var = normal_var(sigma, 0.99, 0.0).unwrap();
        let es = normal_es(sigma, 0.99, 0.0).unwrap();
        assert!(es > var);
    }

    #[test]
    fn student_t_fatter_tail_than_normal_at_equal_sigma() {
        let sigma = 1.0e6;
        let nvar = normal_var(sigma, 0.99, 0.0).unwrap();
        let tvar = student_t_var(sigma, 0.99, 5.0, 0.0).unwrap();
        assert!(tvar > nvar);
    }

    #[test]
    fn invalid_alpha_errors() {
        assert!(empirical_var(&[1.0, -1.0], 1.5, None).is_err());
        assert!(normal_var(-1.0, 0.99, 0.0).is_err());
        assert!(student_t_var(1.0, 0.99, 2.0, 0.0).is_err());
    }

    #[test]
    fn weighted_var_matches_uniform_special_case() {
        let pnl: Vec<f64> = (1..=200).map(|i| -(i as f64)).collect();
        let w = vec![1.0; 200];
        let var_w = empirical_var(&pnl, 0.95, Some(&w)).unwrap();
        let var_u = empirical_var(&pnl, 0.95, None).unwrap();
        assert!((var_w - var_u).abs() < 1e-9);
    }

    #[test]
    fn bad_weights_error() {
        let pnl = vec![-1.0, -2.0, -3.0];
        assert!(empirical_var(&pnl, 0.9, Some(&[1.0, -1.0, 1.0])).is_err());
        assert!(empirical_var(&pnl, 0.9, Some(&[1.0, 1.0])).is_err());
    }
}
