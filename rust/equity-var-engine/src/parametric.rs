//! Parametric (variance–covariance) VaR.
//!
//! Portfolio sigma from dollar exposures `w` and factor-return covariance
//! `Sigma`: `sigma_p = sqrt(w' Sigma w)`. Quantiles from the normal, the
//! variance-matched Student-t, or the Cornish–Fisher expansion with an
//! explicit validity-domain check.
//!
//! Conventions: `alpha` = tail probability; VaR positive for losses; daily
//! covariance in factor-return units matching the dollar exposures.

use crate::matrix::Matrix;
use crate::stats::{normal_ppf, student_t_ppf};
use crate::{validate_alpha, EqVarError, Result, TailModel};

/// Portfolio P&L standard deviation `sqrt(w' Sigma w)` (currency units).
///
/// Errors if the covariance shape does not match the exposure count, if any
/// exposure or covariance entry is non-finite, or if the quadratic form is
/// materially negative (matrix not PSD).
///
/// The finiteness check is load-bearing, not cosmetic. `w' Sigma w` with a
/// single NaN anywhere evaluates to NaN; the PSD test `var < -tol` is
/// **false** for NaN, and `f64::max` propagates the *other* operand, so
/// `NaN.max(0.0).sqrt()` is `0.0`. Without this guard a corrupt covariance
/// therefore reported a portfolio sigma — and a VaR — of exactly **zero**:
/// a broken model that looks like a riskless book.
pub fn portfolio_sigma(exposures: &[f64], cov: &Matrix) -> Result<f64> {
    let n = exposures.len();
    if n == 0 {
        return Err(EqVarError::InvalidInput(
            "exposures must not be empty".to_string(),
        ));
    }
    if cov.rows() != n || cov.cols() != n {
        return Err(EqVarError::InvalidInput(format!(
            "covariance shape {} x {} does not match {n} exposures",
            cov.rows(),
            cov.cols()
        )));
    }
    if exposures.iter().any(|w| !w.is_finite()) {
        return Err(EqVarError::InvalidInput(
            "exposures must all be finite (NaN or infinite exposures silently              produce a zero portfolio sigma)"
                .to_string(),
        ));
    }
    if !cov.all_finite() {
        return Err(EqVarError::InvalidInput(
            "covariance matrix must be finite (a NaN entry silently produces a              zero portfolio sigma, i.e. a zero VaR)"
                .to_string(),
        ));
    }
    let sw = cov.matvec(exposures)?;
    let var: f64 = exposures.iter().zip(sw.iter()).map(|(w, s)| w * s).sum();
    let scale = exposures.iter().fold(1.0f64, |m, w| m.max(w.abs()));
    if var < -1e-10 * scale * scale {
        return Err(EqVarError::InvalidInput(
            "covariance matrix is not positive semi-definite (w'Sw < 0)".to_string(),
        ));
    }
    Ok(var.max(0.0).sqrt())
}

/// Tail quantile `z_alpha` of the chosen unit-variance tail model.
///
/// Normal: `Phi^{-1}(alpha)`. Student-t: `t_df^{-1}(alpha) * sqrt((df-2)/df)`
/// (variance-matched so sigma is unchanged and only the tail fattens);
/// errors if `df <= 2`.
pub fn tail_quantile(alpha: f64, tail: TailModel) -> Result<f64> {
    validate_alpha(alpha)?;
    match tail {
        TailModel::Normal => normal_ppf(alpha),
        TailModel::StudentT { df } => {
            if !(df > 2.0) || !df.is_finite() {
                return Err(EqVarError::InvalidInput(format!(
                    "Student-t df must be finite and > 2 for finite variance, got {df}"
                )));
            }
            Ok(student_t_ppf(alpha, df)? * ((df - 2.0) / df).sqrt())
        }
    }
}

/// Variance–covariance VaR at a 1-day horizon with zero mean.
///
/// `VaR = -z_alpha * sigma_p` with `z_alpha` from [`tail_quantile`].
/// See [`parametric_var_full`] for a drift and multi-day horizon.
///
/// # Examples
///
/// ```
/// use eq_var_engine::matrix::Matrix;
/// use eq_var_engine::parametric::parametric_var;
/// use eq_var_engine::TailModel;
/// let cov = Matrix::from_vec(1, 1, vec![0.0001]).unwrap(); // 1 % daily vol
/// let var = parametric_var(&[1.0e6], &cov, 0.01, TailModel::Normal).unwrap();
/// assert!((var - 23_263.478740408408).abs() < 1e-6); // 2.3263 sigma
/// ```
pub fn parametric_var(
    exposures: &[f64],
    cov: &Matrix,
    alpha: f64,
    tail: TailModel,
) -> Result<f64> {
    parametric_var_full(exposures, cov, alpha, tail, 0.0, 1)
}

/// Variance–covariance VaR with expected daily P&L `mean` and a
/// `horizon_days` horizon (square-root-of-time scaling of sigma, linear
/// scaling of the mean).
///
/// Returns VaR as a positive loss: `-(mu_h + z_alpha * sigma_h)`.
pub fn parametric_var_full(
    exposures: &[f64],
    cov: &Matrix,
    alpha: f64,
    tail: TailModel,
    mean: f64,
    horizon_days: u32,
) -> Result<f64> {
    validate_alpha(alpha)?;
    if horizon_days < 1 {
        return Err(EqVarError::InvalidInput(format!(
            "horizon_days must be >= 1, got {horizon_days}"
        )));
    }
    if !mean.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "expected daily P&L must be finite, got {mean}"
        )));
    }
    let sigma = portfolio_sigma(exposures, cov)? * (horizon_days as f64).sqrt();
    let mu = mean * horizon_days as f64;
    let z = tail_quantile(alpha, tail)?;
    Ok(-(mu + z * sigma))
}

// ---------------------------------------------------------------------------
// Cornish–Fisher
// ---------------------------------------------------------------------------

/// Cornish–Fisher adjusted quantile.
///
/// `z_cf = z + (z^2 - 1) S / 6 + (z^3 - 3z) K / 24 - (2z^3 - 5z) S^2 / 36`
/// with skewness `S` and **excess** kurtosis `K`. Reduces to `z` when
/// `S = K = 0`.
pub fn cornish_fisher_z(z: f64, skew: f64, excess_kurt: f64) -> f64 {
    z + (z * z - 1.0) * skew / 6.0
        + (z * z * z - 3.0 * z) * excess_kurt / 24.0
        - (2.0 * z * z * z - 5.0 * z) * skew * skew / 36.0
}

/// Cornish-Fisher validity at tail probability `alpha` (S6).
///
/// With `z_a = Phi^{-1}(1 - alpha)` the CF map
/// `g(z) = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36` (`K` excess
/// kurtosis) is a valid quantile transform iff `g'(z) > 0` on the closed
/// interval `[-z_a, z_a]`. `g'` is the quadratic
/// `A z^2 + B z + C`, `A = K/8 - S^2/6`, `B = S/3`, `C = 1 - K/8 + 5S^2/36`,
/// so its minimum on the interval is at the vertex (if `A > 0` and the
/// vertex is inside) or at an endpoint. The check is exact, not a grid.
///
/// The interval follows `alpha`: `(S, K) = (0.122, -0.427)` has its
/// smallest derivative at the endpoint, is valid at `alpha = 0.01`
/// (`z_a = 2.33`) and becomes invalid only once `z_a` passes about 4.
///
/// # Errors
/// Non-finite `skew` or `excess_kurt`, or `alpha` outside `(0, 0.5)`.
pub fn cornish_fisher_valid(skew: f64, excess_kurt: f64, alpha: f64) -> Result<bool> {
    validate_alpha(alpha)?;
    if !skew.is_finite() || !excess_kurt.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "Cornish-Fisher moments must be finite, got skew={skew}, excess_kurt={excess_kurt}"
        )));
    }
    let z_a = normal_ppf(1.0 - alpha)?;
    Ok(cf_min_derivative(skew, excess_kurt, z_a) > 0.0)
}

/// Minimum of `g'(z)` over `[-z_range, z_range]` (closed form).
fn cf_min_derivative(skew: f64, excess_kurt: f64, z_range: f64) -> f64 {
    let a = excess_kurt / 8.0 - skew * skew / 6.0;
    let b = skew / 3.0;
    let c = 1.0 - excess_kurt / 8.0 + 5.0 * skew * skew / 36.0;
    let g = |z: f64| a * z * z + b * z + c;
    let ends = g(-z_range).min(g(z_range));
    if a > 0.0 {
        let z_star = -b / (2.0 * a);
        if (-z_range..=z_range).contains(&z_star) {
            return g(z_star).min(ends);
        }
    }
    ends
}

/// Old fixed-interval check `g'(z) > 0` on `[-z_range, z_range]`, kept for
/// one release. The interval should follow the VaR level; use
/// [`cornish_fisher_valid`]. Non-finite input returns `false`.
#[deprecated(since = "0.2.0", note = "use cornish_fisher_valid(skew, excess_kurt, alpha)")]
pub fn cornish_fisher_domain_ok(
    skew: f64,
    excess_kurt: f64,
    z_range: f64,
    _n_grid: usize,
) -> bool {
    if !skew.is_finite() || !excess_kurt.is_finite() || !z_range.is_finite() {
        return false;
    }
    cf_min_derivative(skew, excess_kurt, z_range) > 0.0
}

/// Cornish–Fisher VaR: moment-corrected parametric quantile.
///
/// `VaR = -(mean + z_cf(alpha) * sigma)`. With `check_domain = true`
/// (recommended) an [`EqVarError::InvalidInput`] is returned when
/// [`cornish_fisher_valid`] is false at this `alpha`. Outside that region
/// the expansion is not a quantile function and the number is not a VaR.
pub fn cornish_fisher_var(
    sigma: f64,
    alpha: f64,
    skew: f64,
    excess_kurt: f64,
    mean: f64,
    check_domain: bool,
) -> Result<f64> {
    validate_alpha(alpha)?;
    if !(sigma >= 0.0) || !sigma.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "sigma must be finite and >= 0, got {sigma}"
        )));
    }
    if !skew.is_finite() || !excess_kurt.is_finite() || !mean.is_finite() {
        return Err(EqVarError::InvalidInput(format!(
            "skew, excess_kurt and mean must be finite, got skew={skew}, \
             excess_kurt={excess_kurt}, mean={mean}"
        )));
    }
    if check_domain && !cornish_fisher_valid(skew, excess_kurt, alpha)? {
        return Err(EqVarError::InvalidInput(format!(
            "Cornish-Fisher expansion is non-monotone for skew={skew}, \
             excess_kurt={excess_kurt}; outside its validity region the 'quantile' \
             is not a quantile. Use historical or MC VaR instead."
        )));
    }
    let z = cornish_fisher_z(normal_ppf(alpha)?, skew, excess_kurt);
    Ok(-(mean + z * sigma))
}
