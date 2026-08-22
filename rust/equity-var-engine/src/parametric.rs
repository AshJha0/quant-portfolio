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

/// Check that the CF quantile map is monotone on `[-z_range, z_range]`.
///
/// The fourth-order Cornish–Fisher expansion is only a valid quantile
/// function where `dz_cf/dz > 0`; for large skew/kurtosis the cubic becomes
/// non-monotone and the implied "density" goes negative, producing nonsense
/// VaR (e.g. the 99 % "quantile" above the 95 % one). The analytic
/// derivative
///
/// `dz_cf/dz = 1 + z S / 3 + (3z^2 - 3) K / 24 - (6z^2 - 5) S^2 / 36`
///
/// is itself a quadratic in `z`: `g(z) = A z^2 + B z + C` with
/// `A = K/8 - S^2/6`, `B = S/3`, `C = 1 - K/8 + 5 S^2/36`. Its minimum on
/// `[-z_range, z_range]` therefore has a closed form — the vertex
/// `-B/(2A)` when `g` is convex (`A > 0`) and that point lies in range,
/// otherwise an interval endpoint (a concave/linear `g`, `A <= 0`, always
/// attains its minimum on a closed interval at an endpoint) — checked
/// exactly here rather than by sampling a grid (`n_grid` points; the
/// previous implementation used 2001 points on `|z| <= 3.5`). A grid can
/// under-sample a thin non-monotone dip: `skew = 0.122, excess_kurt =
/// -0.427` is non-monotone on `|z| <= 4` (minimum derivative ~ -9e-4 near
/// `z ~ 3.1`) but an 801-point grid on that range reports it as monotone.
/// `n_grid` is retained for API compatibility but no longer affects the
/// result.
/// Returns `true` when the expansion is monotone (safe to use).
pub fn cornish_fisher_domain_ok(
    skew: f64,
    excess_kurt: f64,
    z_range: f64,
    n_grid: usize,
) -> bool {
    // NaN moments are never "ok": an exact comparison is false for NaN, so
    // a NaN skew/kurtosis would otherwise be reported as a valid domain
    // and produce a NaN "VaR".
    if !skew.is_finite() || !excess_kurt.is_finite() || !z_range.is_finite() {
        return false;
    }
    let _ = n_grid.max(2); // API compatibility only; unused by the closed-form check.
    let a = excess_kurt / 8.0 - skew * skew / 6.0;
    let b = skew / 3.0;
    let c = 1.0 - excess_kurt / 8.0 + 5.0 * skew * skew / 36.0;
    let g = |z: f64| a * z * z + b * z + c;
    let m = if a > 0.0 {
        let z_star = -b / (2.0 * a);
        if (-z_range..=z_range).contains(&z_star) {
            g(z_star)
        } else {
            g(-z_range).min(g(z_range))
        }
    } else {
        g(-z_range).min(g(z_range))
    };
    m > 0.0
}

/// Cornish–Fisher VaR: moment-corrected parametric quantile.
///
/// `VaR = -(mean + z_cf(alpha) * sigma)`. With `check_domain = true`
/// (recommended) an [`EqVarError::InvalidInput`] is returned when
/// `(skew, excess_kurt)` lie outside the monotonicity region — outside it
/// the expansion is not a quantile function and the number is not a VaR;
/// use historical or Monte Carlo VaR instead.
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
    if check_domain && !cornish_fisher_domain_ok(skew, excess_kurt, 3.5, 2001) {
        return Err(EqVarError::InvalidInput(format!(
            "Cornish-Fisher expansion is non-monotone for skew={skew}, \
             excess_kurt={excess_kurt}; outside its validity region the 'quantile' \
             is not a quantile. Use historical or MC VaR instead."
        )));
    }
    let z = cornish_fisher_z(normal_ppf(alpha)?, skew, excess_kurt);
    Ok(-(mean + z * sigma))
}
