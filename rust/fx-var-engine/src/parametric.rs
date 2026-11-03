//! Parametric (variance-covariance) VaR: normal, Student-t, Cornish-Fisher.
//!
//! The book is linearised into factor exposures `w`
//! ([`crate::book::CompiledBook::linear_exposures`]: forwards enter via
//! their deposit legs), and the portfolio P&L variance is `w' Sigma w` with
//! `Sigma` a sample or EWMA covariance of daily factor returns (mirrors
//! Python `fx_var.parametric_var` / C++ `fxvar::parametric_var`).
//!
//! Distributional overlays on the same sigma:
//!
//! * **normal**: RiskMetrics classic; underestimates FX tails.
//! * **Student-t**: standardised (unit-variance) t captures the fat tails
//!   of EM currency returns at equal sigma.
//! * **Cornish-Fisher**: moment-corrected quantile from the portfolio's
//!   empirical skew/kurtosis; only valid inside the monotonicity domain of
//!   the expansion, which is checked explicitly (Maillard 2012: outside
//!   the domain the "quantile" is not a quantile).
//!
//! Multi-day horizon: sigma scales by `sqrt(h)` (i.i.d. assumption).

use crate::book::{Book, CompiledBook, Market};
use crate::expected_shortfall::{normal_es, normal_var, student_t_es, student_t_var};
use crate::matrix::Matrix;
use crate::returns::{ReturnsMatrix, ewma_cov, flag_peg_factors, sample_cov, validate_returns};
use crate::stats::{inv_norm_cdf, validate_alpha, validate_horizon};
use crate::{FxVarError, Result, TailDist};

/// `(VaR, ES)` pair, positive losses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarEs {
    /// Positive VaR loss.
    pub var: f64,
    /// Positive ES loss.
    pub es: f64,
}

/// 1-day portfolio P&L standard deviation `sqrt(w' Sigma w)`.
///
/// # Errors
/// [`FxVarError::Invalid`] on a dimension mismatch, on a non-finite
/// exposure or covariance entry, or if the quadratic form is materially
/// negative (covariance not PSD). The PSD tolerance is **scale-relative**
/// (`1e-10 * max|w|^2`), so a large-notional book is not rejected merely
/// for being large.
pub fn portfolio_sigma(exposures: &[f64], cov: &Matrix) -> Result<f64> {
    // Finiteness first, and for a concrete reason. `w' Sigma w` with a
    // single NaN anywhere is NaN; the PSD test below (`var < -tol`) is
    // FALSE for NaN, and `f64::max` propagates the *other* operand, so
    // `NaN.max(0.0).sqrt()` is `0.0`. Without this guard a corrupt
    // covariance reported a portfolio sigma - and a VaR, and an ES - of
    // exactly ZERO: a broken feed that looks like a perfectly hedged book.
    if exposures.iter().any(|w| !w.is_finite()) {
        return Err(FxVarError::invalid(
            "portfolio_sigma: exposures must all be finite (a NaN exposure \
             silently produces a zero sigma)",
        ));
    }
    if !cov.all_finite() {
        return Err(FxVarError::invalid(
            "portfolio_sigma: covariance must be finite (a NaN entry silently \
             produces a zero sigma, i.e. a zero VaR)",
        ));
    }
    let var = cov.quad_form(exposures)?;
    let scale = exposures.iter().fold(1.0_f64, |m, w| m.max(w.abs()));
    if var < -1e-10 * scale * scale {
        return Err(FxVarError::invalid(
            "portfolio_sigma: covariance matrix is not positive semi-definite",
        ));
    }
    Ok(var.max(0.0).sqrt())
}

/// Closed-form `(VaR, ES)` for a linear book: pure function for testing.
/// `exposures` are base-ccy P&L per unit factor move; `cov` is the daily
/// factor covariance in the same order; `df` is used for
/// [`TailDist::StudentT`]; `mean` is the expected 1-day P&L (usually 0 at
/// daily horizon).
///
/// # Errors
/// [`FxVarError::Invalid`] on bad `alpha`/`horizon_days`, a non-PSD
/// covariance, or (for `StudentT`) `df <= 2`.
pub fn var_covar(
    exposures: &[f64],
    cov: &Matrix,
    alpha: f64,
    horizon_days: f64,
    dist: TailDist,
    df: f64,
    mean: f64,
) -> Result<VarEs> {
    validate_alpha(alpha)?;
    validate_horizon(horizon_days)?;
    if !mean.is_finite() {
        return Err(FxVarError::invalid(format!(
            "var_covar: mean must be finite, got {mean}"
        )));
    }
    let sig1 = portfolio_sigma(exposures, cov)?;
    let scale = horizon_days.sqrt();
    let sig = sig1 * scale;
    let mu = mean * horizon_days;
    match dist {
        TailDist::Normal => Ok(VarEs {
            var: normal_var(sig, alpha, mu)?,
            es: normal_es(sig, alpha, mu)?,
        }),
        TailDist::StudentT => Ok(VarEs {
            var: student_t_var(sig, alpha, df, mu)?,
            es: student_t_es(sig, alpha, df, mu)?,
        }),
    }
}

/// Covariance estimator for [`parametric_var`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovMethod {
    /// Unbiased (ddof=1) sample covariance.
    Sample,
    /// RiskMetrics EWMA covariance.
    Ewma,
}

/// Options controlling [`parametric_var`].
#[derive(Debug, Clone, PartialEq)]
pub struct ParametricOptions {
    /// Tail probability, e.g. `0.01` for 99 % VaR (was a confidence level).
    pub alpha: f64,
    /// Horizon in trading days.
    pub horizon_days: f64,
    /// Tail distribution.
    pub dist: TailDist,
    /// Student-t degrees of freedom (used when `dist == StudentT`).
    pub df: f64,
    /// Covariance estimator.
    pub cov_method: CovMethod,
    /// EWMA decay (used when `cov_method == Ewma`).
    pub ewma_lambda: f64,
    /// Minimum required history length.
    pub min_obs: usize,
    /// Whether to screen `FX:*` factors for peg blindness.
    pub warn_pegs: bool,
}

impl Default for ParametricOptions {
    fn default() -> Self {
        ParametricOptions {
            alpha: 0.01,
            horizon_days: 1.0,
            dist: TailDist::Normal,
            df: 6.0,
            cov_method: CovMethod::Sample,
            ewma_lambda: 0.94,
            min_obs: 60,
            warn_pegs: true,
        }
    }
}

/// Variance-covariance VaR result (positive base-ccy losses).
#[derive(Debug, Clone, PartialEq)]
pub struct ParametricResult {
    /// Positive VaR loss.
    pub var: f64,
    /// Positive ES loss.
    pub es: f64,
    /// Tail probability used.
    pub alpha: f64,
    /// Horizon in trading days used.
    pub horizon_days: f64,
    /// Tail distribution used.
    pub dist: TailDist,
    /// 1-day portfolio P&L std in base ccy.
    pub sigma: f64,
    /// Risk factors of the compiled book.
    pub factors: Vec<String>,
    /// Linear exposures, aligned with `factors`.
    pub exposures: Vec<f64>,
    /// `FX:*` factors flagged as peg-like.
    pub flagged_peg_factors: Vec<String>,
    /// Human-readable peg-blindness diagnostics.
    pub warnings: Vec<String>,
}

/// Variance-covariance VaR/ES of `book` from a factor-return history.
///
/// # Errors
/// [`FxVarError::Invalid`] for an empty book, bad options, or a history
/// that fails validation; a factorless (pure base-ccy cash) book reports
/// exactly zero.
pub fn parametric_var(
    book: &Book,
    market: &Market,
    returns: &ReturnsMatrix,
    opts: &ParametricOptions,
) -> Result<ParametricResult> {
    validate_alpha(opts.alpha)?;
    validate_horizon(opts.horizon_days)?;
    let compiled = CompiledBook::new(book, market)?;

    let mut res = ParametricResult {
        var: 0.0,
        es: 0.0,
        alpha: opts.alpha,
        horizon_days: opts.horizon_days,
        dist: opts.dist,
        sigma: 0.0,
        factors: compiled.factors().to_vec(),
        exposures: Vec::new(),
        flagged_peg_factors: Vec::new(),
        warnings: Vec::new(),
    };
    if res.factors.is_empty() {
        return Ok(res); // pure base-ccy cash: zero risk
    }

    validate_returns(returns, &res.factors, opts.min_obs)?;
    let rets = returns.select(&res.factors)?;

    if opts.warn_pegs {
        res.flagged_peg_factors = flag_peg_factors(&rets, crate::returns::PEG_VOL_THRESHOLD)?;
        for f in &res.flagged_peg_factors {
            res.warnings.push(format!(
                "peg blindness: factor {f} has daily vol < {} (pegged/managed currency). \
                 Historical and parametric VaR are blind to peg-break risk; add the \
                 peg-break stress add-on (crate::stress::peg_break_scenario).",
                crate::returns::PEG_VOL_THRESHOLD
            ));
        }
    }

    let cov = match opts.cov_method {
        CovMethod::Sample => sample_cov(&rets)?,
        CovMethod::Ewma => ewma_cov(&rets, opts.ewma_lambda)?,
    };
    res.exposures = compiled.linear_exposures(1e-6)?;
    let ve = var_covar(
        &res.exposures,
        &cov.cov,
        opts.alpha,
        opts.horizon_days,
        opts.dist,
        opts.df,
        0.0,
    )?;
    res.var = ve.var;
    res.es = ve.es;
    res.sigma = portfolio_sigma(&res.exposures, &cov.cov)?;
    Ok(res)
}

// ---------------------------------------------------------------------
// Cornish-Fisher
// ---------------------------------------------------------------------

/// Cornish-Fisher adjusted quantile:
/// `z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36`.
pub fn cornish_fisher_z(z: f64, skew: f64, excess_kurtosis: f64) -> f64 {
    let z2 = z * z;
    let z3 = z2 * z;
    z + (z2 - 1.0) * skew / 6.0 + (z3 - 3.0 * z) * excess_kurtosis / 24.0
        - (2.0 * z3 - 5.0 * z) * skew * skew / 36.0
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
        return Err(FxVarError::Invalid(format!(
            "Cornish-Fisher moments must be finite, got skew={skew}, excess_kurt={excess_kurt}"
        )));
    }
    let z_a = inv_norm_cdf(1.0 - alpha)?;
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

/// Cornish-Fisher VaR (positive loss) from the P&L sigma, skewness and
/// excess kurtosis at tail probability `alpha`:
/// `VaR = -mean h - sigma sqrt(h) z_cf`, `z_cf = cornish_fisher_z(Phi^{-1}(alpha))`.
///
/// # Errors
/// [`FxVarError::Invalid`] for invalid `alpha` / horizon, a negative or
/// non-finite sigma, non-finite moments or mean, or (with
/// `check_domain`) moments for which [`cornish_fisher_valid`] is false.
#[allow(clippy::too_many_arguments)]
pub fn cornish_fisher_var(
    sigma: f64,
    skew: f64,
    excess_kurtosis: f64,
    alpha: f64,
    mean: f64,
    horizon_days: f64,
    check_domain: bool,
) -> Result<f64> {
    validate_alpha(alpha)?;
    validate_horizon(horizon_days)?;
    if !(sigma >= 0.0) || !sigma.is_finite() {
        return Err(FxVarError::invalid(format!(
            "sigma must be finite and >= 0, got {sigma}"
        )));
    }
    if !skew.is_finite() || !excess_kurtosis.is_finite() || !mean.is_finite() {
        return Err(FxVarError::invalid(format!(
            "skew, excess_kurtosis and mean must be finite, got {skew}, {excess_kurtosis}, {mean}"
        )));
    }
    if check_domain && !cornish_fisher_valid(skew, excess_kurtosis, alpha)? {
        return Err(FxVarError::invalid(format!(
            "Cornish-Fisher expansion is non-monotone for skew={skew}, \
             excess_kurtosis={excess_kurtosis}: outside validity domain; fall back to \
             historical or t VaR (set check_domain=false to force)"
        )));
    }
    // Loss quantile: lower tail of the P&L distribution.
    let z = inv_norm_cdf(alpha)?;
    let zcf = cornish_fisher_z(z, skew, excess_kurtosis);
    let scale = horizon_days.sqrt();
    Ok(-(mean * horizon_days) - sigma * scale * zcf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::{Position, SpotPosition};
    use crate::matrix::Matrix;

    #[test]
    fn portfolio_sigma_matches_hand_computation() {
        let cov = Matrix::from_rows(&[vec![0.0001, 0.0], vec![0.0, 0.0004]]).unwrap();
        let sig = portfolio_sigma(&[1.0e6, 1.0e6], &cov).unwrap();
        let expect = (1.0e12 * 0.0001 + 1.0e12 * 0.0004_f64).sqrt();
        assert!((sig - expect).abs() < 1e-6);
    }

    #[test]
    fn var_covar_normal_matches_z_score() {
        let cov = Matrix::from_rows(&[vec![0.0001]]).unwrap();
        let ve = var_covar(&[1.0e6], &cov, 0.01, 1.0, TailDist::Normal, 6.0, 0.0).unwrap();
        let z = inv_norm_cdf(0.99).unwrap();
        let sigma = 1.0e6 * 0.0001_f64.sqrt();
        assert!((ve.var - z * sigma).abs() < 1e-6);
    }

    #[test]
    fn student_t_var_exceeds_normal_at_equal_sigma() {
        let cov = Matrix::from_rows(&[vec![0.0001]]).unwrap();
        let n = var_covar(&[1.0e6], &cov, 0.01, 1.0, TailDist::Normal, 6.0, 0.0).unwrap();
        let t = var_covar(&[1.0e6], &cov, 0.01, 1.0, TailDist::StudentT, 5.0, 0.0).unwrap();
        assert!(t.var > n.var);
    }

    #[test]
    fn horizon_scaling_is_sqrt_time() {
        let cov = Matrix::from_rows(&[vec![0.0001]]).unwrap();
        let v1 = var_covar(&[1.0e6], &cov, 0.01, 1.0, TailDist::Normal, 6.0, 0.0).unwrap();
        let v10 = var_covar(&[1.0e6], &cov, 0.01, 10.0, TailDist::Normal, 6.0, 0.0).unwrap();
        assert!((v10.var - v1.var * 10f64.sqrt()).abs() < 1e-6 * v10.var);
    }

    #[test]
    fn negative_semidefinite_covariance_errors() {
        let cov = Matrix::from_rows(&[vec![-1.0]]).unwrap();
        assert!(portfolio_sigma(&[1.0], &cov).is_err());
    }

    #[test]
    fn cornish_fisher_reduces_to_normal_at_zero_moments() {
        let z = inv_norm_cdf(0.95).unwrap();
        assert!((cornish_fisher_z(z, 0.0, 0.0) - z).abs() < 1e-12);
    }

    #[test]
    #[allow(deprecated)]
    fn cornish_fisher_domain_rejects_extreme_moments() {
        assert!(!cornish_fisher_valid(5.0, 50.0, 0.01).unwrap());
        assert!(cornish_fisher_valid(0.1, 0.5, 0.01).unwrap());
    }

    #[test]
    fn cornish_fisher_var_errors_outside_domain() {
        let res = cornish_fisher_var(1.0e6, 5.0, 50.0, 0.01, 0.0, 1.0, true);
        assert!(res.is_err());
        let res2 = cornish_fisher_var(1.0e6, 5.0, 50.0, 0.01, 0.0, 1.0, false);
        assert!(res2.is_ok());
    }

    #[test]
    fn cornish_fisher_validity_interval_follows_alpha() {
        // (0.122, -0.427): g' is concave (A < 0), so its minimum on a closed
        // interval is at an endpoint. On [-4, 4] the minimum is at z = -4
        // and is negative; at alpha = 0.01 the interval is [-2.33, 2.33]
        // and the map is valid. The old fixed |z| <= 4 check rejected it.
        let (skew, k) = (0.122, -0.427);
        assert!(cornish_fisher_valid(skew, k, 0.01).unwrap());
        assert!(cornish_fisher_var(1.0, skew, k, 0.01, 0.0, 1.0, true).is_ok());
        assert!(!cornish_fisher_valid(skew, k, 1e-5).unwrap());
        assert!(cornish_fisher_var(1.0, skew, k, 1e-5, 0.0, 1.0, true).is_err());
    }

    #[test]
    fn zero_factor_book_reports_zero_parametric_var() {
        let market = Market::new([("EUR", 1.10)], [("USD", 0.05), ("EUR", 0.03)]).unwrap();
        let book = Book::new(
            vec![Position::Spot(SpotPosition::new("EURUSD", 0.0, None))],
            "USD",
        )
        .unwrap();
        let mut data = Matrix::zeros(120, 1);
        for t in 0..120 {
            data.set(t, 0, 0.01 * (t as f64).sin());
        }
        let rets = ReturnsMatrix::new(vec!["FX:EUR".to_string()], data).unwrap();
        let r = parametric_var(&book, &market, &rets, &ParametricOptions::default()).unwrap();
        assert_eq!(r.var, 0.0);
    }
}
