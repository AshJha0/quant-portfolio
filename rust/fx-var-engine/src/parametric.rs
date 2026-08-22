//! Parametric (variance-covariance) VaR: normal, Student-t, Cornish-Fisher.
//!
//! The book is linearised into factor exposures `w`
//! ([`crate::book::CompiledBook::linear_exposures`] — forwards enter via
//! their deposit legs), and the portfolio P&L variance is `w' Sigma w` with
//! `Sigma` a sample or EWMA covariance of daily factor returns (mirrors
//! Python `fx_var.parametric_var` / C++ `fxvar::parametric_var`).
//!
//! Distributional overlays on the same sigma:
//!
//! * **normal** — RiskMetrics classic; underestimates FX tails.
//! * **Student-t** — standardised (unit-variance) t captures the fat tails
//!   of EM currency returns at equal sigma.
//! * **Cornish-Fisher** — moment-corrected quantile from the portfolio's
//!   empirical skew/kurtosis; only valid inside the monotonicity domain of
//!   the expansion, which is checked explicitly (Maillard 2012 — outside
//!   the domain the "quantile" is not a quantile).
//!
//! Multi-day horizon: sigma scales by `sqrt(h)` (i.i.d. assumption).

use crate::book::{Book, CompiledBook, Market};
use crate::expected_shortfall::{normal_es, normal_var, student_t_es, student_t_var};
use crate::matrix::Matrix;
use crate::returns::{ewma_cov, flag_peg_factors, sample_cov, validate_returns, ReturnsMatrix};
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
        TailDist::Normal => Ok(VarEs { var: normal_var(sig, alpha, mu)?, es: normal_es(sig, alpha, mu)? }),
        TailDist::StudentT => {
            Ok(VarEs { var: student_t_var(sig, alpha, df, mu)?, es: student_t_es(sig, alpha, df, mu)? })
        }
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
    /// Confidence level, e.g. `0.99` for 99 % VaR.
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
            alpha: 0.99,
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
    /// Confidence level used.
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
        res.flagged_peg_factors = flag_peg_factors(&rets, crate::returns::PEG_VOL_THRESHOLD);
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
    let ve = var_covar(&res.exposures, &cov.cov, opts.alpha, opts.horizon_days, opts.dist, opts.df, 0.0)?;
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

/// True if the CF expansion is monotone increasing on `[-z_range, z_range]`
/// — the validity condition for the expansion to define a quantile
/// function (Maillard 2012). Checked exactly via the closed-form minimum
/// of the (quadratic-in-`z`) derivative `dz_cf/dz` on the interval, not by
/// sampling a grid: `dz_cf/dz = 1 + zS/3 + (3z^2-3)K/24 - (6z^2-5)S^2/36 =
/// A z^2 + B z + C` with `A = K/8 - S^2/6`, `B = S/3`,
/// `C = 1 - K/8 + 5 S^2/36`, whose minimum on the interval is the vertex
/// `-B/(2A)` when convex (`A > 0`) and in range, else an endpoint (a
/// concave/linear `g`, `A <= 0`, always attains its interval minimum at an
/// endpoint). A finite-difference grid scan of `z_cf` values (the previous
/// implementation here) can under-sample a thin non-monotone dip:
/// `skew = 0.122, excess_kurtosis = -0.427` is non-monotone on `|z| <= 4`
/// (minimum derivative ~ -9e-4 near `z ~ 3.1`) but the 801-point grid
/// previously used here reported it as monotone. `n_grid` is retained for
/// API compatibility and still validated but no longer affects the result.
///
/// # Panics
/// Panics if `n_grid < 3`.
pub fn cornish_fisher_domain_ok(skew: f64, excess_kurtosis: f64, z_range: f64, n_grid: usize) -> bool {
    assert!(n_grid >= 3, "n_grid must be >= 3");
    if !skew.is_finite() || !excess_kurtosis.is_finite() || !z_range.is_finite() {
        return false;
    }
    let a = excess_kurtosis / 8.0 - skew * skew / 6.0;
    let b = skew / 3.0;
    let c = 1.0 - excess_kurtosis / 8.0 + 5.0 * skew * skew / 36.0;
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

/// Cornish-Fisher VaR (positive loss) with an explicit domain check.
///
/// # Errors
/// [`FxVarError::Invalid`] when `(skew, excess_kurtosis)` lie outside the
/// monotonicity domain and `check_domain` is true — a silently
/// non-monotone CF "quantile" can report 99% VaR below 95% VaR — or when
/// `alpha`/`horizon_days`/`sigma` are invalid.
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
    if sigma < 0.0 {
        return Err(FxVarError::invalid("sigma must be >= 0"));
    }
    if check_domain && !cornish_fisher_domain_ok(skew, excess_kurtosis, 4.0, 801) {
        return Err(FxVarError::invalid(format!(
            "Cornish-Fisher expansion is non-monotone for skew={skew}, \
             excess_kurtosis={excess_kurtosis}: outside validity domain; fall back to \
             historical or t VaR (set check_domain=false to force)"
        )));
    }
    // Loss quantile: lower tail of the P&L distribution.
    let z = inv_norm_cdf(1.0 - alpha);
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
        let ve = var_covar(&[1.0e6], &cov, 0.99, 1.0, TailDist::Normal, 6.0, 0.0).unwrap();
        let z = inv_norm_cdf(0.99);
        let sigma = 1.0e6 * 0.0001_f64.sqrt();
        assert!((ve.var - z * sigma).abs() < 1e-6);
    }

    #[test]
    fn student_t_var_exceeds_normal_at_equal_sigma() {
        let cov = Matrix::from_rows(&[vec![0.0001]]).unwrap();
        let n = var_covar(&[1.0e6], &cov, 0.99, 1.0, TailDist::Normal, 6.0, 0.0).unwrap();
        let t = var_covar(&[1.0e6], &cov, 0.99, 1.0, TailDist::StudentT, 5.0, 0.0).unwrap();
        assert!(t.var > n.var);
    }

    #[test]
    fn horizon_scaling_is_sqrt_time() {
        let cov = Matrix::from_rows(&[vec![0.0001]]).unwrap();
        let v1 = var_covar(&[1.0e6], &cov, 0.99, 1.0, TailDist::Normal, 6.0, 0.0).unwrap();
        let v10 = var_covar(&[1.0e6], &cov, 0.99, 10.0, TailDist::Normal, 6.0, 0.0).unwrap();
        assert!((v10.var - v1.var * 10f64.sqrt()).abs() < 1e-6 * v10.var);
    }

    #[test]
    fn negative_semidefinite_covariance_errors() {
        let cov = Matrix::from_rows(&[vec![-1.0]]).unwrap();
        assert!(portfolio_sigma(&[1.0], &cov).is_err());
    }

    #[test]
    fn cornish_fisher_reduces_to_normal_at_zero_moments() {
        let z = inv_norm_cdf(0.95);
        assert!((cornish_fisher_z(z, 0.0, 0.0) - z).abs() < 1e-12);
    }

    #[test]
    fn cornish_fisher_domain_rejects_extreme_moments() {
        assert!(!cornish_fisher_domain_ok(5.0, 50.0, 4.0, 801));
        assert!(cornish_fisher_domain_ok(0.1, 0.5, 4.0, 801));
    }

    #[test]
    fn cornish_fisher_var_errors_outside_domain() {
        let res = cornish_fisher_var(1.0e6, 5.0, 50.0, 0.99, 0.0, 1.0, true);
        assert!(res.is_err());
        let res2 = cornish_fisher_var(1.0e6, 5.0, 50.0, 0.99, 0.0, 1.0, false);
        assert!(res2.is_ok());
    }

    #[test]
    fn cornish_fisher_domain_check_is_exact_not_grid_resolution_dependent() {
        // Regression for the closed-form rewrite of cornish_fisher_domain_ok.
        // On this crate's default (z_range=4.0, n_grid=801), the previous
        // finite-difference-of-values grid check reported (skew,
        // excess_kurtosis) = (0.122, -0.427) as monotone -- every sampled
        // z_cf value increased -- yet the true minimum of dz_cf/dz on
        // [-4, 4] is ~ -9.15e-4 (near z ~ 3.1, between two grid nodes), so
        // the expansion is genuinely non-monotone.
        let skew = 0.122;
        let excess_kurtosis = -0.427;
        assert!(!cornish_fisher_domain_ok(skew, excess_kurtosis, 4.0, 801));
        let res = cornish_fisher_var(1.0, skew, excess_kurtosis, 0.99, 0.0, 1.0, true);
        assert!(res.is_err());
    }

    #[test]
    fn zero_factor_book_reports_zero_parametric_var() {
        let market = Market::new([("EUR", 1.10)], [("USD", 0.05), ("EUR", 0.03)]).unwrap();
        let book = Book::new(vec![Position::Spot(SpotPosition::new("EURUSD", 0.0, None))], "USD").unwrap();
        let mut data = Matrix::zeros(120, 1);
        for t in 0..120 {
            data.set(t, 0, 0.01 * (t as f64).sin());
        }
        let rets = ReturnsMatrix::new(vec!["FX:EUR".to_string()], data).unwrap();
        let r = parametric_var(&book, &market, &rets, &ParametricOptions::default()).unwrap();
        assert_eq!(r.var, 0.0);
    }
}
